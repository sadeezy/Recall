"""features.py — turn a prompt's captured activations into a compact, named feature set.

We consume a :class:`kbe.model_engine.Capture` (per-layer hidden states, MLP writes, and
pre-MLP residual stream) and emit:
  * ``feats``  : ``[L, F]`` float16 — a per-layer feature matrix the sidecar reads as a depth
                 sequence. F is fixed; ``feat_names`` labels every column.
  * ``globals``: ``[G]`` float32 — depth-summary scalars; ``global_names`` labels each.
  * a little metadata for the GUI (per-layer top token, strongest MLP write, etc.).

Two signal families are implemented (the plan's A & B); C & D are config-gated stubs (default off).

Signal A — crystallization trajectory (logit lens on the residual stream):
  At each layer we lens the residual stream and read how "decided" the next-token distribution is:
  entropy, top-1 logit margin, max softmax prob, and the layer-to-layer KL. A fact the model knows
  tends to collapse to low entropy / high margin early and stay put; an unknown stays diffuse and
  keeps changing its top token late into the stack.

Signal B — MLP recall-write specificity (direct logit attribution on each MLP write):
  Each layer's feed-forward sub-block adds a vector to the residual stream. We measure how big that
  write is relative to the stream (``rel_norm``), how concentrated its own vocab projection is
  (``peak`` = negentropy of the lensed write), and how aligned consecutive writes are (``coh``).
  A confident recall event shows up as a large, peaked, coherent mid-layer write.

Memory: the lens of an MLP write across all positions is ``[seq, V]`` (~20 MB here), so we stream
layer-by-layer and reduce to scalars immediately — we never materialize ``[L, seq, V]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch


# Per-layer feature columns (order matters — the sidecar reads these positionally).
# Signal B pools over *content* positions (the question/instruction tokens), excluding BOS and the
# chat-template structural tokens whose MLP writes are large but content-independent (attention sink).
FEAT_NAMES = [
    "a_entropy_final",     # Signal A: entropy of lensed residual at the final position
    "a_margin_final",      # top-1 minus top-2 logit at the final position
    "a_maxprob_final",     # max softmax prob at the final position
    "a_kl_prev_final",     # KL(p_layer || p_prev_layer) at the final position
    "a_entropy_mean",      # entropy averaged over the last N content positions
    "a_margin_mean",       # top-1 margin averaged over the last N content positions
    "a_maxprob_mean",      # max prob averaged over the last N content positions
    "b_cmax_rel_norm",     # Signal B: max over content positions of ||write|| / (||resid|| + eps)
    "b_finalpos_rel_norm", # rel_norm at the final (generation) position
    "b_cmax_peak",         # max over content positions of negentropy(softmax(lens(write)))
    "b_finalpos_peak",     # peak at the final (generation) position
    "b_mean_coh",          # mean over content positions of cos(write_layer, write_prev_layer)
]
F = len(FEAT_NAMES)

GLOBAL_NAMES = [
    "a_final_entropy",        # entropy at the last layer / final position
    "a_min_entropy",          # min final-position entropy over all layers
    "a_band_mean_entropy",    # mean final-position entropy over the recall band
    "a_settling_layer_norm",  # last layer where the final-position argmax changed, / (L-1)
    "a_mean_kl_last25",       # mean layer-to-layer KL over the last 25% of layers
    "a_max_margin",           # max final-position top-1 margin over all layers
    "b_band_max_peak",        # max MLP-write peak over the recall band
    "b_band_mean_peak",       # mean MLP-write peak over the recall band
    "b_band_max_rel_norm",    # max MLP-write rel_norm over the recall band
    "b_layer_of_max_peak",    # layer index of the band's peak write, / (L-1)
]
G = len(GLOBAL_NAMES)


@dataclass
class FeatureSet:
    feats: np.ndarray              # [L, F] float16
    globals: np.ndarray            # [G] float32
    feat_names: list = field(default_factory=lambda: list(FEAT_NAMES))
    global_names: list = field(default_factory=lambda: list(GLOBAL_NAMES))
    # ---- metadata (small; for the GUI / inspection, not necessarily fed to the sidecar) ----
    layer_argmax_ids: np.ndarray = None   # [L] int — Signal A final-pos argmax token per layer
    layer_argmax_toks: list = None        # [L] str — decoded version of the above
    strongest_write_layer: int = -1       # band layer with the peakiest MLP write
    strongest_write_token: str = ""       # decoded top token of that write
    final_hidden: np.ndarray = None       # [H] float16 — last-layer final-pos hidden (probe baseline)
    seq_len: int = 0
    final_pos: int = 0

    def flat(self) -> np.ndarray:
        """Flatten per-layer feats + globals into one vector (for the simple logreg baseline)."""
        return np.concatenate([self.feats.reshape(-1).astype(np.float32), self.globals])


def _dist_stats(logits: torch.Tensor):
    """Given logits ``[n, V]`` (float32), return per-row stats.

    Returns (entropy[n], margin[n], maxprob[n], argmax[n], logp[n, V]).
    """
    logp = torch.log_softmax(logits, dim=-1)
    p = logp.exp()
    entropy = -(p * logp).sum(dim=-1)
    top2 = logits.topk(2, dim=-1).values
    margin = top2[:, 0] - top2[:, 1]
    maxprob = p.max(dim=-1).values
    argmax = logits.argmax(dim=-1)
    return entropy, margin, maxprob, argmax, logp


def _band_indices(L: int, lo_frac: float, hi_frac: float) -> list[int]:
    lo = int(lo_frac * L)
    hi = int(hi_frac * L)
    lo = max(0, min(lo, L - 1))
    hi = max(lo, min(hi, L - 1))
    return list(range(lo, hi + 1))


@torch.no_grad()
def extract_features(eng, cap, cfg: dict) -> FeatureSet:
    """Compute Signals A & B (and optional C/D) from a Capture into a FeatureSet."""
    L = int(cap.hidden_states.shape[0])
    H = int(cap.hidden_states.shape[2])
    fp = int(cap.final_pos)
    fcfg = cfg.get("features", {})
    N = int(fcfg.get("last_content_tokens", 4))
    eps = float(fcfg.get("eps", 1e-6))
    band = _band_indices(L, cfg["recall_band"]["lo_frac"], cfg["recall_band"]["hi_frac"])
    logV = math.log(eng.vocab_size)

    # Content positions = the question/instruction tokens (no BOS / template). Fall back to all
    # positions if capture couldn't locate the span. Signal B pools over these; Signal A's "mean"
    # uses the last N of them.
    content_pos = cap.content_pos if cap.content_pos else list(range(int(cap.seq_len)))
    cpos_t = torch.tensor(content_pos, dtype=torch.long)
    last_positions = content_pos[-N:] if len(content_pos) >= 1 else [fp]

    feats = np.zeros((L, F), dtype=np.float32)
    argmax_ids = np.zeros(L, dtype=np.int64)

    prev_logp_fp = None       # Signal A: previous layer's final-pos log-probs (for KL)
    prev_write = None         # Signal B: previous layer's full MLP write [seq, H] (for coherence)
    strongest = (-1.0, -1, -1)  # (peak value, layer, position) of the band's strongest write

    hs = cap.hidden_states     # [L, seq, H] cpu float32
    mw = cap.mlp_writes
    rb = cap.resid_before

    for ell in range(L):
        # ---- Signal A: lens the residual stream at the final pos and the last-N positions ----
        h_fp = hs[ell, fp:fp + 1]                      # [1, H]
        logits_fp = eng.lens_logits(h_fp)              # [1, V] on device
        e_f, m_f, mp_f, am_f, logp_fp = _dist_stats(logits_fp)
        argmax_ids[ell] = int(am_f.item())

        kl = 0.0
        if prev_logp_fp is not None:
            p_fp = logp_fp.exp()
            kl = float((p_fp * (logp_fp - prev_logp_fp)).sum().item())
        prev_logp_fp = logp_fp.detach()

        h_last = hs[ell, last_positions]               # [n, H]
        e_l, m_l, mp_l, _, _ = _dist_stats(eng.lens_logits(h_last))

        feats[ell, 0] = e_f.item()
        feats[ell, 1] = m_f.item()
        feats[ell, 2] = mp_f.item()
        feats[ell, 3] = kl
        feats[ell, 4] = e_l.mean().item()
        feats[ell, 5] = m_l.mean().item()
        feats[ell, 6] = mp_l.mean().item()

        # ---- Signal B: direct logit attribution on this layer's MLP write ----
        write = mw[ell]                                # [seq, H]
        resid = rb[ell]
        wnorm = write.norm(dim=-1)                     # [seq]
        rnorm = resid.norm(dim=-1)
        rel = wnorm / (rnorm + eps)

        write_logits = eng.lens_logits(write)          # [seq, V] on device
        w_logp = torch.log_softmax(write_logits, dim=-1)
        w_ent = -(w_logp.exp() * w_logp).sum(dim=-1)   # [seq]
        peak = (logV - w_ent).cpu()                    # negentropy per position [seq]

        # coherence with previous layer's write, over content positions only
        if prev_write is not None:
            coh = torch.cosine_similarity(write[cpos_t], prev_write[cpos_t], dim=-1)
            mean_coh = float(coh.mean().item())
        else:
            mean_coh = 0.0
        prev_write = write

        rel_c = rel[cpos_t]
        peak_c = peak[cpos_t]
        feats[ell, 7] = float(rel_c.max().item())
        feats[ell, 8] = float(rel[fp].item())
        feats[ell, 9] = float(peak_c.max().item())
        feats[ell, 10] = float(peak[fp].item())
        feats[ell, 11] = mean_coh

        if ell in band:
            j = int(peak_c.argmax().item())
            pk_pos = content_pos[j]
            pk_val = float(peak_c[j].item())
            if pk_val > strongest[0]:
                # decode the top token of the strongest content-position write
                tok_id = int(write_logits[pk_pos].argmax().item())
                strongest = (pk_val, ell, tok_id)

        del logits_fp, write_logits, w_logp
    eng._empty_cache()

    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- global summary scalars ----
    ent_final = feats[:, 0]
    margin_final = feats[:, 1]
    kl_final = feats[:, 3]
    peak_per_layer = feats[:, 9]
    relnorm_per_layer = feats[:, 7]

    # settling layer: last layer whose final-pos argmax differs from the previous layer's
    changed = [ell for ell in range(1, L) if argmax_ids[ell] != argmax_ids[ell - 1]]
    settling = (max(changed) / (L - 1)) if changed else 0.0
    last25 = list(range(int(0.75 * L), L))

    glob = np.array([
        ent_final[-1],
        float(ent_final.min()),
        float(ent_final[band].mean()),
        settling,
        float(kl_final[last25].mean()) if last25 else 0.0,
        float(margin_final.max()),
        float(peak_per_layer[band].max()),
        float(peak_per_layer[band].mean()),
        float(relnorm_per_layer[band].max()),
        (strongest[1] / (L - 1)) if strongest[1] >= 0 else 0.0,
    ], dtype=np.float32)
    glob = np.nan_to_num(glob, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- optional Signals C / D (config-gated; default off) ----
    extra_globals = {}
    scfg = cfg.get("signals", {})
    if scfg.get("c", False):
        extra_globals["c_kl"] = _signal_c(eng, cap, cfg)
    if scfg.get("d", False):
        extra_globals["d_maha"] = _signal_d(eng, cap, cfg)

    tok = eng.tokenizer
    fs = FeatureSet(
        feats=feats.astype(np.float16),
        globals=glob,
        layer_argmax_ids=argmax_ids,
        layer_argmax_toks=[tok.decode([int(i)]) for i in argmax_ids],
        strongest_write_layer=int(strongest[1]),
        strongest_write_token=(tok.decode([int(strongest[2])]) if strongest[2] >= 0 else ""),
        final_hidden=hs[-1, fp].numpy().astype(np.float16),
        seq_len=int(cap.seq_len),
        final_pos=fp,
    )
    if extra_globals:
        fs.extra = extra_globals  # type: ignore[attr-defined]
    return fs


# --------------------------------------------------------------------------- Signal C / D stubs
@torch.no_grad()
def _signal_c(eng, cap, cfg) -> float:
    """In-pass perturbation robustness (scaffold).

    Inject Gaussian noise (sigma = c_sigma * residual RMS) into a mid-band residual and re-read the
    final-position lens k times; return mean KL vs the unperturbed distribution. Not wired into v1
    training. Returns 0.0 as a placeholder until enabled and validated.
    """
    return 0.0


@torch.no_grad()
def _signal_d(eng, cap, cfg) -> float:
    """Mahalanobis familiarity (scaffold).

    Distance of a chosen layer's final-pos hidden state to the fitted ``knows=1`` cluster (mean +
    shrinkage covariance estimated at train time). Requires fitted statistics; returns 0.0 until
    those exist. Not wired into v1 training.
    """
    return 0.0
