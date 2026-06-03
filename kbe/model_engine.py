"""model_engine.py — load Gemma-4-E4B-it and capture prompt-only activations.

The Engine loads the base model once (bf16 on MPS), exposes a logit-lens helper, and a
``capture`` method that runs a single prompt-only forward while recording, per decoder layer:
  * the MLP sub-block write (the additive vector the feed-forward block contributes to the
    residual stream), used by Signal B;
  * the residual stream just before the MLP sub-block (the rel-norm denominator);
  * the full residual stream after each layer (used by Signal A's logit lens).

Capture choices specific to Gemma-4-E4B-it (verified against transformers' modeling_gemma4.py):
  * The checkpoint is the multimodal ``Gemma4ForConditionalGeneration``; the text decoder lives
    at ``model.model.language_model`` (``layers`` + final ``norm``). ``AutoModelForCausalLM``
    returns this multimodal class. The text-only forward path is exercised by passing only
    ``input_ids`` (no pixel/audio inputs), so we drop the vision/audio towers to save memory.
  * ``enable_moe_block`` is False, so the MLP write equals the *output of*
    ``decoder_layer.post_feedforward_layernorm`` (added to the residual at the line right after,
    before the per-layer-embedding injection and the final ``layer_scalar`` multiply). We capture
    that module's output directly — it already includes the post-feed-forward norm.
  * ``resid_before`` (rel-norm denominator) is the *input to*
    ``decoder_layer.pre_feedforward_layernorm`` (the post-attention residual stream).
  * The lens applies the model's final RMSNorm then the tied output embedding, then the tanh
    ``final_logit_softcapping`` (30.0). This matches the model's own final logit computation.
"""

from __future__ import annotations

import os
import re
import contextlib
from dataclasses import dataclass

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_CFG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def load_config(path: str | None = None) -> dict:
    import yaml

    with open(path or _CFG_PATH) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# Special-token strings used by Gemma-4's chat template (for defensive stripping).
_THOUGHT_RE = re.compile(r"<\|channel>.*?<channel\|>", re.DOTALL)
_SPECIAL_RE = re.compile(r"<\|?(?:turn|think|channel|tool[^>]*)\|?>|<turn\|>|<channel\|>")


def pick_device(requested: str = "mps") -> str:
    if requested == "mps" and torch.backends.mps.is_available():
        return "mps"
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _resident_gb() -> float:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 1e9
    except Exception:
        return float("nan")


def _mps_alloc_gb() -> float:
    try:
        return torch.mps.current_allocated_memory() / 1e9
    except Exception:
        return float("nan")


@dataclass
class Capture:
    """Raw per-prompt captures, all CPU float32 (small — a few MB)."""

    hidden_states: torch.Tensor   # [L, seq, H]  residual stream after each decoder layer
    mlp_writes: torch.Tensor      # [L, seq, H]  MLP sub-block write per layer
    resid_before: torch.Tensor    # [L, seq, H]  residual stream before the MLP sub-block
    input_ids: torch.Tensor       # [seq]
    final_pos: int                # index of the last non-padding token
    seq_len: int
    content_pos: list | None = None  # positions of the question/instruction tokens (no template/BOS)


def _find_subsequence(haystack: list[int], needle: list[int]) -> tuple[int, int] | None:
    """Return (start, end) of the first occurrence of ``needle`` in ``haystack``, else None."""
    n, m = len(haystack), len(needle)
    if m == 0 or m > n:
        return None
    for i in range(n - m + 1):
        if haystack[i:i + m] == needle:
            return i, i + m
    return None


class Engine:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        mcfg = cfg["model"]
        self.model_id = mcfg["id"]
        self.device = pick_device(mcfg.get("device", "mps"))
        self.dtype = getattr(torch, mcfg.get("dtype", "bfloat16"))
        self.attn_impl = mcfg.get("attn_implementation", "eager")
        self.drop_towers = mcfg.get("drop_multimodal_towers", True)
        self.context_window = mcfg.get("context_window")  # optional cap; falls back to model native
        self.enable_thinking = cfg.get("prompt", {}).get("enable_thinking", False)
        self.answer_instruction = cfg.get("prompt", {}).get("answer_instruction", "")

        self.tokenizer = None
        self.model = None
        self.text_model = None      # Gemma4TextModel (has .layers and .norm)
        self.lm_head = None         # nn.Linear, weight tied to embed_tokens (= W_U)
        self.softcap = None         # final_logit_softcapping, or None
        self.num_layers = None
        self.hidden_size = None
        self.vocab_size = None
        self.max_context = None

    # ----------------------------------------------------------------- load
    def load(self) -> "Engine":
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id, dtype=self.dtype, attn_implementation=self.attn_impl
            )
        except Exception as e:  # surface auth/gating clearly
            msg = str(e).lower()
            if any(k in msg for k in ("gated", "401", "403", "authoriz", "token", "login")):
                raise RuntimeError(
                    f"Could not load {self.model_id}. This model may require HuggingFace auth. "
                    "Run `huggingface-cli login` or set HF_TOKEN, then retry.\nOriginal error: " + str(e)
                ) from e
            raise

        model.to(self.device)
        model.eval()
        torch.set_grad_enabled(False)
        self.model = model

        self.text_model, self.lm_head = self._locate_text_modules(model)
        tcfg = self.text_model.config
        self.num_layers = tcfg.num_hidden_layers
        self.hidden_size = tcfg.hidden_size
        self.vocab_size = tcfg.vocab_size
        self.softcap = getattr(tcfg, "final_logit_softcapping", None)
        model_ctx = getattr(tcfg, "max_position_embeddings", 8192)
        # Honour an explicit context_window from config (bounds the GUI generate budget); never
        # exceed what the model actually supports.
        self.max_context = min(self.context_window, model_ctx) if self.context_window else model_ctx

        if self.drop_towers:
            self._drop_multimodal_towers(model)
            self._empty_cache()

        self._warmup()
        self._print_load_report(tcfg)
        return self

    @torch.no_grad()
    def _warmup(self):
        """Run throwaway forwards. On this MPS build the *first* forward after load returns
        wrong logits/hidden states (a one-time lazy-init/kernel-compile artifact); every forward
        afterward is correct and matches generate(). We warm a couple of lengths to be safe."""
        for q in ("Hello.", "What is the capital of France please tell me now?"):
            enc = self.tokenizer(self.format_prompt(q), return_tensors="pt", add_special_tokens=False).to(self.device)
            self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], use_cache=True)
        self._empty_cache()

    @staticmethod
    def _locate_text_modules(model):
        """Return (text_model, lm_head), handling multimodal vs text-only layouts."""
        base = getattr(model, "model", model)
        text_model = getattr(base, "language_model", base)  # multimodal nests under language_model
        lm_head = model.get_output_embeddings()
        if lm_head is None:
            lm_head = getattr(model, "lm_head", None)
        assert hasattr(text_model, "layers") and hasattr(text_model, "norm"), (
            "Could not locate text decoder layers/norm on the loaded model."
        )
        return text_model, lm_head

    @staticmethod
    def _drop_multimodal_towers(model):
        base = getattr(model, "model", None)
        if base is None:
            return
        for attr in ("vision_tower", "audio_tower", "embed_vision", "embed_audio"):
            if getattr(base, attr, None) is not None:
                setattr(base, attr, None)

    def _empty_cache(self):
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()

    def _print_load_report(self, tcfg):
        print(f"[Engine] loaded {self.model_id} on {self.device} ({self.dtype})")
        print(
            f"[Engine] num_hidden_layers={tcfg.num_hidden_layers} hidden_size={tcfg.hidden_size} "
            f"intermediate_size={getattr(tcfg, 'intermediate_size', '?')} vocab_size={tcfg.vocab_size}"
        )
        print(
            f"[Engine] hidden_size_per_layer_input={getattr(tcfg, 'hidden_size_per_layer_input', '?')} "
            f"final_logit_softcapping={self.softcap}"
        )
        print(f"[Engine] resident RSS={_resident_gb():.2f} GB  MPS alloc={_mps_alloc_gb():.2f} GB")

    # ----------------------------------------------------------------- lens
    def lens_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Logit lens: final RMSNorm -> tied output embedding -> tanh softcap.

        Accepts ``h`` of shape ``[..., H]`` on any device; returns float32 logits ``[..., V]``
        on the model device. Used for Signal A (hidden states) and Signal B (MLP-write peakedness).
        """
        h = h.to(self.device, dtype=self.dtype)
        normed = self.text_model.norm(h)
        logits = self.lm_head(normed)
        if self.softcap is not None:
            c = float(self.softcap)
            logits = c * torch.tanh(logits / c)
        return logits.float()

    # -------------------------------------------------------------- prompt
    def format_prompt(self, question: str, add_instruction: bool = True,
                      system_prompt: str | None = None,
                      enable_thinking: bool | None = None) -> str:
        text = question.strip()
        if add_instruction and self.answer_instruction:
            text = text + self.answer_instruction
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": text})
        return self.tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking if enable_thinking is None else enable_thinking,
        )

    @staticmethod
    def strip_thinking(text: str) -> str:
        text = _THOUGHT_RE.sub("", text)
        text = _SPECIAL_RE.sub("", text)
        return text.strip()

    def _content_positions(self, ids: list[int], inner_text: str) -> list | None:
        """Locate the question(+instruction) token span inside the templated prompt.

        Used to pool Signal B over genuine content tokens and exclude the BOS / chat-template
        structural tokens, whose MLP writes are large but content-independent (attention sink).
        Returns a list of positions, or None if the span can't be matched (caller falls back to
        all positions).
        """
        inner = self.tokenizer(inner_text, add_special_tokens=False)["input_ids"]
        span = _find_subsequence(ids, inner)
        if span is None and len(inner) > 2:
            span = _find_subsequence(ids, inner[1:-1])  # tolerate boundary re-tokenization
        return list(range(span[0], span[1])) if span else None

    # -------------------------------------------------------------- capture
    @contextlib.contextmanager
    def _capture_hooks(self, store: dict):
        handles = []
        layers = self.text_model.layers

        def mk_layer_out(i):
            def hook(_m, _inp, out):
                t = out[0] if isinstance(out, tuple) else out
                store["hidden"][i] = t.detach()[0]  # [seq, H]
            return hook

        def mk_resid_before(i):
            def hook(_m, args, _kwargs=None):
                t = args[0]
                store["resid_before"][i] = t.detach()[0]
            return hook

        def mk_mlp_write(i):
            def hook(_m, _inp, out):
                t = out[0] if isinstance(out, tuple) else out
                store["mlp"][i] = t.detach()[0]
            return hook

        for i, layer in enumerate(layers):
            handles.append(layer.register_forward_hook(mk_layer_out(i)))
            handles.append(layer.pre_feedforward_layernorm.register_forward_pre_hook(mk_resid_before(i)))
            handles.append(layer.post_feedforward_layernorm.register_forward_hook(mk_mlp_write(i)))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    @torch.no_grad()
    def capture(self, prompt_text: str, is_chat: bool = False) -> Capture:
        """Single prompt-only forward; returns CPU float32 captures."""
        text = prompt_text if is_chat else self.format_prompt(prompt_text)
        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(self.device)
        attn = enc.get("attention_mask")
        seq = input_ids.shape[1]
        L = self.num_layers

        store = {"hidden": [None] * L, "resid_before": [None] * L, "mlp": [None] * L}
        with self._capture_hooks(store):
            # use_cache=True is REQUIRED: on this Gemma-4 build the no-cache forward constructs a
            # different attention mask and yields wrong logits/hidden states, whereas the cached
            # path exactly reproduces generate()'s scores. We discard the cache; we only need the
            # one-shot prompt activations to match the model's real generation behavior.
            self.model(input_ids=input_ids, attention_mask=attn.to(self.device) if attn is not None else None,
                       use_cache=True)

        def stack(key):
            return torch.stack([t.float().cpu() for t in store[key]], dim=0)  # [L, seq, H]

        ids_list = input_ids[0].tolist()
        content_pos = None
        if not is_chat:
            # Locate the *question* tokens only; the appended answer_instruction is constant
            # boilerplate and would just dilute the content pools with content-independent writes.
            content_pos = self._content_positions(ids_list, prompt_text.strip())

        cap = Capture(
            hidden_states=stack("hidden"),
            mlp_writes=stack("mlp"),
            resid_before=stack("resid_before"),
            input_ids=input_ids[0].cpu(),
            final_pos=seq - 1,
            seq_len=seq,
            content_pos=content_pos,
        )
        del store, input_ids
        self._empty_cache()
        return cap

    # ------------------------------------------------------------ generate
    @staticmethod
    def _first_line(text: str) -> str:
        for line in text.splitlines():
            if line.strip():
                return line.strip()
        return text.strip()

    def _answer_after_thought(self, raw: str) -> str:
        """Extract the post-thinking answer from a decoded generation. With thinking enabled the
        answer follows the thought's closing ``<channel|>`` (the opening ``<|channel>`` is supplied
        by the chat template / seed and so isn't in the decoded output). Split on the close when it's
        present; otherwise fall back to the regex strip (model answered without a closed thought).

        Returns the full answer with line breaks preserved (only outer whitespace trimmed) — the
        caller gets the complete multi-line response, not just its first line."""
        if "<channel|>" in raw:
            raw = raw.split("<channel|>")[-1]
        return self.strip_thinking(raw)

    def _ctx_budget(self, prompt_len: int, requested: int) -> int:
        """Clamp a requested ``max_new_tokens`` so prompt+new never exceeds the context window. The
        GUI passes ``max_context`` to mean "no limit" — this turns that into "fill the remaining
        context", letting generation stop at EOS rather than at an arbitrary token cap."""
        return max(min(requested, self.max_context - prompt_len), 1)

    @torch.no_grad()
    def generate_answer(self, question: str, max_new_tokens: int | None = None,
                        do_sample: bool = False, temperature: float = 0.7,
                        system_prompt: str | None = None,
                        enable_thinking: bool | None = None) -> str:
        gcfg = self.cfg.get("generation", {})
        thinking = self.enable_thinking if enable_thinking is None else enable_thinking
        max_new = max_new_tokens or gcfg.get("max_new_tokens", 32)
        stop_ids = gcfg.get("stop_token_ids", [1, 106])
        text = self.format_prompt(question, system_prompt=system_prompt, enable_thinking=enable_thinking)
        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.device)
        out = self.model.generate(
            **enc,
            max_new_tokens=self._ctx_budget(enc["input_ids"].shape[1], max_new),
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            eos_token_id=stop_ids,
            pad_token_id=self.tokenizer.pad_token_id or 0,
        )
        new_tokens = out[0, enc["input_ids"].shape[1]:]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        return self._answer_after_thought(raw) if thinking else self._first_line(self.strip_thinking(raw))

    @torch.no_grad()
    def generate_answer_seeded(self, question: str, seed_thought: str,
                               max_new_tokens: int | None = None,
                               do_sample: bool = False, temperature: float = 0.7) -> str:
        """Thought method: enable thinking and prefill the open thought channel with ``seed_thought``,
        then generate. The model continues the seeded thought, closes it with ``<channel|>``, and
        emits the answer; we keep only the text after the channel-close.

        The budget defaults to filling the context window (``_ctx_budget``) so the model can finish
        thinking *and* answer without truncation. The seed's opening ``<|channel>`` lives in the
        prompt, so the close-tag split in ``_answer_after_thought`` (not ``strip_thinking``'s regex)
        is what recovers the answer.
        """
        gcfg = self.cfg.get("generation", {})
        max_new = max_new_tokens or gcfg.get("thinking_max_new_tokens", 256)
        stop_ids = gcfg.get("stop_token_ids", [1, 106])
        prompt = self.format_prompt(question, enable_thinking=True) + "<|channel>thought\n" + seed_thought.strip()
        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        out = self.model.generate(
            **enc,
            max_new_tokens=self._ctx_budget(enc["input_ids"].shape[1], max_new),
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            eos_token_id=stop_ids,
            pad_token_id=self.tokenizer.pad_token_id or 0,
        )
        new_tokens = out[0, enc["input_ids"].shape[1]:]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        return self._answer_after_thought(raw)
