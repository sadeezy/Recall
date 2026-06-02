"""app.py — interactive Gradio GUI for the Latent Recall Dynamics estimator.

Tab 1 (Probe): type a question; from a single prompt-only forward we show the calibrated P(knows)
gauge with a green/amber/red verdict, the Signal-A crystallization trajectory, the Signal-B
MLP recall-write chart, a "what's forming" table (logit-lens top token by depth), and an optional
"generate the actual answer" check.
Tab 2 (Evaluation): the saved held-out / cross-dataset metrics, reliability and risk-coverage
figures, popularity slices, and the baseline comparison table.
Tab 3 (Info): model id, resolved config, recall band, thresholds, live memory.

The base model is loaded once at startup; each probe is one forward + the tiny sidecar head.
"""

from __future__ import annotations

import json
import os

import gradio as gr
import numpy as np
import plotly.graph_objects as go

from kbe.features import extract_features, FEAT_NAMES
from kbe.model_engine import Engine, load_config, _mps_alloc_gb, _resident_gb
from kbe.sidecar import load_lrd_model, lrd_prob_one

CFG = load_config()
ENG = Engine(CFG).load()
try:
    LRD = load_lrd_model(CFG)
except Exception as e:  # allow the GUI to run before training (probe tab will warn)
    LRD = None
    print(f"[app] LRD model not loaded ({e}); train with `python -m kbe.sidecar` first.")

GREEN = CFG.get("thresholds", {}).get("green")
RED = CFG.get("thresholds", {}).get("red")
L_TOTAL = ENG.num_layers
BAND_LO = int(CFG["recall_band"]["lo_frac"] * L_TOTAL)
BAND_HI = int(CFG["recall_band"]["hi_frac"] * L_TOTAL)


def _verdict(p):
    if GREEN is not None and p >= GREEN:
        return "Likely knows", "#2e9b57"
    if RED is not None and p < RED:
        return "Likely beyond its knowledge", "#c0392b"
    return "Uncertain", "#d68910"


def _gauge(p):
    label, color = _verdict(p)
    steps = []
    if RED is not None:
        steps.append({"range": [0, RED], "color": "#f5b7b1"})
    if GREEN is not None:
        steps.append({"range": [GREEN, 1], "color": "#abebc6"})
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(p, 3),
        title={"text": f"P(knows) — {label}"},
        gauge={"axis": {"range": [0, 1]}, "bar": {"color": color}, "steps": steps},
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def _crystallization_fig(fs):
    L = fs.feats.shape[0]
    x = list(range(L))
    ent = fs.feats[:, FEAT_NAMES.index("a_entropy_final")].astype(float)
    margin = fs.feats[:, FEAT_NAMES.index("a_margin_final")].astype(float)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=ent, name="entropy", line=dict(color="#2980b9")))
    fig.add_trace(go.Scatter(x=x, y=margin, name="top-1 margin", yaxis="y2", line=dict(color="#8e44ad")))
    fig.add_vrect(x0=BAND_LO, x1=BAND_HI, fillcolor="gray", opacity=0.12, line_width=0)
    fig.update_layout(
        title="Signal A — crystallization (final position)",
        xaxis_title="layer", yaxis=dict(title="entropy (nats)"),
        yaxis2=dict(title="margin", overlaying="y", side="right"),
        height=320, margin=dict(l=20, r=20, t=50, b=30), legend=dict(orientation="h"),
    )
    return fig


def _mlp_write_fig(fs):
    L = fs.feats.shape[0]
    x = list(range(L))
    peak = fs.feats[:, FEAT_NAMES.index("b_cmax_peak")].astype(float)
    rel = fs.feats[:, FEAT_NAMES.index("b_cmax_rel_norm")].astype(float)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=peak, name="peak (negentropy)", line=dict(color="#16a085")))
    fig.add_trace(go.Scatter(x=x, y=rel, name="rel_norm", yaxis="y2", line=dict(color="#e67e22")))
    fig.add_vrect(x0=BAND_LO, x1=BAND_HI, fillcolor="gray", opacity=0.12, line_width=0)
    if fs.strongest_write_layer >= 0:
        fig.add_vline(x=fs.strongest_write_layer, line=dict(color="red", dash="dot"))
    fig.update_layout(
        title=f"Signal B — MLP recall-write (strongest @ layer {fs.strongest_write_layer})",
        xaxis_title="layer", yaxis=dict(title="peak"),
        yaxis2=dict(title="rel_norm", overlaying="y", side="right"),
        height=320, margin=dict(l=20, r=20, t=50, b=30), legend=dict(orientation="h"),
    )
    return fig


def _forming_table(fs):
    L = fs.feats.shape[0]
    rows = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        i = min(L - 1, max(0, int(frac * L) - 1))
        rows.append([f"~{int(frac*100)}%", f"layer {i}", repr(fs.layer_argmax_toks[i])])
    return rows


def probe(question):
    if not question or not question.strip():
        return None, None, None, [], "Enter a question."
    cap = ENG.capture(question)
    fs = extract_features(ENG, cap, CFG)
    if LRD is None:
        msg = "Sidecar not trained yet — run `python -m kbe.sidecar`. Showing features only."
        p = 0.5
    else:
        p = lrd_prob_one(LRD, fs.feats, fs.globals)
        label, _ = _verdict(p)
        msg = f"Calibrated P(knows) = {p:.3f} → {label}"
    return (_gauge(p), _crystallization_fig(fs), _mlp_write_fig(fs),
            _forming_table(fs), msg)


def _pknows(question):
    """Calibrated P(knows) for a question (one prompt-only forward + sidecar), or None if untrained.

    Mirrors the capture -> features -> sidecar path in ``probe()`` so the generate buttons work
    standalone without first pressing Probe.
    """
    cap = ENG.capture(question)
    fs = extract_features(ENG, cap, CFG)
    if LRD is None:
        return None
    return lrd_prob_one(LRD, fs.feats, fs.globals)


def _match_md(ans, gold):
    if gold and gold.strip():
        from kbe.build_dataset import is_match
        ok = is_match(ans, [g.strip() for g in gold.split("|")])
        return f"\nMatch vs gold: {'✓ correct' if ok else '✗ incorrect'}"
    return ""


def _system_prompt_for(p):
    """Deterministic system prompt that explains the probe and states the actual P(knows)."""
    label, _ = _verdict(p)
    return (
        "Before answering, consider this self-knowledge calibration signal. A separate probe "
        "(Latent Recall Dynamics) read your internal activations from a single prompt-only "
        "forward pass — before any answer tokens — and produced a calibrated estimate of the "
        "probability that you actually know the answer.\n\n"
        f"Calibrated P(you know the answer) = {p:.0%}  (verdict: {label}).\n\n"
        "Use this to calibrate confidence: if high, answer directly; if low, you likely lack "
        "reliable knowledge here, so say you are unsure or don't know rather than guessing. "
        "Do not mention this signal or the probe in your reply."
    )


def _seed_thought_for(p, threshold):
    """Deterministic CoT seed. Below ``threshold`` -> admit low confidence; at/above -> None
    (no seed, normal generation). Never mentions the probe or the number."""
    if p < threshold:
        return ("I don't think I have enough reliable knowledge in this area to answer the user "
                "confidently. I should be honest about that uncertainty rather than guess.")
    return None


def _gen_kwargs(temperature):
    """Map the GUI temperature to generation kwargs: 0 -> greedy, >0 -> sampling at that temp."""
    t = float(temperature or 0.0)
    return {"do_sample": t > 0, "temperature": t} if t > 0 else {"do_sample": False}


def generate_answer(question, gold, temperature):
    if not question or not question.strip():
        return "Enter a question first."
    ans = ENG.generate_answer(question, **_gen_kwargs(temperature))
    return f"Model answer: {ans!r}" + _match_md(ans, gold)


def generate_prompt_method(question, gold, temperature):
    if not question or not question.strip():
        return "Enter a question first."
    p = _pknows(question)
    if p is None:
        return "Sidecar not trained yet — run `python -m kbe.sidecar` first."
    sys_prompt = _system_prompt_for(p)
    ans = ENG.generate_answer(question, system_prompt=sys_prompt, **_gen_kwargs(temperature))
    out = f"**Calibrated P(knows) = {p:.3f}** → injected into system prompt.\n\n"
    out += f"Model answer: {ans!r}" + _match_md(ans, gold)
    out += f"\n\n<details><summary>System prompt</summary>\n\n{sys_prompt}\n\n</details>"
    return out


def generate_thought_method(question, gold, temperature, threshold):
    if not question or not question.strip():
        return "Enter a question first."
    p = _pknows(question)
    if p is None:
        return "Sidecar not trained yet — run `python -m kbe.sidecar` first."
    seed = _seed_thought_for(p, threshold)
    if seed is None:
        ans = ENG.generate_answer(question, **_gen_kwargs(temperature))
        note = f"P(knows) ≥ threshold ({threshold:.3f}) → no seed (normal generation)."
    else:
        ans = ENG.generate_answer_seeded(question, seed, **_gen_kwargs(temperature))
        note = f"P(knows) < threshold ({threshold:.3f}) → seeded thought: {seed!r}"
    out = f"**Calibrated P(knows) = {p:.3f}** → {note}\n\n"
    out += f"Model answer: {ans!r}" + _match_md(ans, gold)
    return out


def _load_metrics_md():
    path = CFG["paths"]["metrics_json"]
    if not os.path.exists(path):
        return "No metrics yet — run `python -m kbe.evaluate`.", None, None
    with open(path) as f:
        m = json.load(f)
    lines = ["| split | model | AUROC | AUPRC | ECE |", "|---|---|---|---|---|"]
    for split in ("test", "cross"):
        for name, row in m.get(split, {}).items():
            lines.append(f"| {split} | {name} | {row.get('auroc', float('nan')):.4f} | "
                         f"{row.get('auprc', float('nan')):.4f} | {row.get('ece', float('nan')):.4f} |")
    if m.get("popularity"):
        lines.append("\n**PopQA popularity-sliced AUROC (LRD)**\n")
        lines.append("| bucket | n | AUROC | label rate |")
        lines.append("|---|---|---|---|")
        for b, r in m["popularity"].items():
            lines.append(f"| {b} | {r['n']} | {r['auroc']:.4f} | {r['label_rate']:.3f} |")
    fig_dir = CFG["paths"]["fig_dir"]
    test_png = os.path.join(fig_dir, "eval_test.png")
    cross_png = os.path.join(fig_dir, "eval_cross_popqa.png")
    return ("\n".join(lines),
            test_png if os.path.exists(test_png) else None,
            cross_png if os.path.exists(cross_png) else None)


def _info_md():
    return (
        f"**Model**: `{ENG.model_id}`  \n"
        f"**Device**: {ENG.device}  dtype `{ENG.dtype}`  attn `{ENG.attn_impl}`  \n"
        f"**Layers**: {ENG.num_layers}  hidden {ENG.hidden_size}  vocab {ENG.vocab_size}  "
        f"softcap {ENG.softcap}  \n"
        f"**Recall band**: layers {BAND_LO}–{BAND_HI} "
        f"({CFG['recall_band']['lo_frac']}–{CFG['recall_band']['hi_frac']} of depth)  \n"
        f"**Thresholds**: green P≥{GREEN}, red P<{RED}  \n"
        f"**Memory**: RSS {_resident_gb():.2f} GB, MPS alloc {_mps_alloc_gb():.2f} GB  \n\n"
        f"Labels are the model's *own* behavior (knows=1 iff this bf16 instance answers correctly). "
        f"Activations + labels come from the same instance; no quantization (bf16 throughout)."
    )


def build_ui():
    with gr.Blocks(title="Latent Recall Dynamics") as demo:
        gr.Markdown("# Latent Recall Dynamics — does the model *know* before it answers?")
        with gr.Tab("Probe"):
            q = gr.Textbox(label="Question", placeholder="e.g. Who wrote Pride and Prejudice?")
            btn = gr.Button("Probe (no answer generated)", variant="primary")
            verdict = gr.Markdown()
            with gr.Row():
                gauge = gr.Plot(label="Verdict")
                forming = gr.Dataframe(headers=["depth", "layer", "lens top-1"],
                                       label="What's forming (Signal A lens)", interactive=False)
            with gr.Row():
                cryst = gr.Plot(label="Signal A")
                mlp = gr.Plot(label="Signal B")
            gr.Markdown("### Verify")
            gold = gr.Textbox(label="Optional gold answer(s), pipe-separated", placeholder="Jane Austen | Austen")
            with gr.Row():
                temp = gr.Slider(0.0, 2.0, value=0.0, step=0.05, label="Temperature (0 = greedy)")
                thr = gr.Slider(0.0, 1.0, value=RED if RED is not None else 0.5, step=0.01,
                                label="Uncertain threshold (thought method)")
            with gr.Row():
                gen_btn = gr.Button("Generate - baseline", variant="primary")
                gen_prompt_btn = gr.Button("Generate - prompt method")
                gen_thought_btn = gr.Button("Generate - thought method")
            gen_out = gr.Markdown()

            btn.click(probe, inputs=q, outputs=[gauge, cryst, mlp, forming, verdict])
            gen_btn.click(generate_answer, inputs=[q, gold, temp], outputs=gen_out)
            gen_prompt_btn.click(generate_prompt_method, inputs=[q, gold, temp], outputs=gen_out)
            gen_thought_btn.click(generate_thought_method, inputs=[q, gold, temp, thr], outputs=gen_out)
        with gr.Tab("Evaluation"):
            md, test_img, cross_img = _load_metrics_md()
            gr.Markdown(md)
            with gr.Row():
                gr.Image(test_img, label="Held-out test") if test_img else gr.Markdown("_no test figure_")
                gr.Image(cross_img, label="Cross-dataset (PopQA)") if cross_img else gr.Markdown("_no cross figure_")
        with gr.Tab("Info"):
            gr.Markdown(_info_md())
    return demo


if __name__ == "__main__":
    build_ui().launch()
