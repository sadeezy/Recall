# Latent Recall Dynamics (LRD) — a pre-generation knowledge-boundary estimator

**Question:** *Does the model know the answer before it starts writing it?*

LRD reads a single **prompt-only forward pass** of `google/gemma-4-E4B-it` (no answer tokens are
generated) and predicts whether the model's parametric knowledge is sufficient to answer the
question correctly. A tiny "sidecar" estimator consumes mechanistic features extracted across depth
and emits a calibrated **P(knows)**.

The label is the model's **own behavior**: `knows = 1` iff *this exact bf16 instance* answers the
question correctly (matched against gold answers + aliases). Activations and labels come from the
**same** model instance — so the estimator predicts what the model will actually do, not world-truth.

---

## Method — three signal families → a small sidecar

- **Signal A — crystallization trajectory.** Apply the logit lens (final RMSNorm → tied output
  embedding → `tanh` softcap 30.0) at the final prompt position at *every* layer. Track entropy,
  top-1 margin, inter-layer KL, and the settling layer. Known facts crystallize early; unknown ones
  stay diffuse.
- **Signal B — MLP recall-write specificity (centerpiece).** Direct logit attribution on each
  layer's MLP write (the additive feed-forward contribution to the residual stream). Measure its
  relative norm, peakedness (negentropy of its vocab projection), and directional coherence across
  the mid-layer "recall band". A confident factual recall shows a strong, peaked, coherent write.
- **Signal C / D** — perturbation robustness and Mahalanobis familiarity. Scaffolded behind config
  flags, **default off**; not used by the v1 sidecar.

The sidecar is a small GRU over depth (`[L, F]` → mean+last pool) concatenated with global scalars
→ MLP → one logit, trained with class-balanced BCE and temperature-scaled for calibration
(~5×10⁴ params).

---

## Hard constraints (by design)

- **Apple Silicon / MPS only.** No CUDA assumptions anywhere. Peak memory stays under ~24 GB.
- **bf16 throughout** (`attn_implementation="eager"`, `PYTORCH_ENABLE_MPS_FALLBACK=1`). Not fp16.
- **No GGUF / llama.cpp / Ollama.** Activations *and* labels must come from one bf16 instance, so a
  quantized runtime is never used for extraction or labeling.
- **Features are stored, never raw activations.** Each capture is detached → moved to CPU → cast to
  float16 → freed immediately.
- Model dimensions are read from `model.config` at runtime (42 layers, hidden 2560, vocab 262144,
  softcap 30.0 for this checkpoint) — nothing is hardcoded. Thinking mode is **off**.

---

## Setup

This project **reuses the existing system Python 3.12 environment**. The only missing dependency is
`plotly`:

```bash
pip install --user plotly
```

`requirements.txt` lists the full set with the versions present at build time, for reproducibility.

You also need access to the gated `google/gemma-4-E4B-it` checkpoint:

```bash
huggingface-cli login        # once, with a token that has access to the Gemma 4 weights
```

The ~15 GB bf16 checkpoint is loaded once and runs on MPS.

---

## Run order

```bash
# 1. Smoke test the full pipeline on 100 examples first (fast sanity check)
python -m kbe.build_dataset --limit 100

# 2. Build the real dataset (~3000 balanced PopQA + TriviaQA examples; ~2–3 h on Apple Silicon)
python -m kbe.build_dataset

# 3. Train the LRD sidecar + baselines; writes calibrated thresholds back into config.yaml
python -m kbe.sidecar

# 4. Evaluate vs baselines on held-out + cross-dataset splits; writes figures + metrics.json
python -m kbe.evaluate

# 5. Launch the interactive GUI
python -m kbe.app
```

Each stage reads `kbe/config.yaml` (model id, recall band, signal flags, dataset sizes, thresholds)
and writes its outputs under `data/` and `ckpts/`.

---

## What you get

- **`build_dataset`** → `data/dataset.parquet` (one row per example: id, dataset, question,
  popularity, label, prediction, strongest-write token) and `data/feats.npz` (aligned feature
  arrays), plus stratified and cross-dataset splits in `data/splits.json`.
- **`sidecar`** → trained `ckpts/lrd.pt` + `ckpts/baselines.joblib`, and green/red verdict
  thresholds in `config.yaml`.
- **`evaluate`** → AUROC / AUPRC / ECE, reliability diagram, risk–coverage curve, PopQA
  popularity-sliced AUROC, and a baseline-comparison table (LRD vs last-layer probe vs global-scalar
  logreg vs negative-entropy trivial baseline). Figures land in `data/figs/`, summary in
  `data/metrics.json`.
- **`app`** (Gradio + Plotly) — three tabs:
  - **Probe**: type a question → calibrated P(knows) gauge with a green/amber/red verdict, the
    Signal-A crystallization trajectory, the Signal-B MLP recall-write chart (strongest layer
    highlighted), and a "what's forming" logit-lens table. Two **generate** buttons then check the
    actual answer (optionally matched against pipe-separated gold answers):
    - **baseline** — plain greedy generation, with no confidence signal injected.
    - **thought seed** — feeds the model its *own* calibrated P(knows) back into an open thinking
      channel. The seed states the probability as a percent and, graded by adjustable low/high band
      cutoffs (default 0.4 / 0.7), directs the model to either answer confidently, lead with a hedge,
      or admit it doesn't know; the model then finishes the seeded thought and emits its answer.
  - **Evaluation**: the saved held-out / cross-dataset metrics and figures.
  - **Info**: resolved config, recall band, thresholds, and live memory.

---

## Repository layout

```
kbe/
  config.yaml        resolved run configuration (model, band, flags, thresholds)
  model_engine.py    Engine: load on MPS, prompt-only capture with MLP-write hooks, logit lens, generate (baseline + thought-seeded)
  features.py        Signals A & B → per-layer feature matrix [L,F] + global scalars
  build_dataset.py   capture features + self-behavioral labels for PopQA + TriviaQA
  sidecar.py         LRD GRU estimator + logistic baselines + calibration + thresholds
  evaluate.py        metrics, reliability / risk-coverage figures, baseline comparison
  app.py             Gradio + Plotly GUI
data/                datasets, features, splits, figures, metrics (gitignored)
ckpts/               trained sidecar + baselines (gitignored)
```

---

## Caveats

- PopQA targets long-tail entities, so this model's PopQA accuracy is genuinely low — that low
  base-rate is the *point*: it gives the estimator a hard "unknown" distribution to separate from
  the easier TriviaQA "known" distribution.
- The raw logit lens of an MLP *write* reads noisy mid-stack tokens (an untuned-lens artifact); it
  is the write **magnitudes** (peak, relative norm) that carry the usable Signal-B information. The
  GUI's "what's forming" table uses the Signal-A hidden-state lens, which is well-behaved.
- The first model forward after load is warmed up internally to avoid an MPS first-pass logit
  artifact.
