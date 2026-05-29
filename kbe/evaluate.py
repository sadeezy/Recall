"""evaluate.py — score the LRD sidecar against baselines on held-out and cross-dataset splits.

Metrics: AUROC, AUPRC, ECE (+ reliability diagram), and a risk–coverage curve. We compare:
  * LRD (the trained sidecar),
  * last-layer logistic probe,
  * global-scalar logistic regression,
  * a trivial baseline = negative final-layer entropy.

We also slice AUROC by PopQA popularity bucket. Figures are written to ``data/figs/``; a JSON
summary to ``data/metrics.json``.
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from kbe.model_engine import load_config
from kbe.sidecar import load_arrays, load_lrd_model, lrd_scores


def _auroc(y, s):
    return roc_auc_score(y, s) if len(np.unique(y)) > 1 else float("nan")


def _auprc(y, s):
    return average_precision_score(y, s) if len(np.unique(y)) > 1 else float("nan")


def ece(probs, labels, n_bins=10):
    """Expected Calibration Error with equal-width bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    e, n = 0.0, len(labels)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (probs >= lo) & (probs < hi if hi < 1 else probs <= hi)
        if m.sum() == 0:
            continue
        e += (m.sum() / n) * abs(probs[m].mean() - labels[m].mean())
    return float(e)


def reliability_points(probs, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    xs, ys = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (probs >= lo) & (probs < hi if hi < 1 else probs <= hi)
        if m.sum():
            xs.append(probs[m].mean())
            ys.append(labels[m].mean())
    return np.array(xs), np.array(ys)


def risk_coverage(probs, labels):
    """Sort by confidence; at each coverage report the error rate of the accepted set.

    Confidence = distance from 0.5 (we accept the most decisive predictions first), and a prediction
    is 'correct' if argmax(prob,0.5) matches the label.
    """
    conf = np.abs(probs - 0.5)
    order = np.argsort(-conf)
    correct = ((probs >= 0.5).astype(int) == labels)[order]
    cov = np.arange(1, len(labels) + 1) / len(labels)
    risk = 1.0 - np.cumsum(correct) / np.arange(1, len(labels) + 1)
    return cov, risk


def score_all(cfg, data, idx):
    """Return {model_name: probs} for the given example indices."""
    import joblib

    lrd = load_lrd_model(cfg)
    ll, ll_s = joblib.load(os.path.join(cfg["paths"]["ckpt_dir"], "baselines.joblib"))["lastlayer"]
    gl, gl_s = joblib.load(os.path.join(cfg["paths"]["ckpt_dir"], "baselines.joblib"))["globals"]

    gnames = data["global_names"]
    ent_col = gnames.index("a_final_entropy")
    return {
        "LRD": lrd_scores(lrd, data, idx),
        "last-layer probe": ll.predict_proba(ll_s.transform(data["final_hidden"][idx]))[:, 1],
        "global logreg": gl.predict_proba(gl_s.transform(data["globals"][idx]))[:, 1],
        "neg-entropy (trivial)": -data["globals"][idx][:, ent_col],
    }


def evaluate_split(cfg, data, idx, tag, fig_dir):
    y = data["labels"][idx]
    scores = score_all(cfg, data, idx)
    rows = {}
    for name, s in scores.items():
        rows[name] = {"auroc": _auroc(y, s), "auprc": _auprc(y, s)}
        # ECE/reliability only meaningful for probabilistic outputs in [0,1]
        if s.min() >= 0 and s.max() <= 1:
            rows[name]["ece"] = ece(s, y)

    print(f"\n=== {tag} (n={len(idx)}, label_rate={y.mean():.3f}) ===")
    print(f"{'model':<24}{'AUROC':>8}{'AUPRC':>8}{'ECE':>8}")
    for name, m in rows.items():
        print(f"{name:<24}{m['auroc']:>8.4f}{m['auprc']:>8.4f}{m.get('ece', float('nan')):>8.4f}")

    # reliability diagram (LRD) + risk-coverage (all)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    xs, ys = reliability_points(scores["LRD"], y)
    ax[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax[0].plot(xs, ys, "o-", color="tab:blue")
    ax[0].set_title(f"Reliability — LRD ({tag})")
    ax[0].set_xlabel("predicted P(knows)"); ax[0].set_ylabel("empirical accuracy")
    for name, s in scores.items():
        ss = s if (s.min() >= 0 and s.max() <= 1) else (s - s.min()) / (np.ptp(s) + 1e-9)
        cov, risk = risk_coverage(ss, y)
        ax[1].plot(cov, risk, label=name)
    ax[1].set_title(f"Risk–coverage ({tag})")
    ax[1].set_xlabel("coverage"); ax[1].set_ylabel("risk (error rate)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.2)
    fig.tight_layout()
    out = os.path.join(fig_dir, f"eval_{tag}.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"    figure -> {out}")
    return rows


def popularity_slices(cfg, data, idx, df):
    """AUROC of LRD by PopQA popularity bucket (log-popularity quartiles)."""
    sub = df.iloc[idx].reset_index(drop=True)
    mask = (sub["dataset"] == "popqa") & np.isfinite(sub["popularity"].to_numpy())
    pop_idx = np.array(idx)[mask.to_numpy()]
    if len(pop_idx) < 8:
        return {}
    lrd = load_lrd_model(cfg)
    probs = lrd_scores(lrd, data, pop_idx)
    y = data["labels"][pop_idx]
    logpop = np.log10(df.iloc[pop_idx]["popularity"].to_numpy() + 1)
    qs = np.quantile(logpop, [0.25, 0.5, 0.75])
    buckets = np.digitize(logpop, qs)
    out = {}
    print("\n=== PopQA popularity-sliced AUROC (LRD) ===")
    for b in range(4):
        m = buckets == b
        if m.sum() >= 4:
            a = _auroc(y[m], probs[m])
            out[f"bucket_{b}"] = {"n": int(m.sum()), "auroc": a, "label_rate": float(y[m].mean())}
            print(f"  bucket {b}: n={m.sum():<4} AUROC={a:.4f} label_rate={y[m].mean():.3f}")
    return out


def main():
    cfg = load_config()
    data, splits = load_arrays(cfg)
    df = pd.read_parquet(cfg["paths"]["dataset_parquet"])
    fig_dir = cfg["paths"]["fig_dir"]
    os.makedirs(fig_dir, exist_ok=True)

    summary = {}
    summary["test"] = evaluate_split(cfg, data, splits["main"]["test"], "test", fig_dir)
    summary["cross"] = evaluate_split(cfg, data, splits["cross"]["test"], "cross_popqa", fig_dir)
    summary["popularity"] = popularity_slices(cfg, data, splits["main"]["test"], df)

    with open(cfg["paths"]["metrics_json"], "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] metrics -> {cfg['paths']['metrics_json']}")

    lrd_a = summary["test"]["LRD"]["auroc"]
    probe_a = summary["test"]["last-layer probe"]["auroc"]
    verdict = "PASS" if (np.isnan(probe_a) or lrd_a >= probe_a) else "BELOW"
    print(f"[check] LRD test AUROC={lrd_a:.4f} vs last-layer probe={probe_a:.4f} -> {verdict}")


if __name__ == "__main__":
    main()
