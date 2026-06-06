"""sidecar.py — train the LRD knowledge-boundary estimator and the comparison baselines.

Inputs (from build_dataset.py): ``data/feats.npz`` (feats [N,L,F], globals [N,G], final_hidden
[N,H], labels) and ``data/splits.json``.

Models:
  1. **LRD** (primary): reads the per-layer feature matrix [L,F] as a depth sequence with a small
     GRU, mean+last pools it, concatenates the standardized global scalars, and an MLP emits one
     logit. Trained with class-balanced BCE, early-stopped on validation AUROC.
  2. **Last-layer probe** (baseline): logistic regression on the final-layer final-position hidden.
  3. **Global-scalar logreg** (baseline): logistic regression on the global scalars.

After training we temperature-scale the LRD logits on the validation split and derive green/red
decision thresholds from validation precision targets, writing them back into config.yaml.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from kbe.model_engine import load_config, pick_device, set_seed


# ----------------------------------------------------------------- data
def load_arrays(cfg: dict):
    paths = cfg["paths"]
    npz = np.load(paths["feats_npz"], allow_pickle=True)
    with open(paths["splits_json"]) as f:
        splits = json.load(f)
    return {
        "feats": npz["feats"].astype(np.float32),          # [N, L, F]
        "globals": npz["globals"].astype(np.float32),      # [N, G]
        "final_hidden": npz["final_hidden"].astype(np.float32),  # [N, H]
        "labels": npz["labels"].astype(np.int64),          # [N]
        "feat_names": list(npz["feat_names"]),
        "global_names": list(npz["global_names"]),
    }, splits


class _PerFeatScaler:
    """Standardize [N, L, F] per feature column (shared across layers), fit on train rows."""

    def fit(self, x):  # x: [N, L, F]
        self.mean = x.mean(axis=(0, 1), keepdims=True)
        self.std = x.std(axis=(0, 1), keepdims=True) + 1e-6
        return self

    def transform(self, x):
        return (x - self.mean) / self.std

    def state(self):
        return {"mean": self.mean, "std": self.std}


# ----------------------------------------------------------------- model
class LRDNet(nn.Module):
    def __init__(self, n_feat: int, n_global: int, hidden: int = 96):
        super().__init__()
        self.gru = nn.GRU(input_size=n_feat, hidden_size=hidden, batch_first=True)
        head_in = hidden * 2 + n_global  # mean-pool + last-step + globals
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, 1)
        )

    def forward(self, seq, glob):           # seq: [B, L, F]   glob: [B, G]
        out, h_last = self.gru(seq)         # out: [B, L, H]   h_last: [1, B, H]
        pooled = torch.cat([out.mean(dim=1), h_last[-1], glob], dim=-1)
        return self.head(pooled).squeeze(-1)  # [B]


def _auroc(y, score):
    return roc_auc_score(y, score) if len(np.unique(y)) > 1 else float("nan")


# ----------------------------------------------------------------- LRD training
def train_lrd(data, splits, cfg):
    tcfg = cfg["train"]
    set_seed(tcfg["seed"])
    device = pick_device(tcfg.get("device", "auto"))
    tr, va = splits["main"]["train"], splits["main"]["val"]

    fscaler = _PerFeatScaler().fit(data["feats"][tr])
    gscaler = StandardScaler().fit(data["globals"][tr])

    def batch(idx):
        s = torch.tensor(fscaler.transform(data["feats"][idx]), dtype=torch.float32, device=device)
        g = torch.tensor(gscaler.transform(data["globals"][idx]), dtype=torch.float32, device=device)
        y = torch.tensor(data["labels"][idx], dtype=torch.float32, device=device)
        return s, g, y

    s_tr, g_tr, y_tr = batch(tr)
    s_va, g_va, y_va = batch(va)

    net = LRDNet(data["feats"].shape[2], data["globals"].shape[1], tcfg["hidden"]).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    pos = float(y_tr.sum().item())
    neg = float(len(y_tr) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(net.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])

    bs = tcfg["batch_size"]
    best = {"auroc": -1.0, "state": None, "epoch": -1}
    patience, since = tcfg["patience"], 0
    rng = np.random.default_rng(tcfg["seed"])

    for epoch in range(tcfg["epochs"]):
        net.train()
        order = rng.permutation(len(tr))
        for i in range(0, len(order), bs):
            bi = order[i:i + bs]
            opt.zero_grad()
            loss = crit(net(s_tr[bi], g_tr[bi]), y_tr[bi])
            loss.backward()
            opt.step()

        net.eval()
        with torch.no_grad():
            va_score = torch.sigmoid(net(s_va, g_va)).cpu().numpy()
        auroc = _auroc(data["labels"][va], va_score)
        if auroc > best["auroc"]:
            best = {"auroc": auroc, "state": {k: v.cpu().clone() for k, v in net.state_dict().items()},
                    "epoch": epoch}
            since = 0
        else:
            since += 1
            if since >= patience:
                break

    net.load_state_dict(best["state"])
    net.eval()

    # temperature scaling on val (1-param NLL minimization)
    with torch.no_grad():
        va_logits = net(s_va, g_va).cpu()
    T = _fit_temperature(va_logits, torch.tensor(data["labels"][va], dtype=torch.float32))

    print(f"[LRD] params={n_params} best val AUROC={best['auroc']:.4f} @epoch {best['epoch']} T={T:.3f}")
    return {
        "net": net, "fscaler": fscaler, "gscaler": gscaler, "T": T, "device": device,
        "val_auroc": best["auroc"], "n_params": n_params,
    }


def _fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float:
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=60)
    crit = nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = crit(logits / logT.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(logT.exp().item())


def load_lrd_model(cfg: dict) -> dict:
    """Rebuild a trained LRD model (net + scalers + temperature) from ckpts/lrd.pt."""
    device = pick_device(cfg["train"].get("device", "auto"))
    ck = torch.load(os.path.join(cfg["paths"]["ckpt_dir"], "lrd.pt"), map_location=device, weights_only=False)
    net = LRDNet(ck["n_feat"], ck["n_global"], ck["hidden"]).to(device)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    fscaler = _PerFeatScaler()
    fscaler.mean, fscaler.std = ck["fscaler"]["mean"], ck["fscaler"]["std"]
    gscaler = StandardScaler()
    gscaler.mean_, gscaler.scale_ = ck["gscaler"]["mean"], ck["gscaler"]["scale"]
    return {"net": net, "fscaler": fscaler, "gscaler": gscaler, "T": ck["T"], "device": device,
            "feat_names": ck["feat_names"], "global_names": ck["global_names"]}


def lrd_scores(model: dict, data, idx) -> np.ndarray:
    net, dev = model["net"], model["device"]
    s = torch.tensor(model["fscaler"].transform(data["feats"][idx]), dtype=torch.float32, device=dev)
    g = torch.tensor(model["gscaler"].transform(data["globals"][idx]), dtype=torch.float32, device=dev)
    net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(s, g) / model["T"]).cpu().numpy()


def lrd_prob_one(model: dict, feats: np.ndarray, globals_: np.ndarray) -> float:
    """Calibrated P(knows) for a single example's [L,F] feats + [G] globals."""
    net, dev = model["net"], model["device"]
    s = torch.tensor(model["fscaler"].transform(feats[None].astype(np.float32)), dtype=torch.float32, device=dev)
    g = torch.tensor(model["gscaler"].transform(globals_[None].astype(np.float32)), dtype=torch.float32, device=dev)
    net.eval()
    with torch.no_grad():
        return float(torch.sigmoid(net(s, g) / model["T"]).cpu().item())


# ----------------------------------------------------------------- baselines
def train_baselines(data, splits):
    tr, va = splits["main"]["train"], splits["main"]["val"]
    y_tr = data["labels"][tr]

    ll_scaler = StandardScaler().fit(data["final_hidden"][tr])
    ll = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    ll.fit(ll_scaler.transform(data["final_hidden"][tr]), y_tr)

    g_scaler = StandardScaler().fit(data["globals"][tr])
    gl = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    gl.fit(g_scaler.transform(data["globals"][tr]), y_tr)

    def ll_score(idx):
        return ll.predict_proba(ll_scaler.transform(data["final_hidden"][idx]))[:, 1]

    def gl_score(idx):
        return gl.predict_proba(g_scaler.transform(data["globals"][idx]))[:, 1]

    print(f"[baseline] last-layer probe   val AUROC={_auroc(data['labels'][va], ll_score(va)):.4f}")
    print(f"[baseline] global-scalar logreg val AUROC={_auroc(data['labels'][va], gl_score(va)):.4f}")
    return {"lastlayer": (ll, ll_scaler), "globals": (gl, g_scaler)}


# ----------------------------------------------------------------- thresholds
def derive_thresholds(probs: np.ndarray, labels: np.ndarray, precision_target: float = 0.85):
    """green = lowest prob whose >=-threshold positives hit the precision target;
    red = highest prob whose <-threshold negatives hit the precision target (for the 0 class)."""
    order = np.argsort(-probs)
    p_sorted, y_sorted = probs[order], labels[order]
    green = None
    tp = 0
    for k in range(len(p_sorted)):
        tp += y_sorted[k]
        if tp / (k + 1) >= precision_target:
            green = float(p_sorted[k])
    # red: scan ascending, precision of the negative class among predicted-negative
    order2 = np.argsort(probs)
    p2, y2 = probs[order2], labels[order2]
    red = None
    tn = 0
    for k in range(len(p2)):
        tn += (1 - y2[k])
        if tn / (k + 1) >= precision_target:
            red = float(p2[k])
    return green, red


# ----------------------------------------------------------------- save / main
def save_all(cfg, lrd, baselines, data):
    import joblib

    ckpt_dir = cfg["paths"]["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save({
        "state_dict": lrd["net"].state_dict(),
        "fscaler": lrd["fscaler"].state(),
        "gscaler": {"mean": lrd["gscaler"].mean_, "scale": lrd["gscaler"].scale_},
        "T": lrd["T"],
        "n_feat": data["feats"].shape[2],
        "n_global": data["globals"].shape[1],
        "hidden": cfg["train"]["hidden"],
        "feat_names": data["feat_names"],
        "global_names": data["global_names"],
    }, os.path.join(ckpt_dir, "lrd.pt"))
    joblib.dump(baselines, os.path.join(ckpt_dir, "baselines.joblib"))


def main():
    cfg = load_config()
    data, splits = load_arrays(cfg)
    print(f"[data] N={len(data['labels'])} feats={data['feats'].shape} "
          f"globals={data['globals'].shape} label_rate={data['labels'].mean():.3f}")

    lrd = train_lrd(data, splits, cfg)
    baselines = train_baselines(data, splits)

    va = splits["main"]["val"]
    va_probs = lrd_scores(lrd, data, va)
    green, red = derive_thresholds(va_probs, data["labels"][va])
    print(f"[thresholds] green(P>=)={green}  red(P<)={red}")

    save_all(cfg, lrd, baselines, data)

    # write thresholds back into config.yaml
    import yaml
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    raw.setdefault("thresholds", {})
    raw["thresholds"]["green"] = green
    raw["thresholds"]["red"] = red
    with open(cfg_path, "w") as f:
        yaml.safe_dump(raw, f, sort_keys=False)
    print(f"[done] saved models to {cfg['paths']['ckpt_dir']} and thresholds to config.yaml")


if __name__ == "__main__":
    main()
