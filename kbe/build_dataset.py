"""build_dataset.py — capture features and self-behavioral labels for PopQA + TriviaQA.

For each question we (1) run one prompt-only forward and extract Signal A/B features, and
(2) greedily generate the model's own answer and label ``knows=1`` iff that answer matches the
gold (or an alias). The label is the *model's behavior*, not world truth — both the activations
and the label come from the same bf16 instance.

Outputs (paths from config.yaml):
  * ``data/dataset.parquet`` — one row per example (id, dataset, question, popularity, label,
    consistency, prediction, strongest_write_token, shapes).
  * ``data/feats.npz``       — aligned arrays: feats [N,L,F] f16, globals [N,G] f32,
    final_hidden [N,H] f16, ids, labels.
  * ``data/splits.json``     — a stratified train/val/test split (by dataset+label) and a
    cross-dataset split (train TriviaQA, test PopQA).

Run a smoke test first:  ``python -m kbe.build_dataset --limit 100``
Full build:              ``python -m kbe.build_dataset``
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from kbe.model_engine import Engine, load_config, set_seed


# ----------------------------------------------------------------- answer matching
_ARTICLE_RE = re.compile(r"\b(a|an|the)\b")
_PUNCT_TABLE = {ord(c): " " for c in string.punctuation}
_WS_RE = re.compile(r"\s+")


def _normalize(s: str) -> str:
    s = (s or "").lower().translate(_PUNCT_TABLE)
    s = _ARTICLE_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def _toks(s: str) -> list[str]:
    return _normalize(s).split()


def is_match(prediction: str, golds: list[str]) -> bool:
    """True if any gold appears as a contiguous word-subsequence of the prediction (or vice versa)."""
    pt = _toks(prediction)
    if not pt:
        return False
    for g in golds:
        gt = _toks(g)
        if not gt:
            continue
        if len(gt) <= len(pt):
            for i in range(len(pt) - len(gt) + 1):
                if pt[i:i + len(gt)] == gt:
                    return True
        elif pt == gt[: len(pt)]:  # prediction is a prefix of a longer gold
            return True
    return False


# ----------------------------------------------------------------- dataset loaders
def _load_popqa(n: int | None, seed: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("akariasai/PopQA", split="test").shuffle(seed=seed)
    if n:
        ds = ds.select(range(min(n, len(ds))))
    out = []
    for i, ex in enumerate(ds):
        golds: list[str] = []
        pa = ex.get("possible_answers")
        if isinstance(pa, str):
            try:
                golds = list(json.loads(pa))
            except Exception:
                golds = [pa]
        elif isinstance(pa, list):
            golds = list(pa)
        if ex.get("obj"):
            golds.append(ex["obj"])
        pop = ex.get("s_pop")
        out.append({
            "id": f"popqa-{ex.get('id', i)}",
            "dataset": "popqa",
            "question": ex["question"],
            "golds": [g for g in golds if g],
            "popularity": float(pop) if pop is not None else float("nan"),
        })
    return out


def _load_triviaqa(n: int | None, seed: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation").shuffle(seed=seed)
    if n:
        ds = ds.select(range(min(n, len(ds))))
    out = []
    for ex in ds:
        ans = ex.get("answer", {}) or {}
        golds = [ans.get("value")] + list(ans.get("aliases", [])) + list(ans.get("normalized_aliases", []))
        out.append({
            "id": f"triviaqa-{ex.get('question_id', '')}",
            "dataset": "triviaqa",
            "question": ex["question"],
            "golds": [g for g in golds if g],
            "popularity": float("nan"),
        })
    return out


_LOADERS = {"popqa": _load_popqa, "triviaqa": _load_triviaqa}


# ----------------------------------------------------------------- resumable checkpoint
# The capture+label loop is long (~hours for the full set). We periodically flush partial
# progress to a single atomically-replaced .npz so a killed/reaped run can resume. The
# checkpoint is keyed on (datasets, total, seed); a config change invalidates it. Because
# load_examples() is deterministic given that key, resuming by example index is exact.
FLUSH_EVERY = 50


def _ckpt_path(paths) -> str:
    return os.path.join(paths["data_dir"], "build_ckpt.npz")


def _save_ckpt(paths, records, feats_list, glob_list, fh_list, meta):
    p = _ckpt_path(paths)
    tmp = p + ".tmp.npz"
    np.savez(
        tmp,
        feats=np.stack(feats_list).astype(np.float16),
        globals=np.stack(glob_list).astype(np.float32),
        final_hidden=np.stack(fh_list).astype(np.float16),
        records=np.array(json.dumps(records)),
        meta=np.array(json.dumps(meta)),
    )
    os.replace(tmp, p)  # single atomic commit: npz + records together


def _load_ckpt(paths, meta):
    p = _ckpt_path(paths)
    if not os.path.exists(p):
        return None
    try:
        npz = np.load(p, allow_pickle=False)
        if json.loads(str(npz["meta"])) != meta:
            return None
        records = json.loads(str(npz["records"]))
        feats_list = list(npz["feats"])
        glob_list = list(npz["globals"])
        fh_list = list(npz["final_hidden"])
    except Exception as e:
        print(f"[resume] ignoring unreadable checkpoint ({e})")
        return None
    # truncate to the common length defensively (records is authoritative)
    n = min(len(records), len(feats_list), len(glob_list), len(fh_list))
    return records[:n], feats_list[:n], glob_list[:n], fh_list[:n]


def load_examples(names: list[str], total: int, seed: int) -> list[dict]:
    """Load a roughly balanced set of ``total`` examples across the requested datasets."""
    per = max(1, total // len(names))
    examples: list[dict] = []
    for k, name in enumerate(names):
        # last dataset soaks up the remainder so the sum equals `total`
        want = per if k < len(names) - 1 else total - per * (len(names) - 1)
        loaded = _LOADERS[name](want, seed)
        examples.extend(loaded)
        print(f"[data] {name}: requested {want}, loaded {len(loaded)}")
    rng = np.random.default_rng(seed)
    rng.shuffle(examples)
    return examples


# ----------------------------------------------------------------- splits
def stratified_split(df: pd.DataFrame, seed: int, val_frac: float, test_frac: float):
    rng = np.random.default_rng(seed)
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, (d, l) in enumerate(zip(df["dataset"], df["label"])):
        groups[(d, int(l))].append(i)
    train, val, test = [], [], []
    for idxs in groups.values():
        idxs = list(idxs)
        rng.shuffle(idxs)
        n = len(idxs)
        nv, nt = int(round(val_frac * n)), int(round(test_frac * n))
        val += idxs[:nv]
        test += idxs[nv:nv + nt]
        train += idxs[nv + nt:]
    return sorted(train), sorted(val), sorted(test)


def cross_dataset_split(df: pd.DataFrame, train_ds: str = "triviaqa", test_ds: str = "popqa"):
    train = [i for i, d in enumerate(df["dataset"]) if d == train_ds]
    test = [i for i, d in enumerate(df["dataset"]) if d == test_ds]
    return sorted(train), sorted(test)


# ----------------------------------------------------------------- main
def main():
    cfg = load_config()
    dcfg = cfg["dataset"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(dcfg["datasets"]),
                    help="comma-separated subset of: popqa,triviaqa")
    ap.add_argument("--limit", type=int, default=None,
                    help="total examples (overrides config target_total; use for smoke tests)")
    args = ap.parse_args()

    names = [s.strip() for s in args.datasets.split(",") if s.strip()]
    total = args.limit or dcfg["target_total"]
    seed = dcfg["seed"]
    set_seed(seed)

    from kbe.features import extract_features, FEAT_NAMES, GLOBAL_NAMES

    examples = load_examples(names, total, seed)
    print(f"[data] total examples to process: {len(examples)}")

    paths = cfg["paths"]
    os.makedirs(paths["data_dir"], exist_ok=True)
    meta = {"datasets": names, "total": total, "seed": seed}
    ckpt = _load_ckpt(paths, meta)
    if ckpt is not None and len(ckpt[0]) > 0:
        records, feats_list, glob_list, fh_list = [list(x) for x in ckpt]
        start = len(records)
        print(f"[resume] checkpoint found: {start}/{len(examples)} already done; continuing")
    else:
        records, feats_list, glob_list, fh_list = [], [], [], []
        start = 0

    eng = Engine(cfg).load()
    gcfg = cfg.get("generation", {})
    n_consistency = int(gcfg.get("consistency_samples", 0))
    temp = float(gcfg.get("consistency_temperature", 0.7))

    for ex in tqdm(examples[start:], desc="capture+label", initial=start, total=len(examples)):
        cap = eng.capture(ex["question"])
        fs = extract_features(eng, cap, cfg)
        pred = eng.generate_answer(ex["question"])
        label = int(is_match(pred, ex["golds"]))

        consistency = float("nan")
        if n_consistency > 0:
            hits = sum(is_match(eng.generate_answer(ex["question"], do_sample=True, temperature=temp),
                                ex["golds"]) for _ in range(n_consistency))
            consistency = hits / n_consistency

        records.append({
            "id": ex["id"],
            "dataset": ex["dataset"],
            "question": ex["question"],
            "popularity": ex["popularity"],
            "label": label,
            "consistency": consistency,
            "prediction": pred,
            "strongest_write_layer": fs.strongest_write_layer,
            "strongest_write_token": fs.strongest_write_token,
            "n_layers": fs.feats.shape[0],
            "n_feat": fs.feats.shape[1],
        })
        feats_list.append(fs.feats)
        glob_list.append(fs.globals)
        fh_list.append(fs.final_hidden)

        if len(records) % FLUSH_EVERY == 0:
            _save_ckpt(paths, records, feats_list, glob_list, fh_list, meta)

    df = pd.DataFrame(records)
    feats = np.stack(feats_list).astype(np.float16)       # [N, L, F]
    globs = np.stack(glob_list).astype(np.float32)        # [N, G]
    fhid = np.stack(fh_list).astype(np.float16)           # [N, H]

    df.to_parquet(paths["dataset_parquet"])
    np.savez_compressed(
        paths["feats_npz"],
        feats=feats, globals=globs, final_hidden=fhid,
        ids=df["id"].to_numpy(), labels=df["label"].to_numpy(),
        feat_names=np.array(FEAT_NAMES), global_names=np.array(GLOBAL_NAMES),
    )

    tr, va, te = stratified_split(df, seed, dcfg["val_frac"], dcfg["test_frac"])
    ctr, cte = cross_dataset_split(df)
    with open(paths["splits_json"], "w") as f:
        json.dump({"main": {"train": tr, "val": va, "test": te},
                   "cross": {"train": ctr, "test": cte}}, f)

    # resolved config alongside the outputs
    import yaml
    with open(os.path.join(paths["data_dir"], "run_config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # final outputs are committed; drop the resume checkpoint
    ck = _ckpt_path(paths)
    if os.path.exists(ck):
        os.remove(ck)

    # ---- report ----
    print(f"\n[done] {len(df)} examples -> {paths['dataset_parquet']}, {paths['feats_npz']}")
    print(f"[split] main: train={len(tr)} val={len(va)} test={len(te)}   cross: train={len(ctr)} test={len(cte)}")
    print("[balance] label rate overall: %.3f" % df["label"].mean())
    for name in names:
        sub = df[df["dataset"] == name]
        if len(sub):
            print(f"          {name:<10} n={len(sub):<5} label_rate={sub['label'].mean():.3f}")
    print("\n[samples] (question | label | prediction | strongest write token)")
    for _, r in df.sample(min(8, len(df)), random_state=seed).iterrows():
        q = (r["question"][:60] + "…") if len(r["question"]) > 60 else r["question"]
        print(f"  [{r['label']}] {q!r} -> {r['prediction']!r}  | write={r['strongest_write_token']!r}")


if __name__ == "__main__":
    main()
