"""Stage 2: malware TYPE classifier (adware / ransomware / ...), trained on MALICIOUS windows only.

Cascade: the binary detector (stage 1) raises the alarm; this model is only asked "which type does the
flagged traffic resemble?". It never sees benign traffic, so it says nothing about benign-vs-malicious.

* Labels = `malware_category` from the split index (capture-level weak labels, like stage 1).
* Same inputs / normaliser / QAT layers as stage 1 (HybridResNet1D with `n_out` = number of classes).
* A flow's type is rarely visible in one window, so predictions are POOLED per capture (mean softmax over
  the capture's windows); this is what an alarm (many flagged flows) would feed in.
* An "unsure" threshold on the pooled confidence is chosen on VALIDATION only: the largest coverage whose
  accuracy on kept captures reaches `--target-acc`. Test captures below it are reported as "unsure".
* Split is the stage-1 split (capture-level, no leakage). Test is used once, at the end.

    python -m src.training.train_type --dataset data/datasets/full_group_random_s0 --out runs/type_s0

Optional extras (all selected / reported on validation, then test once):
  --stage1 <model.pt>      adds a 5th class "NotMalware" trained on HARD NEGATIVES: benign train windows that the
                           stage-1 detector scores highest (what a false alarm looks like). Flagged-but-benign bursts
                           can then be routed to "probably benign".
  --exclude-class Adware   hold one category OUT of training; its captures are then an "unknown malware type" test:
                           we report how many of them fall below the unsure threshold (-> "unknown type").
  The report always includes a POOL-SIZE curve: accuracy when only N random windows of a capture are available
  (an alarm burst is much shorter than a whole capture).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.data.normalization import Normalizer
from src.models.model_factory import build_model, load_model_config
from src.training.data import SplitData
from src.training.evaluate import resolve_device
from src.utils.logging_utils import setup_logging
from src.utils.reproducibility import set_seed

log = logging.getLogger("train_type")


def _macro_f1(y: np.ndarray, p: np.ndarray, n: int) -> float:
    f = []
    for c in range(n):
        tp = float(((y == c) & (p == c)).sum()); fp = float(((y != c) & (p == c)).sum()); fn = float(((y == c) & (p != c)).sum())
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def _per_class(y: np.ndarray, p: np.ndarray, classes: List[str]) -> Dict[str, Dict[str, float]]:
    out = {}
    for c, name in enumerate(classes):
        tp = float(((y == c) & (p == c)).sum()); fp = float(((y != c) & (p == c)).sum()); fn = float(((y == c) & (p != c)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        out[name] = {"precision": pr, "recall": rc, "f1": 0.0 if pr + rc == 0 else 2 * pr * rc / (pr + rc), "support": int((y == c).sum())}
    return out


NEG = "NotMalware"


class TypeData:
    """Windows of one split with integer labels: malicious windows by `malware_category`, plus (optionally)
    hard-negative benign windows labelled NEG. Excluded categories are kept apart as `ood_rows`."""

    def __init__(self, dataset_dir: str, split: str, norm: Normalizer, classes: Optional[List[str]] = None,
                 exclude: Sequence[str] = (), neg_rows: Optional[np.ndarray] = None):
        self.d = SplitData(dataset_dir, split, norm)
        ix = self.d.index
        mal = (ix["binary_label"].astype(int) == 1).to_numpy()
        cat = ix["malware_category"].astype(str).to_numpy()
        if classes is None:
            classes = sorted(set(cat[mal]) - set(exclude))
            if neg_rows is not None:
                classes = classes + [NEG]
        self.classes = classes
        lut = {c: i for i, c in enumerate(classes)}
        keep = mal & np.array([c in lut for c in cat])
        rows = np.flatnonzero(keep)
        y = np.array([lut[c] for c in cat[rows]], dtype=np.int64)
        if neg_rows is not None and NEG in lut:
            rows = np.concatenate([rows, np.asarray(neg_rows, dtype=np.int64)])
            y = np.concatenate([y, np.full(len(neg_rows), lut[NEG], dtype=np.int64)])
        self.rows, self.y = rows, y
        cap = ix["capture_id"].astype(str).to_numpy()
        self.capture = cap[rows]
        ood = mal & np.isin(cat, list(exclude)) if len(exclude) else np.zeros(len(mal), bool)
        self.ood_rows = np.flatnonzero(ood)
        self.ood_capture = cap[self.ood_rows]


def hard_negatives(dataset_dir: str, split: str, norm: Normalizer, stage1: str, device: str,
                   top_frac: float = 0.2, sample: int = 150000, max_n: int = 30000, seed: int = 0) -> np.ndarray:
    """Benign windows that the stage-1 detector scores highest (its false-alarm look-alikes)."""
    from src.models.model_factory import load_checkpoint
    from src.training.evaluate import predict
    d = SplitData(dataset_dir, split, norm)
    ben = np.flatnonzero(d.index["binary_label"].astype(int).to_numpy() == 0)
    rng = np.random.default_rng(seed)
    if len(ben) > sample:
        ben = np.sort(rng.choice(ben, sample, replace=False))
    m, _ = load_checkpoint(stage1, device)
    d.sel = ben
    sc = predict(m, d, 1024, device)
    top = ben[np.argsort(-sc)[: max(int(top_frac * len(ben)), 1)]]
    if len(top) > max_n:
        top = rng.choice(top, max_n, replace=False)
    log.info("%s: %d hard-negative benign windows (top %.0f%% of %d stage-1 scores)", split, len(top), 100 * top_frac, len(ben))
    return np.sort(top)


def _logits(model, td: TypeData, kinds, device: str, bs: int, prefix_k: Optional[int] = None,
            rows: Optional[np.ndarray] = None) -> np.ndarray:
    model.eval()
    out = []
    rows = td.rows if rows is None else rows
    if len(rows) == 0:
        return np.zeros((0, model.n_out))
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            b = td.d.batch(rows[i:i + bs], kinds, prefix_k)
            b = {k: v.to(device) for k, v in b.items()}
            out.append(model(b).cpu().numpy())
    return np.concatenate(out)


def _pool(td: TypeData, logits: np.ndarray):
    """Mean softmax per capture -> (capture ids, probs [C, K], true label [C])."""
    p = torch.softmax(torch.from_numpy(logits), 1).numpy()
    df = pd.DataFrame(p); df["cap"] = td.capture; df["y"] = td.y
    g = df.groupby("cap")
    probs = g[list(range(p.shape[1]))].mean().to_numpy()
    y = g["y"].first().to_numpy()
    return g.size().index.to_numpy(), probs, y


def _choose_unsure(probs: np.ndarray, y: np.ndarray, target_acc: float) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    best = 1.01
    for t in np.unique(np.round(conf, 4)):
        keep = conf >= t
        if keep.sum() >= max(5, int(0.2 * len(y))) and (pred[keep] == y[keep]).mean() >= target_acc:
            best = min(best, float(t))
    return best


def pool_curve(td: TypeData, logits: np.ndarray, thr: float, sizes=(1, 5, 20, 50), draws: int = 20, seed: int = 0):
    """Accuracy when only N random windows of a capture are available (alarm bursts are short)."""
    p = torch.softmax(torch.from_numpy(logits), 1).numpy()
    caps = pd.Series(np.arange(len(p))).groupby(td.capture).apply(np.array)
    rng = np.random.default_rng(seed)
    out = {}
    for n in sizes:
        acc, cov, accc = [], [], []
        for _ in range(draws):
            pr, yy = [], []
            for idx in caps:
                sel = rng.choice(idx, min(n, len(idx)), replace=False)
                pr.append(p[sel].mean(0)); yy.append(td.y[idx[0]])
            pr, yy = np.array(pr), np.array(yy)
            pred, conf = pr.argmax(1), pr.max(1)
            keep = conf >= thr
            acc.append((pred == yy).mean()); cov.append(keep.mean())
            accc.append((pred[keep] == yy[keep]).mean() if keep.any() else np.nan)
        out[str(n)] = {"capture_acc": float(np.mean(acc)), "coverage": float(np.mean(cov)),
                       "acc_on_confident": float(np.nanmean(accc))}
    return out


def ood_report(td: TypeData, model, kinds, device: str, classes: List[str], thr: float, n_windows: int = 0):
    """Held-out (never trained) malware category: how are its captures labelled?"""
    if len(td.ood_rows) == 0:
        return None
    lg = _logits(model, td, kinds, device, 512, rows=td.ood_rows)
    p = torch.softmax(torch.from_numpy(lg), 1).numpy()
    df = pd.DataFrame(p); df["cap"] = td.ood_capture
    probs = df.groupby("cap").mean().to_numpy()
    conf, pred = probs.max(1), probs.argmax(1)
    unsure = conf < thr
    lab = {c: int(((pred == i) & ~unsure).sum()) for i, c in enumerate(classes)}
    return {"captures": int(len(probs)), "unknown_type": int(unsure.sum()), "labelled_as": lab,
            "fraction_unknown": float(unsure.mean()),
            "fraction_wrongly_confident": float(((~unsure) & (classes_is_known(classes, pred))).mean())}


def classes_is_known(classes: List[str], pred: np.ndarray) -> np.ndarray:
    return np.array([classes[i] != NEG for i in pred])


def _report(name: str, td: TypeData, logits: np.ndarray, classes: List[str], thr: float) -> Dict[str, Any]:
    wp = logits.argmax(1)
    _, probs, y = _pool(td, logits)
    pred = probs.argmax(1)
    conf = probs.max(1)
    keep = conf >= thr
    cm = np.zeros((len(classes), len(classes)), int)
    for a, b in zip(y, pred):
        cm[a, b] += 1
    return {"split": name, "windows": int(len(td.y)), "captures": int(len(y)),
            "window_acc": float((wp == td.y).mean()), "window_macro_f1": _macro_f1(td.y, wp, len(classes)),
            "capture_acc": float((pred == y).mean()), "capture_macro_f1": _macro_f1(y, pred, len(classes)),
            "per_class_capture": _per_class(y, pred, classes), "confusion_capture": cm.tolist(),
            "unsure_threshold": thr, "coverage": float(keep.mean()),
            "acc_on_confident": float((pred[keep] == y[keep]).mean()) if keep.any() else None}


def train_type(dataset_dir: str, out_dir: str, epochs: int = 12, batch_size: int = 128, lr: float = 1e-3,
               prefix_aug: float = 0.5, seed: int = 0, device: str = "auto", target_acc: float = 0.90,
               max_steps_per_epoch: int = 0, model_cfg_path: str = "configs/model.yaml",
               qat_start_epoch: int = 6, stage1: str = "", exclude: Sequence[str] = ()) -> Dict[str, Any]:
    set_seed(seed)
    device = resolve_device(device)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    ddir = Path(dataset_dir)
    norm = Normalizer.load(ddir / "normalization.json")
    negs = {sp: (hard_negatives(dataset_dir, sp, norm, stage1, device, seed=seed) if stage1 else None)
            for sp in ("train", "val", "test")}
    tr = TypeData(dataset_dir, "train", norm, None, exclude, negs["train"])
    classes = tr.classes
    va = TypeData(dataset_dir, "val", norm, classes, exclude, negs["val"])
    te = TypeData(dataset_dir, "test", norm, classes, exclude, negs["test"])
    K = len(classes)
    cfg = load_model_config(model_cfg_path, "hybrid_resnet")
    cfg.update({"n_out": K, "qat_start_epoch": qat_start_epoch})
    model = build_model("hybrid_resnet", cfg).to(device)
    kinds = set(model.inputs)
    cnt = np.bincount(tr.y, minlength=K).astype(float)
    w = torch.tensor(cnt.sum() / (K * np.maximum(cnt, 1)), dtype=torch.float32, device=device)
    log.info("classes %s | windows train %d val %d test %d | captures train %d val %d test %d", classes, len(tr.y),
             len(va.y), len(te.y), len(set(tr.capture)), len(set(va.capture)), len(set(te.capture)))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best, hist = -1.0, []
    for ep in range(1, epochs + 1):
        qat = model.set_qat_epoch(ep)
        if qat and ep == qat_start_epoch:
            best = -1.0
            for pg in opt.param_groups:
                pg["lr"] = max(pg["lr"] * 0.5, 1e-4)
        model.train()
        rng = np.random.default_rng(seed * 1000 + ep)
        order = rng.permutation(len(tr.rows))
        losses = []
        for s, i in enumerate(range(0, len(order), batch_size)):
            if max_steps_per_epoch and s >= max_steps_per_epoch:
                break
            sel = order[i:i + batch_size]
            if len(sel) < 2:
                continue
            pk = np.where(rng.random(len(sel)) < prefix_aug, rng.choice([8, 16, 32], len(sel)), tr.d.window_size)
            b = tr.d.batch(tr.rows[sel], kinds, pk)
            y = torch.from_numpy(tr.y[sel]).to(device)
            b = {k: v.to(device) for k, v in b.items()}
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(b), y, weight=w, label_smoothing=0.05)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        lv = _logits(model, va, kinds, device, 512)
        _, pv, yv = _pool(va, lv)
        f1 = _macro_f1(yv, pv.argmax(1), K)
        hist.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val_window_acc": float((lv.argmax(1) == va.y).mean()),
                     "val_capture_macro_f1": f1})
        log.info("epoch %2d loss %.4f val window acc %.3f | val capture macroF1 %.3f%s", ep, hist[-1]["train_loss"],
                 hist[-1]["val_window_acc"], f1, " [QAT]" if qat else "")
        if f1 > best:
            best = f1
            torch.save({"state": model.state_dict(), "model_cfg": cfg, "classes": classes, "epoch": ep,
                        "norm_spec": norm.spec}, out / "best.pt")
    ck = torch.load(out / "best.pt", map_location=device)
    model.load_state_dict(ck["state"])
    lv, lt = _logits(model, va, kinds, device, 512), _logits(model, te, kinds, device, 512)
    _, pv, yv = _pool(va, lv)
    thr = _choose_unsure(pv, yv, target_acc)
    res = {"classes": classes, "best_epoch": ck["epoch"], "target_acc_for_unsure": target_acc,
           "stage1_hard_negatives": stage1 or None, "excluded_from_training": list(exclude),
           "val": _report("val", va, lv, classes, thr), "test": _report("test", te, lt, classes, thr),
           "pool_curve_val": pool_curve(va, lv, thr), "pool_curve_test": pool_curve(te, lt, thr)}
    if exclude:
        res["unknown_type_val"] = ood_report(va, model, kinds, device, classes, thr)
        res["unknown_type_test"] = ood_report(te, model, kinds, device, classes, thr)
    ck["unsure_threshold"] = thr
    torch.save(ck, out / "best.pt")
    (out / "type_metrics.json").write_text(json.dumps(res, indent=2))
    pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
    (out / "type_report.md").write_text(render(res))
    print(render(res))
    return res


def render(r: Dict[str, Any]) -> str:
    cl = r["classes"]
    L = ["# Stage 2 - malware type (trained on malicious windows only)", "",
         "Weak capture-level labels; type is inferred from traffic similarity, not confirmed. Pooled per capture.", ""]
    for sp in ("val", "test"):
        s = r[sp]
        L += [f"## {sp}: {s['captures']} captures, {s['windows']} windows",
              f"- window acc {s['window_acc']:.3f}, macro-F1 {s['window_macro_f1']:.3f}",
              f"- capture acc {s['capture_acc']:.3f}, macro-F1 {s['capture_macro_f1']:.3f}",
              f"- 'unsure' below confidence {s['unsure_threshold']:.3f}: coverage {s['coverage']:.2f}, "
              f"accuracy on confident {s['acc_on_confident'] if s['acc_on_confident'] is None else round(s['acc_on_confident'], 3)}",
              "", "| class | precision | recall | F1 | captures |", "|---|---|---|---|---|"]
        for c in cl:
            m = s["per_class_capture"][c]
            L.append(f"| {c} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} | {m['support']} |")
        L += ["", "confusion (rows true, cols predicted): " + ", ".join(cl)]
        L += ["    " + " ".join(f"{v:4d}" for v in row) for row in s["confusion_capture"]] + [""]
    for sp in ("val", "test"):
        L += [f"## pool-size curve ({sp}): only N random windows per capture are available", "",
              "| N windows | capture acc | coverage (not unsure) | acc on confident |", "|---|---|---|---|"]
        for n, m in r[f"pool_curve_{sp}"].items():
            L.append(f"| {n} | {m['capture_acc']:.3f} | {m['coverage']:.2f} | {m['acc_on_confident']:.3f} |")
        L.append("")
    for sp in ("val", "test"):
        u = r.get(f"unknown_type_{sp}")
        if u:
            L += [f"## unknown-type check ({sp}): category(ies) {r['excluded_from_training']} were NEVER trained on", "",
                  f"- {u['captures']} captures: {u['unknown_type']} flagged 'unknown type' "
                  f"({u['fraction_unknown']:.2f}), {u['fraction_wrongly_confident']:.2f} confidently given a wrong "
                  f"known type; labelled as: {u['labelled_as']}", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--prefix-aug", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--target-acc", type=float, default=0.90)
    ap.add_argument("--max-steps-per-epoch", type=int, default=0)
    ap.add_argument("--qat-start-epoch", type=int, default=6)
    ap.add_argument("--stage1", default="", help="stage-1 model.pt -> add NotMalware class from hard negatives")
    ap.add_argument("--exclude-class", action="append", default=[], help="malware category held out of training (unknown-type test)")
    a = ap.parse_args(argv)
    setup_logging()
    train_type(a.dataset, a.out, a.epochs, a.batch_size, a.lr, a.prefix_aug, a.seed, a.device, a.target_acc,
               a.max_steps_per_epoch, qat_start_epoch=a.qat_start_epoch, stage1=a.stage1, exclude=a.exclude_class)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
