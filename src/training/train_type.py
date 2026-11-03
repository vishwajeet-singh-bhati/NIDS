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
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

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


class TypeData:
    """Malicious windows of one split with integer category labels."""

    def __init__(self, dataset_dir: str, split: str, norm: Normalizer, classes: Optional[List[str]] = None):
        self.d = SplitData(dataset_dir, split, norm)
        ix = self.d.index
        mal = (ix["binary_label"].astype(int) == 1).to_numpy()
        cat = ix["malware_category"].astype(str).to_numpy()
        if classes is None:
            classes = sorted(set(cat[mal]))
        self.classes = classes
        lut = {c: i for i, c in enumerate(classes)}
        keep = mal & np.array([c in lut for c in cat])
        self.rows = np.flatnonzero(keep)
        self.y = np.array([lut[c] for c in cat[self.rows]], dtype=np.int64)
        self.capture = ix["capture_id"].astype(str).to_numpy()[self.rows]


def _logits(model, td: TypeData, kinds, device: str, bs: int, prefix_k: Optional[int] = None) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(td.rows), bs):
            b = td.d.batch(td.rows[i:i + bs], kinds, prefix_k)
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
               qat_start_epoch: int = 6) -> Dict[str, Any]:
    set_seed(seed)
    device = resolve_device(device)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    ddir = Path(dataset_dir)
    norm = Normalizer.load(ddir / "normalization.json")
    tr = TypeData(dataset_dir, "train", norm)
    classes = tr.classes
    va, te = TypeData(dataset_dir, "val", norm, classes), TypeData(dataset_dir, "test", norm, classes)
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
            losses.append(float(loss))
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
           "val": _report("val", va, lv, classes, thr), "test": _report("test", te, lt, classes, thr)}
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
    a = ap.parse_args(argv)
    setup_logging()
    train_type(a.dataset, a.out, a.epochs, a.batch_size, a.lr, a.prefix_aug, a.seed, a.device, a.target_acc,
               a.max_steps_per_epoch, qat_start_epoch=a.qat_start_epoch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
