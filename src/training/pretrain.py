"""Self-supervised pre-training of PacketBERT (masked packet modelling) - no labels used.

    python -m src.training.pretrain --dataset data/datasets/<name> --out runs/<name>/pretrain_s0 --epochs 20

15 % of the real packets of a window are replaced by a learned [MASK] token and the encoder must
reconstruct their normalised metadata (direction, payload length, inter-arrival time, protocol, flags)
from the other packets. Windows are randomly truncated to packet prefixes (like the streaming detector
sees them). Only the TRAIN split is used for the loss; the VAL split is used for model selection
(reconstruction loss, still label-free). Test is never touched. The normaliser is the train-fitted one.

Reconstruction baseline: predicting 0 (the normalised mean) gives MSE ~ the per-channel variance; the
log reports the ratio `val_mpm / val_zero_baseline` (< 1 means the encoder learned structure).

Output: <out>/encoder.pt = {encoder_state, model_cfg, feature_spec, normalization, ...}; use it with
    python -m src.training.train --model packet_bert --init-from <out>/encoder.pt ...
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.normalization import Normalizer
from src.models.model_factory import build_model, count_parameters, load_model_config
from src.training.data import SplitData
from src.training.evaluate import resolve_device
from src.utils.logging_utils import setup_logging
from src.utils.reproducibility import set_seed

log = logging.getLogger("pretrain")

DEFAULTS: Dict[str, Any] = {"seed": 0, "epochs": 20, "batch_size": 256, "lr": 5e-4, "weight_decay": 1e-2,
                            "mask_frac": 0.15, "prefix_prob": 0.5, "prefix_choices": [4, 8, 16, 32],
                            "max_steps_per_epoch": 1500, "max_train_examples": 0, "patience": 5, "grad_clip": 1.0}


def _mask_packets(mask: torch.Tensor, frac: float, gen: torch.Generator) -> torch.Tensor:
    """Bool [B,W]: random subset of REAL packets, at least one per window that has >= 2 packets."""
    r = torch.rand(mask.shape, generator=gen)
    sel = (r < frac) & (mask > 0)
    need = (sel.sum(1) == 0) & (mask.sum(1) >= 2)
    if need.any():
        rr = torch.where(mask > 0, torch.rand(mask.shape, generator=gen), torch.full_like(mask, -1.0))
        pick = rr.argmax(1)
        sel[need, pick[need]] = True
    return sel


def _step_loss(model, b, frac, gen, device):
    m = _mask_packets(b["mask"], frac, gen).to(device)
    pred = model.mpm_predict({k: v.to(device) for k, v in b.items()}, m)
    tgt = b["meta"].to(device)
    se = ((pred - tgt) ** 2).mean(-1)                                    # [B,W]
    n = m.sum().clamp(min=1)
    base = (tgt ** 2).mean(-1)
    return (se * m).sum() / n, (base * m).sum() / n


def load_pretrained_encoder(model: torch.nn.Module, path: str) -> str:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    state = ck["encoder_state"]
    own = model.state_dict()
    bad = [k for k, v in state.items() if k in own and own[k].shape != v.shape]
    if bad:
        raise ValueError(f"pretrained encoder does not match the model config (shape mismatch, e.g. {bad[:3]}); "
                         "use the same d_model/layers/use_payload as in pretraining")
    res = model.load_state_dict(state, strict=False)
    return f"{len(state)} tensors loaded; not in checkpoint (fresh): {sorted(res.missing_keys)[:4]}..."


def pretrain(dataset_dir: str, out_dir: str | Path, cfg: Optional[Dict[str, Any]] = None,
             model_cfg: Optional[Dict[str, Any]] = None, model_cfg_path: str = "configs/model.yaml",
             device: str = "auto") -> Dict[str, Any]:
    c = {**DEFAULTS, **(cfg or {})}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(int(c["seed"]))
    device = resolve_device(device)
    ddir = Path(dataset_dir)
    card = json.loads((ddir / "dataset_card.json").read_text())
    norm = Normalizer.load(ddir / "normalization.json")
    mc = dict(model_cfg) if model_cfg is not None else load_model_config(model_cfg_path, "packet_bert")
    model = build_model("packet_bert", mc).to(device)
    kinds = set(model.inputs)
    tr = SplitData(dataset_dir, "train", norm, max_examples=int(c["max_train_examples"]), seed=int(c["seed"]))
    va = SplitData(dataset_dir, "val", norm)
    W = tr.window_size
    choices = np.minimum(np.asarray(c["prefix_choices"], dtype=np.int64), W)
    opt = torch.optim.AdamW(model.parameters(), lr=float(c["lr"]), weight_decay=float(c["weight_decay"]))
    log.info("PacketBERT params=%d device=%s train windows=%d val=%d (labels are NOT used)",
             count_parameters(model)["total"], device, len(tr), len(va))

    def val_loss() -> tuple[float, float]:
        model.eval()
        g = torch.Generator().manual_seed(123)                       # fixed masks -> comparable across epochs
        rng = np.random.default_rng(123)
        tot = base = n = 0.0
        with torch.no_grad():
            for i in range(0, len(va), 1024):
                rows = va.sel[i:i + 1024]
                pk = np.where(rng.random(len(rows)) < c["prefix_prob"], rng.choice(choices, len(rows)), W)
                b = va.batch(rows, kinds, pk)
                l, bl = _step_loss(model, b, float(c["mask_frac"]), g, device)
                tot, base, n = tot + float(l) * len(rows), base + float(bl) * len(rows), n + len(rows)
        return tot / max(n, 1), base / max(n, 1)

    hist: List[dict] = []
    best, bad, gen = float("inf"), 0, torch.Generator().manual_seed(int(c["seed"]))
    for ep in range(1, int(c["epochs"]) + 1):
        t0 = time.time()
        model.train()
        rng = np.random.default_rng(int(c["seed"]) * 1000 + ep)
        order, losses = rng.permutation(tr.sel), []
        for s, i in enumerate(range(0, len(order), int(c["batch_size"]))):
            rows = order[i:i + int(c["batch_size"])]
            if len(rows) < 2 or (c["max_steps_per_epoch"] and s >= int(c["max_steps_per_epoch"])):
                break
            pk = np.where(rng.random(len(rows)) < c["prefix_prob"], rng.choice(choices, len(rows)), W)
            loss, _ = _step_loss(model, tr.batch(rows, kinds, pk), float(c["mask_frac"]), gen, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(c["grad_clip"]))
            opt.step()
            losses.append(float(loss.detach()))
        vl, vb = val_loss()
        hist.append({"epoch": ep, "train_mpm": float(np.mean(losses)) if losses else float("nan"), "val_mpm": vl,
                     "val_zero_baseline": vb, "ratio": vl / max(vb, 1e-9), "seconds": round(time.time() - t0, 1)})
        log.info("epoch %2d train_mpm=%.4f val_mpm=%.4f baseline=%.4f ratio=%.3f (%.1fs)", ep, hist[-1]["train_mpm"],
                 vl, vb, hist[-1]["ratio"], hist[-1]["seconds"])
        if vl < best - 1e-4:
            best, bad = vl, 0
            torch.save({"encoder_state": model.encoder_state(), "model_cfg": mc, "feature_spec": card["feature_spec"],
                        "normalization": norm.spec, "dataset_dir": str(dataset_dir), "epoch": ep, "val_mpm": vl,
                        "val_zero_baseline": vb, "pretrain_cfg": c}, out / "encoder.pt")
        else:
            bad += 1
            if bad >= int(c["patience"]):
                break
        pd.DataFrame(hist).to_csv(out / "pretrain_history.csv", index=False)
    pd.DataFrame(hist).to_csv(out / "pretrain_history.csv", index=False)
    res = {"encoder": str(out / "encoder.pt"), "best_val_mpm": best, "epochs_run": len(hist),
           "final_ratio_to_zero_baseline": hist[-1]["ratio"] if hist else None}
    (out / "pretrain_summary.json").write_text(json.dumps(res, indent=2))
    return res


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--model-config", default="configs/model.yaml")
    ap.add_argument("--train-config", default="configs/pretrain.yaml")
    for k, t in (("epochs", int), ("batch_size", int), ("lr", float), ("seed", int), ("max_steps_per_epoch", int),
                 ("max_train_examples", int)):
        ap.add_argument("--" + k.replace("_", "-"), type=t, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args(argv)
    setup_logging(a.log_level)
    cfg = yaml.safe_load(Path(a.train_config).read_text()) if Path(a.train_config).is_file() else {}
    for k in ("epochs", "batch_size", "lr", "seed", "max_steps_per_epoch", "max_train_examples"):
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)
    out = a.out or str(Path("runs") / Path(a.dataset).name / f"pretrain_s{cfg.get('seed', 0)}")
    try:
        res = pretrain(a.dataset, out, cfg, None, a.model_config, a.device)
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        return 2
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
