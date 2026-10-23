"""Train one model on one consolidated dataset, then evaluate it.

    python -m src.training.train --dataset data/datasets/trial20_group_random_s0 --model cnn_meta
    python -m src.training.train --dataset data/datasets/trial20_family_holdout_s0 --model mlp --epochs 10 --seed 1

Run directory (default runs/<dataset name>/<model>_s<seed>):
    config.json          resolved train/model config + dataset card + environment
    model_summary.txt    layers and parameter counts
    history.csv          per-epoch train/val metrics
    best.pt              best checkpoint by --monitor (weights, model cfg, feature spec, normaliser, label map)
    metrics.json         full evaluation (see src/training/evaluate.py)
    predictions_*.csv    per-window logits for val/test

Choices that matter for honesty:
  * model selection (best epoch) uses VAL only; test is touched once, after training;
  * the normaliser is the TRAIN-fitted one stored with the dataset and copied into the checkpoint;
  * imbalance is handled with a loss weight (default) or in-loop resampling - the stored data are
    never duplicated;
  * seeds are fixed; deterministic algorithms are requested (CPU runs are repeatable).
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.normalization import Normalizer
from src.models.model_factory import (MODEL_NAMES, build_model, count_parameters, load_model_config, model_summary,
                                       save_checkpoint)
from src.training.data import SplitData
from src.training.evaluate import evaluate_run, predict, print_summary, resolve_device
from src.training.losses import make_loss
from src.utils import metrics as M
from src.utils.logging_utils import setup_logging
from src.utils.reproducibility import set_seed

log = logging.getLogger("train")

DEFAULT_TRAIN_CFG: Dict[str, Any] = {
    "seed": 0, "epochs": 15, "batch_size": 256, "eval_batch_size": 1024, "lr": 1e-3, "weight_decay": 1e-4,
    "grad_clip": 1.0, "loss": "bce", "focal_gamma": 2.0, "sampler": "none", "max_steps_per_epoch": 1500,
    "max_train_examples": 0, "prefix_aug": 0.0, "init_from": "", "monitor": "val_pr_auc", "patience": 4, "min_delta": 5e-4, "lr_schedule": "plateau",
    "num_threads": 0, "fpr_targets": [0.01, 0.05], "prefix_lengths": [4, 8, 16, 32],
    "capture_alert_fracs": [0.1, 0.25, 0.5],
}
MONITORS = {"val_pr_auc": "max", "val_roc_auc": "max", "val_loss": "min", "val_recall_at_fpr1": "max"}


def _bce_mean(y: np.ndarray, logit: np.ndarray) -> float:
    p = np.clip(M.sigmoid(logit), 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _val_metrics(y: np.ndarray, s: np.ndarray) -> Dict[str, float]:
    thr = M.threshold_for_fpr(y, s, 0.01)
    return {"val_loss": _bce_mean(y, s), "val_roc_auc": M.safe_roc_auc(y, s), "val_pr_auc": M.safe_pr_auc(y, s),
            "val_recall_at_fpr1": M.binary_metrics(y, s, thr)["recall"]}


def train(dataset_dir: str, model_name: str, run_dir: str | Path, train_cfg: Dict[str, Any],
          model_cfg: Optional[Dict[str, Any]] = None, model_cfg_path: str = "configs/model.yaml",
          device: str = "auto", evaluate: bool = True) -> Dict[str, Any]:
    cfg = {**DEFAULT_TRAIN_CFG, **train_cfg}
    if cfg["monitor"] not in MONITORS:
        raise ValueError(f"monitor must be one of {sorted(MONITORS)}")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(cfg["seed"]))
    torch.use_deterministic_algorithms(True, warn_only=True)
    if cfg["num_threads"]:
        torch.set_num_threads(int(cfg["num_threads"]))
    device = resolve_device(device)
    ddir = Path(dataset_dir)
    card = json.loads((ddir / "dataset_card.json").read_text())
    norm = Normalizer.load(ddir / "normalization.json")
    model_cfg = dict(model_cfg) if model_cfg is not None else load_model_config(model_cfg_path, model_name)
    model = build_model(model_name, model_cfg).to(device)
    train_d = SplitData(dataset_dir, "train", norm, max_examples=int(cfg["max_train_examples"]), seed=int(cfg["seed"]))
    val_d = SplitData(dataset_dir, "val", norm)
    y_tr, y_val = train_d.y.astype(int), val_d.y.astype(int)
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
        raise ValueError(f"train/val must each contain both classes (train={np.bincount(y_tr)}, val={np.bincount(y_val)})")
    n_pos, n_neg = int((y_tr == 1).sum()), int((y_tr == 0).sum())
    pos_weight = 1.0 if cfg["sampler"] == "balanced" else n_neg / max(n_pos, 1)
    loss_fn = make_loss(cfg["loss"], pos_weight, float(cfg["focal_gamma"]))
    if cfg["init_from"]:
        from src.training.pretrain import load_pretrained_encoder
        log.info("initialising encoder from %s: %s", cfg["init_from"], load_pretrained_encoder(model, cfg["init_from"]))
    kinds = set(model.inputs)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    mode = MONITORS[cfg["monitor"]]
    sched = (torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode=mode, factor=0.5, patience=2)
             if cfg["lr_schedule"] == "plateau" else None)
    (run_dir / "model_summary.txt").write_text(model_summary(model) + "\n")
    (run_dir / "config.json").write_text(json.dumps({
        "train_cfg": cfg, "model_name": model_name, "model_cfg": model_cfg, "dataset_dir": str(dataset_dir),
        "dataset_card": card, "pos_weight": pos_weight, "train_examples": int(len(train_d)),
        "train_class_counts": {"benign": n_neg, "malicious_associated": n_pos},
        "env": {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
                "platform": platform.platform(), "device": device}}, indent=2, default=str))
    log.info("model=%s params=%d device=%s train=%d (pos=%d neg=%d pos_weight=%.2f) val=%d", model_name,
             count_parameters(model)["total"], device, len(train_d), n_pos, n_neg, pos_weight, len(val_d))

    best, best_epoch, bad, history = None, -1, 0, []
    sign = 1.0 if mode == "max" else -1.0
    for epoch in range(1, int(cfg["epochs"]) + 1):
        t0 = time.time()
        if hasattr(model, "set_qat_epoch"):
            qat_now = model.set_qat_epoch(epoch)
            if epoch == getattr(model, "qat_start_epoch", -1) and qat_now:
                bad = 0
                best = None
                for pg in opt.param_groups:
                    pg["lr"] = max(pg["lr"] * 0.5, 1e-4)
                log.info("INT8 QAT activated at epoch %d (reset patience &best, lr=%.6f)", epoch, opt.param_groups[0]["lr"])
        model.train()
        rng = np.random.default_rng(int(cfg["seed"]) * 1000 + epoch)
        losses = []
        for b in train_d.iter_train_batches(int(cfg["batch_size"]), kinds, rng, cfg["sampler"],
                                            int(cfg["max_steps_per_epoch"]),
                                            float(cfg["prefix_aug"]), cfg["prefix_lengths"]):
            b = {k: v.to(device) for k, v in b.items()}
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(b), b["y"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}; lower the learning rate or check the data")
            loss.backward()
            if cfg["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            opt.step()
            losses.append(float(loss.item()))
        if float(cfg["prefix_aug"]) > 0:
            # model selection must reflect what the streaming detector sees: pool val windows seen at each prefix length
            ks = sorted({min(int(k), val_d.window_size) for k in cfg["prefix_lengths"]} | {val_d.window_size})
            s_val = np.concatenate([predict(model, val_d, int(cfg["eval_batch_size"]), device, k) for k in ks])
            vm = _val_metrics(np.tile(y_val, len(ks)), s_val)
        else:
            s_val = predict(model, val_d, int(cfg["eval_batch_size"]), device)
            vm = _val_metrics(y_val, s_val)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else float("nan"), **vm,
               "lr": opt.param_groups[0]["lr"], "seconds": round(time.time() - t0, 1)}
        history.append(row)
        log.info("epoch %2d train_loss=%.4f val_loss=%.4f val_pr_auc=%.4f val_roc_auc=%.4f recall@fpr1%%=%.3f (%.1fs)",
                 epoch, row["train_loss"], vm["val_loss"], vm["val_pr_auc"], vm["val_roc_auc"], vm["val_recall_at_fpr1"],
                 row["seconds"])
        score = vm[cfg["monitor"]]
        if sched is not None:
            sched.step(score)
        if best is None or sign * (score - best) > float(cfg["min_delta"]):
            best, best_epoch, bad = score, epoch, 0
            save_checkpoint(run_dir / "best.pt", model, model_name, model_cfg, card["feature_spec"], norm.spec, cfg,
                            epoch, vm, {"dataset_dir": str(dataset_dir), "pos_weight": pos_weight})
        else:
            bad += 1
            if bad >= int(cfg["patience"]):
                log.info("early stopping (no improvement in %d epochs); best epoch %d", bad, best_epoch)
                break
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    out: Dict[str, Any] = {"run_dir": str(run_dir), "best_epoch": best_epoch, "best_" + cfg["monitor"]: best}
    if evaluate:
        res = evaluate_run(run_dir, device=device)
        print_summary(res)
        out["metrics_path"] = str(run_dir / "metrics.json")
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="consolidated dataset dir (src.data.consolidate output)")
    ap.add_argument("--model", required=True, choices=MODEL_NAMES)
    ap.add_argument("--train-config", default="configs/train.yaml")
    ap.add_argument("--model-config", default="configs/model.yaml")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--sampler", choices=["none", "balanced"], default=None)
    ap.add_argument("--loss", choices=["bce", "focal"], default=None)
    ap.add_argument("--max-steps-per-epoch", type=int, default=None)
    ap.add_argument("--max-train-examples", type=int, default=None)
    ap.add_argument("--prefix-aug", type=float, default=None, help="prob. of training on a random packet prefix")
    ap.add_argument("--init-from", default=None, help="pretrained encoder checkpoint (src.training.pretrain)")
    ap.add_argument("--monitor", choices=sorted(MONITORS), default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args(argv)
    setup_logging(a.log_level)
    cfg = yaml.safe_load(Path(a.train_config).read_text()) if Path(a.train_config).is_file() else {}
    for k in ("seed", "epochs", "batch_size", "lr", "sampler", "loss", "max_steps_per_epoch", "max_train_examples", "monitor", "prefix_aug", "init_from"):
        v = getattr(a, k)
        if v is not None:
            cfg[k] = v
    seed = int(cfg.get("seed", 0))
    run_dir = a.run_dir or str(Path(a.runs_root) / Path(a.dataset).name / f"{a.model}_s{seed}")
    try:
        out = train(a.dataset, a.model, run_dir, cfg, None, a.model_config, a.device, not a.no_eval)
    except (FileNotFoundError, ValueError, FloatingPointError) as exc:
        log.error("%s", exc)
        return 2
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
