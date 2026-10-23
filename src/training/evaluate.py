"""Evaluation of a trained run (Phase 6).

    python -m src.training.evaluate --run-dir runs/group_random_s0/cnn_meta_s0
    python -m src.training.evaluate --run-dir runs/x/cnn_meta_s0 --eval-dataset data/datasets/other --eval-splits test

What is reported (everything is computed, nothing is hard-coded or estimated):
  * ROC-AUC, PR-AUC, calibration (ECE/Brier, before and after a VAL-fitted temperature) per split
  * operating points chosen on VAL only: window FPR <= each target, max-F1, logit >= 0; each applied
    unchanged to test. For each: confusion matrix, precision/recall/F1, FPR, FNR, per-class numbers
  * per malware family / category (recall), per benign subset (false-positive rate), per flow tag
    (local_infra vs the rest), and AUC with local infrastructure windows removed
  * capture-level view (labels are capture-level): fraction of benign captures that raise alerts
  * detection delay: model applied to the first k packets only (k = prefix_lengths); packets and seconds
    needed
  * `--eval-dataset`: apply the same checkpoint, the same train-fitted normaliser and the same val
    thresholds to ANOTHER consolidated dataset (other split policy, other capture source, ...)
The normaliser always comes from the checkpoint (train-fitted), never from the data being evaluated.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from src.data.normalization import Normalizer
from src.models.model_factory import count_parameters, load_checkpoint
from src.training.data import SplitData
from src.utils import metrics as M
from src.utils.logging_utils import setup_logging

log = logging.getLogger("evaluate")


def resolve_device(name: str = "auto") -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


@torch.no_grad()
def predict(model: torch.nn.Module, data: SplitData, batch_size: int, device: str,
            prefix_k: Optional[int] = None) -> np.ndarray:
    """Logits [len(data)] in the order of data.sel."""
    model.eval()
    kinds = set(model.inputs)
    out: List[np.ndarray] = []
    for b in data.iter_eval_batches(batch_size, kinds, prefix_k):
        b = {k: v.to(device) for k, v in b.items()}
        out.append(model(b).detach().cpu().numpy().astype(np.float64))
    return np.concatenate(out) if out else np.zeros(0)


def window_delay(data: SplitData, k: int) -> Dict[str, float]:
    """Packets and seconds needed before a prediction at prefix length k."""
    if data.rel_time is None:
        return {}
    sel = data.sel
    n_real = np.asarray(data.mask_raw[sel]).sum(1).astype(int)
    need = np.minimum(k, n_real)
    rel = np.asarray(data.rel_time[sel])
    sec = rel[np.arange(len(sel)), np.maximum(need - 1, 0)]
    return {"mean_packets_needed": float(need.mean()), "median_delay_s": float(np.median(sec)),
            "p90_delay_s": float(np.quantile(sec, 0.9)), "mean_delay_s": float(sec.mean()),
            "frac_windows_already_complete": float((n_real <= k).mean())}


def _group_tables(index: pd.DataFrame, y: np.ndarray, s: np.ndarray, thr: float) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    pos = y == 1
    if pos.any() and "malware_family" in index:
        out["per_family_recall"] = M.grouped_rates(index[pos].reset_index(drop=True), y[pos], s[pos], thr, "malware_family")
        out["per_category_recall"] = M.grouped_rates(index[pos].reset_index(drop=True), y[pos], s[pos], thr, "malware_category")
    neg = y == 0
    if neg.any() and "subset" in index:
        out["per_benign_subset_fpr"] = M.grouped_rates(index[neg].reset_index(drop=True), y[neg], s[neg], thr, "subset")
    if "flow_tag" in index:
        out["per_flow_tag"] = M.grouped_rates(index, y, s, thr, "flow_tag")
        keep = (index["flow_tag"] != "local_infra").to_numpy()
        out["auc_excluding_local_infra"] = {"roc_auc": M.safe_roc_auc(y[keep], s[keep]),
                                            "pr_auc": M.safe_pr_auc(y[keep], s[keep]), "n": int(keep.sum())}
    return out


def evaluate_run(run_dir: str | Path, dataset_dir: Optional[str] = None, eval_dataset: Optional[str] = None,
                 eval_splits: Sequence[str] = ("test",), device: str = "auto", out_name: str = "metrics.json",
                 save_predictions: bool = True) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    device = resolve_device(device)
    model, ck = load_checkpoint(run_dir / "best.pt", device)
    tcfg = ck["train_cfg"]
    norm = Normalizer(ck["normalization"])
    dataset_dir = dataset_dir or ck["extra"]["dataset_dir"]
    bs = int(tcfg.get("eval_batch_size", 1024))
    fpr_targets = list(tcfg.get("fpr_targets", [0.01, 0.05]))
    prefixes = [k for k in tcfg.get("prefix_lengths", [4, 8, 16, 32])]
    fracs = list(tcfg.get("capture_alert_fracs", [0.1, 0.25, 0.5]))

    sets: Dict[str, SplitData] = {s: SplitData(dataset_dir, s, norm) for s in ("val", "test")
                                  if (Path(dataset_dir) / f"{s}_index.csv").exists()}
    if "val" not in sets:
        raise ValueError(f"{dataset_dir} has no val split; thresholds cannot be chosen")
    if eval_dataset:
        for s in eval_splits:
            sets[f"ext_{s}"] = SplitData(eval_dataset, s, norm)
    logits = {name: predict(model, d, bs, device) for name, d in sets.items()}
    ys = {name: d.y.astype(int) for name, d in sets.items()}
    y_val, s_val = ys["val"], logits["val"]
    ops = M.operating_points(y_val, s_val, fpr_targets)
    T = M.fit_temperature(y_val, s_val)
    result: Dict[str, Any] = {
        "run": {"name": run_dir.name, "dataset_dir": str(dataset_dir), "model": ck["model_name"],
                "seed": tcfg.get("seed"), "best_epoch": ck["epoch"], "device": device,
                "parameters": count_parameters(model),
                "feature_spec_version": ck["feature_spec"].get("feature_spec_version"),
                "window_size": ck["feature_spec"].get("window_size"),
                "label_mapping": ck["label_mapping"], "eval_dataset": eval_dataset},
        "thresholds_from_val": {k: {"logit": v, "prob": float(M.sigmoid(np.array([v]))[0])} for k, v in ops.items()},
        "temperature_fit_on_val": T, "splits": {}, "notes": [
            "Labels are capture-level weak labels (malicious_associated = captured while a malware sample ran).",
            "Thresholds are chosen on val only. Metrics are window-level unless stated.",
            "Class 1 = malicious_associated; benign = 0."]}
    primary = f"fpr<={fpr_targets[0]:g}"
    for name, d in sets.items():
        y, s = ys[name], logits[name]
        idx = d.sel_index
        entry: Dict[str, Any] = M.summary_at(y, s, ops)
        entry["n_windows"] = int(len(y))
        entry["class_counts"] = {"benign": int((y == 0).sum()), "malicious_associated": int((y == 1).sum())}
        entry["calibration_raw"] = {k: v for k, v in M.calibration(y, M.sigmoid(s)).items() if k != "table"}
        entry["calibration_after_val_temperature"] = {k: v for k, v in M.calibration(y, M.sigmoid(s / T)).items() if k != "table"}
        if name != "val":
            thr = ops[primary]
            entry["breakdown_at_" + primary] = _group_tables(idx, y, s, thr)
            entry["capture_level"] = {op: M.capture_level(idx, y, s, t, fracs) for op, t in ops.items()
                                      if op in (primary, "max_f1")}
        result["splits"][name] = entry
        if save_predictions:
            pd.DataFrame({"example_id": idx["example_id"], "capture_id": idx["capture_id"], "y": y,
                          "logit": s}).to_csv(run_dir / f"predictions_{name}.csv", index=False)
    # detection delay / early prediction (val picks the threshold for each prefix length)
    W = sets["val"].window_size
    prefix_rows = []
    for k in sorted({int(p) for p in prefixes if 1 <= int(p) <= W} | {W}):
        s_val_k = logits["val"] if k == W else predict(model, sets["val"], bs, device, k)
        thr_k = M.threshold_for_fpr(y_val, s_val_k, fpr_targets[0])
        for name in [n for n in sets if n != "val"]:
            s_k = logits[name] if k == W else predict(model, sets[name], bs, device, k)
            m = M.binary_metrics(ys[name], s_k, thr_k)
            prefix_rows.append({"split": name, "k_packets": k, "roc_auc": M.safe_roc_auc(ys[name], s_k),
                                "pr_auc": M.safe_pr_auc(ys[name], s_k), "threshold_logit": thr_k,
                                "recall": m["recall"], "fpr": m["fpr"], "precision": m["precision"],
                                **window_delay(sets[name], k)})
    result["early_prediction"] = {"threshold_policy": f"val FPR <= {fpr_targets[0]:g} recomputed for each k",
                                  "rows": prefix_rows}
    (run_dir / out_name).write_text(json.dumps(result, indent=2, default=lambda o: None if o != o else str(o)))
    return result


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dataset", default=None, help="consolidated dataset dir (default: the one used for training)")
    ap.add_argument("--eval-dataset", default=None, help="another consolidated dataset to evaluate on")
    ap.add_argument("--eval-splits", nargs="+", default=["test"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-name", default="metrics.json")
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args(argv)
    setup_logging(a.log_level)
    try:
        res = evaluate_run(a.run_dir, a.dataset, a.eval_dataset, a.eval_splits, a.device, a.out_name)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        log.error("%s", exc)
        return 2
    print_summary(res)
    return 0


def print_summary(res: Dict[str, Any]) -> None:
    r = res["run"]
    print(f"run {r['name']} model={r['model']} params={r['parameters']['total']} best_epoch={r['best_epoch']}")
    for name, e in res["splits"].items():
        print(f"  [{name}] windows={e['n_windows']} {e['class_counts']} ROC-AUC={e['roc_auc']:.4f} PR-AUC={e['pr_auc']:.4f}")
        for op, m in e["at"].items():
            print(f"      {op:12s} thr={m['threshold']:+.3f} recall={m['recall']:.3f} fpr={m['fpr']:.3f} "
                  f"precision={m['precision']:.3f} f1={m['f1']:.3f}")


if __name__ == "__main__":
    sys.exit(main())
