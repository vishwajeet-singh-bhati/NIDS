"""Metrics and threshold utilities (numpy + sklearn). No torch dependency.

Conventions
-----------
* class 1 = malicious_associated (positive), class 0 = benign.
* The score is the model LOGIT (not a probability): logits do not saturate, so thresholds chosen
  from quantiles of benign scores stay meaningful. A window is flagged when score >= threshold.
* Operating points are chosen on the VALIDATION split only and then applied unchanged to test.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

NAN = float("nan")
CLASS_NAMES = {0: "benign", 1: "malicious_associated"}


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _div(a: float, b: float) -> float:
    return float(a) / float(b) if b else NAN


def safe_roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return NAN
    return float(roc_auc_score(y, s))


def safe_pr_auc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return NAN
    return float(average_precision_score(y, s))


def binary_metrics(y: np.ndarray, score: np.ndarray, thr: float) -> Dict[str, Any]:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(score) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec, rec = _div(tp, tp + fp), _div(tp, tp + fn)
    f1 = _div(2 * prec * rec, prec + rec) if not (math.isnan(prec) or math.isnan(rec)) else NAN
    prec0, rec0 = _div(tn, tn + fn), _div(tn, tn + fp)
    f10 = _div(2 * prec0 * rec0, prec0 + rec0) if not (math.isnan(prec0) or math.isnan(rec0)) else NAN
    return {
        "threshold": float(thr), "n": int(len(y)), "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn, "confusion_matrix": [[tn, fp], [fn, tp]],
        "precision": prec, "recall": rec, "f1": f1, "fpr": _div(fp, fp + tn), "fnr": _div(fn, fn + tp),
        "accuracy": _div(tp + tn, len(y)), "balanced_accuracy": float(np.nanmean([rec, rec0])),
        "flagged_rate": _div(tp + fp, len(y)),
        "per_class": {"benign": {"precision": prec0, "recall": rec0, "f1": f10, "support": tn + fp},
                      "malicious_associated": {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}},
    }


def threshold_for_fpr(y: np.ndarray, score: np.ndarray, target_fpr: float) -> float:
    """Smallest threshold whose window-level FPR on (y, score) is <= target_fpr (flag if score >= thr)."""
    neg = np.sort(np.asarray(score)[np.asarray(y) == 0])[::-1]
    if len(neg) == 0:
        return 0.0
    allowed = int(math.floor(target_fpr * len(neg)))
    if allowed >= len(neg):
        return float(neg[-1])
    return float(np.nextafter(neg[allowed], np.inf))   # negatives equal to neg[allowed] are NOT flagged


def best_f1_threshold(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return 0.0
    p, r, t = precision_recall_curve(y, score)
    f1 = np.where((p + r) > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)[:-1]
    return float(t[int(np.argmax(f1))]) if len(t) else 0.0


def calibration(y: np.ndarray, prob: np.ndarray, bins: int = 15) -> Dict[str, Any]:
    y = np.asarray(y).astype(float)
    prob = np.asarray(prob, dtype=np.float64)
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(prob, edges[1:-1]), 0, bins - 1)
    table, ece = [], 0.0
    for b in range(bins):
        m = which == b
        if not m.any():
            continue
        conf, acc = float(prob[m].mean()), float(y[m].mean())
        ece += m.mean() * abs(conf - acc)
        table.append({"bin": b, "n": int(m.sum()), "mean_prob": conf, "frac_positive": acc})
    return {"ece": float(ece), "brier": float(np.mean((prob - y) ** 2)), "table": table}


def fit_temperature(y: np.ndarray, logit: np.ndarray) -> float:
    """Scalar temperature T minimising NLL of sigmoid(logit / T) on (VAL) data; grid search."""
    y = np.asarray(y).astype(float)
    best, best_nll = 1.0, float("inf")
    for T in np.exp(np.linspace(np.log(0.1), np.log(10.0), 81)):
        p = np.clip(sigmoid(np.asarray(logit) / T), 1e-7, 1 - 1e-7)
        nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if nll < best_nll:
            best, best_nll = float(T), float(nll)
    return best


def operating_points(y_val: np.ndarray, s_val: np.ndarray, fpr_targets: Sequence[float]) -> Dict[str, float]:
    """Thresholds (logit space) chosen on VALIDATION data only."""
    ops = {f"fpr<={t:g}": threshold_for_fpr(y_val, s_val, t) for t in fpr_targets}
    ops["max_f1"] = best_f1_threshold(y_val, s_val)
    ops["logit>=0"] = 0.0
    return ops


def summary_at(y: np.ndarray, s: np.ndarray, ops: Dict[str, float]) -> Dict[str, Any]:
    return {"roc_auc": safe_roc_auc(y, s), "pr_auc": safe_pr_auc(y, s),
            "at": {name: binary_metrics(y, s, thr) for name, thr in ops.items()}}


def grouped_rates(index: pd.DataFrame, y: np.ndarray, s: np.ndarray, thr: float, by: str,
                  min_n: int = 1) -> Dict[str, Dict[str, Any]]:
    """Per-group window counts and flag rates. For positives-only groups (families) the useful number is
    `flag_rate` (= recall); for benign-only groups it is the false-positive rate."""
    out: Dict[str, Dict[str, Any]] = {}
    flagged = np.asarray(s) >= thr
    df = pd.DataFrame({"g": index[by].astype(str).to_numpy(), "y": np.asarray(y), "f": flagged, "s": np.asarray(s)})
    for g, d in df.groupby("g"):
        if g == "" or len(d) < min_n:
            continue
        out[g] = {"n": int(len(d)), "frac_positive": float(d["y"].mean()), "flag_rate": float(d["f"].mean()),
                  "mean_logit": float(d["s"].mean())}
    return out


def capture_level(index: pd.DataFrame, y: np.ndarray, s: np.ndarray, thr: float,
                  fracs: Iterable[float]) -> Dict[str, Any]:
    """Aggregate windows to captures (labels are capture-level, so this is the honest unit).
    A capture is flagged when at least `frac` of its windows are flagged."""
    df = pd.DataFrame({"cap": index["capture_id"].to_numpy(), "y": np.asarray(y), "s": np.asarray(s)})
    df["f"] = df["s"] >= thr
    g = df.groupby("cap").agg(y=("y", "max"), n=("y", "size"), rate=("f", "mean"), ms=("s", "mean"))
    res: Dict[str, Any] = {"n_captures": int(len(g)), "n_pos_captures": int(g["y"].sum()),
                           "auc_mean_logit": safe_roc_auc(g["y"].to_numpy(), g["ms"].to_numpy()),
                           "auc_alert_rate": safe_roc_auc(g["y"].to_numpy(), g["rate"].to_numpy()), "rules": {}}
    for fr in fracs:
        flagged = g["rate"] >= fr
        pos, neg = g["y"] == 1, g["y"] == 0
        res["rules"][f"alert_rate>={fr:g}"] = {
            "capture_tpr": _div(int((flagged & pos).sum()), int(pos.sum())),
            "capture_fpr": _div(int((flagged & neg).sum()), int(neg.sum())),
            "n_benign_captures_flagged": int((flagged & neg).sum()), "n_benign_captures": int(neg.sum())}
    neg_rates = g.loc[g["y"] == 0, "rate"]
    if len(neg_rates):
        res["benign_alert_rate_quantiles"] = {q: float(neg_rates.quantile(q)) for q in (0.5, 0.9, 0.99, 1.0)}
    return res
