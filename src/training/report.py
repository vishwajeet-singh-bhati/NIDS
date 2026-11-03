"""Turn metrics.json files into table rows / readable text. Pure functions, no torch."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

PRIMARY = "fpr<=0.01"


def _f(x: Any, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def run_row(m: Dict[str, Any]) -> Dict[str, Any]:
    r = m["run"]
    t = m["splits"].get("test", {})
    at = t.get("at", {})
    p1, p5 = at.get(PRIMARY, {}), at.get("fpr<=0.05", {})
    cl = t.get("capture_level", {}).get(PRIMARY, {})
    rule = cl.get("rules", {}).get("alert_rate>=0.5", {})
    early = [e for e in m.get("early_prediction", {}).get("rows", []) if e["split"] == "test"]
    k8 = next((e for e in early if e["k_packets"] == 8), {})
    return {
        "dataset": Path(r["dataset_dir"]).name, "model": r["model"], "seed": r.get("seed"),
        "params": r["parameters"]["total"], "window": r.get("window_size"), "best_epoch": r["best_epoch"],
        "test_windows": t.get("n_windows"), "test_roc_auc": t.get("roc_auc"), "test_pr_auc": t.get("pr_auc"),
        "recall@val_fpr1%": p1.get("recall"), "test_fpr@val_fpr1%": p1.get("fpr"), "precision@val_fpr1%": p1.get("precision"),
        "f1@val_fpr1%": p1.get("f1"), "recall@val_fpr5%": p5.get("recall"), "test_fpr@val_fpr5%": p5.get("fpr"),
        "auc_excl_local_infra": t.get(f"breakdown_at_{PRIMARY}", {}).get("auc_excluding_local_infra", {}).get("roc_auc"),
        "capture_auc": cl.get("auc_mean_logit"), "capture_tpr@rate0.5": rule.get("capture_tpr"),
        "benign_captures_flagged@rate0.5": rule.get("capture_fpr"),
        "recall@8pkts": k8.get("recall"), "fpr@8pkts": k8.get("fpr"), "median_delay_s@8pkts": k8.get("median_delay_s"),
        "ece_raw": t.get("calibration_raw", {}).get("ece"), "ece_after_T": t.get("calibration_after_val_temperature", {}).get("ece"),
    }


def load_runs(runs_root: str | Path, name: str = "metrics.json") -> List[Dict[str, Any]]:
    out = []
    for p in sorted(Path(runs_root).rglob(name)):
        try:
            m = json.loads(p.read_text())
            m["_path"] = str(p)
            out.append(m)
        except json.JSONDecodeError:
            continue
    return out


def markdown_table(rows: List[Dict[str, Any]], cols: List[str]) -> str:
    L = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        L.append("| " + " | ".join(_f(r.get(c)) if not isinstance(r.get(c), str) else r[c] for c in cols) + " |")
    return "\n".join(L)


def describe_run(m: Dict[str, Any], max_groups: int = 12) -> str:
    """Compact human-readable digest of one metrics.json (goes into the collector report)."""
    r, L = m["run"], []
    L.append(f"### {r['dataset_dir'].split('/')[-1]} / {r['model']} (seed {r['seed']}, {r['parameters']['total']} params, "
             f"best epoch {r['best_epoch']}, window {r['window_size']}, spec v{r['feature_spec_version']})")
    L.append("thresholds chosen on VAL: " + ", ".join(f"{k}: logit {v['logit']:+.3f}" for k, v in m["thresholds_from_val"].items()))
    for name, e in m["splits"].items():
        L.append(f"[{name}] windows={e['n_windows']} {e['class_counts']} ROC-AUC={_f(e['roc_auc'])} PR-AUC={_f(e['pr_auc'])} "
                 f"ECE raw={_f(e['calibration_raw']['ece'])} after val-T={_f(e['calibration_after_val_temperature']['ece'])}")
        for op, x in e["at"].items():
            L.append(f"    {op:10s} recall={_f(x['recall'])} FPR={_f(x['fpr'])} FNR={_f(x['fnr'])} precision={_f(x['precision'])} "
                     f"F1={_f(x['f1'])} confusion[[tn,fp],[fn,tp]]={x['confusion_matrix']}")
        bd = e.get(f"breakdown_at_{PRIMARY}")
        if bd:
            if "auc_excluding_local_infra" in bd:
                a = bd["auc_excluding_local_infra"]
                L.append(f"    AUC with local-infrastructure windows removed: ROC={_f(a['roc_auc'])} PR={_f(a['pr_auc'])} (n={a['n']})")
            if "per_flow_tag" in bd:
                L.append("    by flow tag (n / share positive / flagged share): " + "; ".join(
                    f"{k}: {v['n']}/{_f(v['frac_positive'], 2)}/{_f(v['flag_rate'], 2)}" for k, v in bd["per_flow_tag"].items()))
            if "per_benign_subset_fpr" in bd:
                L.append("    benign FPR by subset: " + "; ".join(f"{k}: {_f(v['flag_rate'])} (n={v['n']})"
                                                                for k, v in bd["per_benign_subset_fpr"].items()))
            if "per_category_recall" in bd:
                L.append("    recall by category: " + "; ".join(f"{k}: {_f(v['flag_rate'])} (n={v['n']})"
                                                               for k, v in bd["per_category_recall"].items()))
            fam = sorted(bd.get("per_family_recall", {}).items(), key=lambda kv: kv[1]["flag_rate"])
            if fam:
                L.append(f"    recall by family, lowest {min(max_groups, len(fam))} of {len(fam)}: " + "; ".join(
                    f"{k}: {_f(v['flag_rate'], 2)} (n={v['n']})" for k, v in fam[:max_groups]))
        for op, cl in e.get("capture_level", {}).items():
            rules = "; ".join(f"{k}: TPR={_f(v['capture_tpr'], 2)} FPR={_f(v['capture_fpr'], 2)} "
                              f"({v['n_benign_captures_flagged']}/{v['n_benign_captures']} benign captures flagged)"
                              for k, v in cl["rules"].items())
            L.append(f"    capture-level @{op}: captures={cl['n_captures']} AUC={_f(cl['auc_mean_logit'])} | {rules}")
    rows = m.get("early_prediction", {}).get("rows", [])
    if rows:
        L.append("    early prediction (threshold re-chosen on val for each k):")
        for e in rows:
            L.append(f"      [{e['split']}] k={e['k_packets']:>2} ROC-AUC={_f(e['roc_auc'])} recall={_f(e['recall'])} FPR={_f(e['fpr'])} "
                     f"mean packets={_f(e.get('mean_packets_needed'), 1)} median delay={_f(e.get('median_delay_s'))}s "
                     f"p90={_f(e.get('p90_delay_s'))}s")
    return "\n".join(L)
