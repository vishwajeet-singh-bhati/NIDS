"""Aggregate every runs/**/metrics.json into one comparison table (CSV + Markdown) and show which planned
experiments are still pending.

    python scripts/aggregate_results.py --runs-root runs --out results/results
Outputs: <out>.csv, <out>.md. Nothing is invented: a planned row without a finished run says "pending".
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.report import load_runs, markdown_table, run_row  # noqa: E402

COLS = ["dataset", "model", "seed", "params", "window", "test_windows", "test_roc_auc", "test_pr_auc",
        "recall@val_fpr1%", "test_fpr@val_fpr1%", "precision@val_fpr1%", "recall@val_fpr5%", "test_fpr@val_fpr5%",
        "auc_excl_local_infra", "capture_auc", "benign_captures_flagged@rate0.5", "recall@8pkts", "median_delay_s@8pkts"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--out", default="results/results")
    ap.add_argument("--plan", default="configs/experiments.yaml")
    a = ap.parse_args()
    runs = load_runs(a.runs_root)
    rows = [run_row(m) for m in runs]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    md = ["# Results (generated; test metrics at thresholds chosen on validation)", "",
          "Labels are capture-level weak labels. Windows from the same capture are correlated; see capture-level columns.", ""]
    if df.empty:
        md.append("No finished runs found.")
    else:
        df.to_csv(out.with_suffix(".csv"), index=False)
        md += ["## All runs", "", markdown_table(rows, COLS), ""]
        num = [c for c in df.columns if df[c].dtype.kind == "f"]
        if df.groupby(["dataset", "model"]).size().max() > 1:
            g = df.groupby(["dataset", "model"])[num].agg(["mean", "std"]).round(4)
            g.columns = [f"{a}_{b}" for a, b in g.columns]
            md += ["## Mean/std over seeds", "", g.reset_index().to_markdown(index=False), ""]
    if Path(a.plan).is_file():
        plan = yaml.safe_load(Path(a.plan).read_text())["experiments"]
        md += ["## Experiment plan status", "", "| id | question | dataset pattern | model | status | test PR-AUC | recall@val FPR1% | test FPR |", "|---|---|---|---|---|---:|---:|---:|"]
        for e in plan:
            hit = [r for r in rows if e["dataset"] in r["dataset"] and r["model"] == e["model"]]
            # `family_holdout` must not swallow `family_holdout_noinfra`
            if e["dataset"] == "family_holdout":
                hit = [r for r in hit if "noinfra" not in r["dataset"]]
            if hit:
                r = hit[-1]
                md.append(f"| {e['id']} | {e['question']} | {e['dataset']} | {e['model']} | done ({len(hit)} run(s)) | "
                          f"{r['test_pr_auc']:.3f} | {r['recall@val_fpr1%']:.3f} | {r['test_fpr@val_fpr1%']:.3f} |")
            else:
                md.append(f"| {e['id']} | {e['question']} | {e['dataset']} | {e['model']} | **pending** | - | - | - |")
    out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
