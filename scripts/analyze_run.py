"""Look inside a finished run's device-level result (needs the saved replay traces; no model, no retraining).

    python scripts/analyze_run.py --run-dir runs/full_family_holdout_s0/final_s0

Prints: which benign captures raised device alarms (and which servers drove them), how close the missed/just-detected
malware captures were to the rule, per-family recall, and time to first alarm. Uses the thresholds and the device rule
that replay_eval selected on VAL; nothing is re-tuned here.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.stream.device_rule import BurstParams, alarm_times  # noqa: E402
from src.stream.policy import PolicyParams  # noqa: E402
from src.stream.replay_eval import apply_policy  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    sd = Path(a.run_dir) / "stream"
    m = json.loads((sd / "stream_metrics.json").read_text())
    sc, sel = m["stream_config"], m["device_level"]["selected"]
    p = PolicyParams(m["threshold_selection"]["threshold_logit"], sc["ema_alpha"], sc["min_hits"], sc["hard_margin"])
    bp = BurstParams(sel["window_s"], sel["min_alerts"])
    fl = apply_policy(pd.read_pickle(sd / f"traces_{a.split}.pkl"), p)
    fl["t_alert"] = fl["flow_start_ts"] + fl["delay_s"]
    rows = []
    for cid, g in fl.groupby("capture_id"):
        al = g[g.alert]
        alarms = alarm_times(al["t_alert"].tolist(), bp)
        t0 = g["flow_start_ts"].min()
        top_resp = al["responder_ip"].value_counts().head(3)
        rows.append({"capture_id": cid[-60:], "y": int(g["y"].iloc[0]), "family": g["malware_family"].iloc[0], "flows": len(g),
                     "flow_alerts": len(al), "alarm": int(len(alarms) > 0), "first_alarm_s": (alarms[0] - t0) if alarms else np.nan,
                     "span_s": g["flow_start_ts"].max() - t0, "top_responders": ", ".join(f"{k}x{v}" for k, v in top_resp.items())})
    c = pd.DataFrame(rows)
    print(f"{a.split}: {len(c)} captures with scored flows ({int((c.y == 1).sum())} malware, {int((c.y == 0).sum())} benign); "
          f"rule >= {bp.min_alerts} flow alerts in {bp.window_s:g}s; flow threshold {p.threshold:.2f}")
    fp = c[(c.y == 0) & (c.alarm == 1)].sort_values("flow_alerts", ascending=False)
    print(f"\nBENIGN captures that raised an alarm: {len(fp)} of {int((c.y == 0).sum())}")
    print(fp[["capture_id", "flows", "flow_alerts", "span_s", "top_responders"]].head(a.top).to_string(index=False))
    ben = c[c.y == 0].flow_alerts
    print(f"\nbenign flow-alert counts per capture: median {ben.median():.0f}, p90 {ben.quantile(.9):.0f}, p99 {ben.quantile(.99):.0f}, max {ben.max()}")
    mal = c[c.y == 1]
    print(f"malware flow-alert counts per capture: min {mal.flow_alerts.min()}, p10 {mal.flow_alerts.quantile(.1):.0f}, median {mal.flow_alerts.median():.0f}")
    print("\nlowest-alert MALWARE captures (closest to a miss):")
    print(mal.sort_values("flow_alerts")[["capture_id", "family", "flows", "flow_alerts", "alarm"]].head(6).to_string(index=False))
    print("\nper-family device recall:")
    print(mal.groupby("family").agg(captures=("alarm", "size"), detected=("alarm", "sum"), median_alerts=("flow_alerts", "median")).to_string())
    d = mal[mal.alarm == 1].first_alarm_s
    print(f"\ntime from first flow of a malware capture to its first device alarm: median {d.median():.1f}s, p90 {d.quantile(.9):.1f}s, max {d.max():.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
