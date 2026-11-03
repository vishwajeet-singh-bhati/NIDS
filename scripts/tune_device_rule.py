"""Re-select the device-alarm rule (N flow alerts within T s) from a finished run's saved traces - no model, no replay.

    python scripts/tune_device_rule.py --run-dir runs/full_family_holdout_s0/final_s0 --target-fa 0.01 0.02 0.05
    python scripts/tune_device_rule.py --run-dir ... --target-fa 0.01 --write     # store that choice in stream_metrics.json
    python -m src.stream.bundle --run-dir ... --out models/nids_v1                 # then refresh the bundle + model card

The rule is selected on VAL captures only (max recall subject to benign-capture false-alarm rate <= target) and applied
unchanged to TEST. Flow threshold stays the one replay_eval chose on val. Capture durations are approximated by the span
of their flow start times, so 'alarms per benign hour' is approximate here.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.stream.policy import PolicyParams  # noqa: E402
from src.stream.replay_eval import device_level  # noqa: E402


def load(sd: Path, split: str):
    f = pd.read_pickle(sd / f"traces_{split}.pkl")
    g = f.groupby("capture_id")
    caps = g.agg(binary_label=("y", "first")).reset_index()
    cs = g["flow_start_ts"].agg(lambda x: float(x.max() - x.min())).rename("duration_s").reset_index()
    return {"flows": f, "caps": caps, "captures": cs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--target-fa", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05])
    ap.add_argument("--write", action="store_true", help="store the LAST --target-fa choice in stream_metrics.json")
    a = ap.parse_args()
    sd = Path(a.run_dir) / "stream"
    m = json.loads((sd / "stream_metrics.json").read_text())
    sc = m["stream_config"]
    p = PolicyParams(m["threshold_selection"]["threshold_logit"], sc["ema_alpha"], sc["min_hits"], sc["hard_margin"])
    R = {"val": load(sd, "val"), "test": load(sd, "test")}
    print(f"flow threshold {p.threshold:.2f}; val {len(R['val']['caps'])} / test {len(R['test']['caps'])} captures\n")
    print("target FA | rule (alerts/window) | VAL recall  FA | TEST acc  prec  recall  F1  benign-FA  alarms/benign-h | median s to first alarm")
    last = None
    for fa in a.target_fa:
        d = device_level(R, p, "test", target_fa=fa)
        v, t, s = d["val"], d["test"], d["selected"]
        tt = (t.get("time_to_first_alarm_s") or {}).get("median", float("nan"))
        print(f"{fa:8.3f} | {s['min_alerts']:>3}/{s['window_s']:>5g}s | {v['recall']:.3f} {v['benign_capture_false_alarm_rate']:.3f} | "
              f"{t['accuracy']:.3f} {t['precision']:.3f} {t['recall']:.3f} {t['f1']:.3f} {t['benign_capture_false_alarm_rate']:.3f} "
              f"{t['alarms_per_benign_hour']:.2f} | {tt:.0f}" + (f"   [{d['note']}]" if d.get("note") else ""))
        last = d
    if a.write and last is not None:
        last["note"] = (last.get("note", "") + " recomputed offline from saved traces (durations approximate)").strip()
        m["device_level"] = last
        (sd / "stream_metrics.json").write_text(json.dumps(m, indent=2, default=lambda o: None if o != o else str(o)))
        print("\nwrote device_level to stream_metrics.json; now run: python -m src.stream.bundle --run-dir", a.run_dir, "--out <bundle dir>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
