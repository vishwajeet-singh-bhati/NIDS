"""Deployment bundle: everything the live detector needs, in ONE folder.

    python -m src.stream.bundle --run-dir runs/<ds>/final_s0 --out models/nids_v1

models/nids_v1/
    model.pt       checkpoint (weights, feature spec, train-fitted normaliser, QAT state)
    config.json    alert threshold, early-prediction checkpoints, flow policy, device-alarm rule (all selected on VAL)
    MODEL_CARD.md  what it was trained on, held-out metrics, and the honest limits

Run it:  python -m src.stream.run --bundle models/nids_v1 --iface <if>   (or --pcap file.pcap)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def export_bundle(run_dir: str | Path, out_dir: str | Path) -> Path:
    run_dir, out = Path(run_dir), Path(out_dir)
    sm_path = run_dir / "stream" / "stream_metrics.json"
    if not (run_dir / "best.pt").is_file() or not sm_path.is_file():
        raise FileNotFoundError(f"{run_dir} needs best.pt and stream/stream_metrics.json (run src.stream.replay_eval first)")
    sm = json.loads(sm_path.read_text())
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_dir / "best.pt", out / "model.pt")
    sc = sm["stream_config"]
    dl = sm.get("device_level", {})
    cfg: Dict[str, Any] = {
        "bundle_version": 1, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "model": sm["model"],
        "threshold_logit": sm["threshold_selection"]["threshold_logit"],
        "checkpoints": sc["checkpoints"], "min_hits": sc["min_hits"], "ema_alpha": sc["ema_alpha"], "hard_margin": sc["hard_margin"],
        "cooldown_s": sc["cooldown_s"],
        "device_rule": dl.get("selected", {"window_s": sc.get("device_window_s", 60.0), "min_alerts": sc.get("device_min_alerts", 10)}),
        "selected_on": "validation captures only", "label_meaning": "malicious_associated (weak capture-level label), not confirmed attack/C&C",
        "behaviour": "detection and logging only; nothing is blocked; no traffic leaves the process"}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    (out / "MODEL_CARD.md").write_text(model_card(sm, cfg))
    return out


def _f(x, nd=3) -> str:
    return "n/a" if x is None or x != x else f"{x:.{nd}f}"


def model_card(sm: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    t = sm["test"]
    fl, dv = t["flow_level"], sm.get("device_level", {}).get("test")
    L: List[str] = [f"# Model card - {cfg['model']}", "", f"Created {cfg['created']}.", "",
                    "**Task:** flag traffic that looks like malware-associated Android traffic (CIC-AndMal2017 weak labels). "
                    "Detection and logging only.", "", "## Operating point (chosen on validation, applied unchanged to test)", "",
                    f"- flow alert threshold: logit {cfg['threshold_logit']:.3f} (early decisions after {cfg['checkpoints']} packets)",
                    f"- device alarm: >= {cfg['device_rule']['min_alerts']} flow alerts within {cfg['device_rule']['window_s']:g} s", "",
                    "## Held-out test results (replay of unseen captures through the live engine)", "",
                    "| level | accuracy | precision | recall | F1 | false-alarm |", "|---|---|---|---|---|---|",
                    f"| single flow | {_f(fl['accuracy'])} | {_f(fl['precision'])} | {_f(fl['recall'])} | {_f(fl['f1'])} | FPR {_f(fl['fpr'])} |"]
    if dv:
        L.append(f"| device (capture) | {_f(dv['accuracy'])} | {_f(dv['precision'])} | {_f(dv['recall'])} | {_f(dv['f1'])} | "
                 f"{_f(dv['benign_capture_false_alarm_rate'])} of benign captures; {_f(dv['alarms_per_benign_hour'], 2)} alarms / benign hour |")
        L.append(f"\nDevice-level test set: {dv['n_captures']} captures (tp {dv['tp']}, fp {dv['fp']}, fn {dv['fn']}, tn {dv['tn']}).")
    if dv and dv.get("time_to_first_alarm_s"):
        d = dv["time_to_first_alarm_s"]
        L.append(f"\nTime from the first flow of a malware capture to its first device alarm: median {_f(d['median'], 1)} s, p90 {_f(d['p90'], 1)} s "
                 f"({d['n']} detected malware captures). The alarm needs a burst of flow alerts, so it is much slower than a single flow alert.")
    ttd = t.get("time_to_detect_true_positive_flows")
    if ttd:
        L.append(f"\nTime to detect (true-positive flows): median {_f(ttd['median_s'])} s / {_f(ttd['median_packets'], 0)} packets; "
                 f"p90 {_f(ttd['p90_s'])} s / {_f(ttd['p90_packets'], 0)} packets.")
    L += ["", "## Limits - read before trusting it", "",
          "- Labels are per capture: benign background traffic inside malware captures counts as malicious, so flow-level recall is capped.",
          "- Benign and malware captures were recorded on largely different dates; the model may partly learn capture-time artefacts.",
          "- Device-level numbers come from the capture counts above (a handful of malware families); seeds share the same split, so spread over seeds "
          "reflects training noise only, not split-to-split variation.",
          "- Precision at device level depends on how common malware is: the test set is roughly one-fifth malware; with rarer malware, precision falls.",
          "- Payload bytes are mostly ciphertext; the signal may come from handshake fields that identify apps rather than maliciousness.",
          "- Not a production-grade detector. Not validated on other datasets or on live device traffic."]
    return "\n".join(L) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        p = export_bundle(a.run_dir, a.out)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"bundle written: {p}  ({', '.join(sorted(x.name for x in p.iterdir()))})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
