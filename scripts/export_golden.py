"""Export the runtime config + golden test vectors that the Kotlin/Android port is checked against.

    python scripts/export_golden.py --bundle models/nids_deploy_v1 --dataset data/datasets/full_group_random_s0 \
        --pcap <some.pcap> [--pcap <other.pcap>] --out android/golden [--max-frames 6000] [--snaplen 300]

Writes into --out:
  runtime_config.txt   everything the phone needs besides the .tflite: window/payload size, alert threshold, policy,
                       device rule, train-fitted normalisation constants (plain key=value lines, no JSON needed)
  frames.txt           the raw frames fed to the detector:  <ts> <linktype> <wirelen> <hex>      (cut to --snaplen bytes,
                       like `tcpdump -s 300` on the phone, so the Python reference sees exactly what the phone sees)
  windows.txt          every window the Python StreamDetector scored, in scoring order:
                       W <flow_index> <start> <k> <logit>          (+ for the first --n-tensors windows the model inputs:
                       M <12*32 floats, channels-first>, A <22 floats>, P <128*32 ints channels-first>, MASK <32 floats>)
  events.txt           flow_alert / device_alarm events the reference produced
The reference is the real src.stream.StreamDetector (PyTorch model, same bundle config). Nothing is re-implemented here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scapy.utils import RawPcapReader  # noqa: E402

from src.data.parse_pcap import _timestamp  # noqa: E402
from src.stream.engine import StreamConfig, StreamDetector  # noqa: E402
from src.training.data import tensors_from_arrays  # noqa: E402


def fl(a) -> str:
    return " ".join(f"{float(x):.9g}" for x in np.asarray(a).ravel())


def runtime_config(det: StreamDetector, sc: StreamConfig, cooldown_s: float) -> str:
    spec = det.norm.spec
    ch = spec["meta"]["channels"]
    names = spec["meta_channels"]
    cfg = det.cfg
    lines = {
        "window": det.W, "stride": det.S, "payload_bytes": det.payload_bytes,
        "threshold_logit": sc.threshold, "checkpoints": ",".join(map(str, sc.checkpoints)),
        "ema_alpha": sc.ema_alpha, "min_hits": sc.min_hits, "hard_margin": sc.hard_margin,
        "max_scored_windows": sc.max_scored_windows, "final_min_packets": sc.final_min_packets,
        "cooldown_s": cooldown_s, "device_window_s": sc.device_window_s, "device_min_alerts": sc.device_min_alerts,
        "max_active_flows": sc.max_active_flows, "drain_every": sc.drain_every,
        "iat_log_unit_ms": int(cfg.sequence.iat_log_unit_ms), "dns_ports": ",".join(map(str, cfg.parse.dns_ports)),
        "tcp_idle_timeout_s": cfg.flow.tcp_idle_timeout_s, "udp_idle_timeout_s": cfg.flow.udp_idle_timeout_s,
        "other_idle_timeout_s": cfg.flow.other_idle_timeout_s, "max_flow_duration_s": cfg.flow.max_flow_duration_s,
        "tcp_time_wait_s": cfg.flow.tcp_time_wait_s, "max_packets_per_flow": cfg.flow.max_packets_per_flow,
        "drop_tcp_retransmissions": int(cfg.flow.drop_tcp_retransmissions),
        "expire_check_interval_s": cfg.flow.expire_check_interval_s,
        "meta_channels": ",".join(names),
        "meta_transform": ",".join(ch[c]["transform"] for c in names),
        "meta_mean": fl([ch[c]["mean"] for c in names]), "meta_std": fl([ch[c]["std"] for c in names]),
        "agg_names": ",".join(spec["aggregate"]["names"]), "agg_transform": ",".join(spec["aggregate"]["transforms"]),
        "agg_mean": fl(spec["aggregate"]["mean"]), "agg_std": fl(spec["aggregate"]["std"]),
        "model": det.ck["model_name"],
    }
    return "\n".join(f"{k}={v}" for k, v in lines.items()) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--dataset", default="", help="unused when the bundle's checkpoint carries its normaliser (kept for symmetry)")
    ap.add_argument("--pcap", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=6000, help="frames per pcap")
    ap.add_argument("--snaplen", type=int, default=300)
    ap.add_argument("--n-tensors", type=int, default=40)
    ap.add_argument("--threshold", type=float, default=None, help="override (toy bundles only)")
    ap.add_argument("--device-min-alerts", type=int, default=None, help="override (toy bundles only)")
    a = ap.parse_args()

    b = Path(a.bundle)
    bc = json.loads((b / "config.json").read_text())
    kw = dict(checkpoints=tuple(bc.get("checkpoints", (8, 16, 32))), threshold=float(bc["threshold_logit"]))
    for k_src, k_dst in (("min_hits", "min_hits"), ("ema_alpha", "ema_alpha"), ("hard_margin", "hard_margin"), ("cooldown_s", "cooldown_s")):
        if k_src in bc:
            kw[k_dst] = bc[k_src]
    dr = bc.get("device_rule")
    if dr:
        kw["device_window_s"], kw["device_min_alerts"] = float(dr["window_s"]), int(dr["min_alerts"])
    if a.threshold is not None:
        kw["threshold"] = a.threshold
    if a.device_min_alerts is not None:
        kw["device_min_alerts"] = a.device_min_alerts
    sc = StreamConfig(**kw)
    det = StreamDetector(str(b / "model.pt"), stream_cfg=sc)
    det.record_scores = True

    tensors = {}   # scoring-order index -> model inputs, for the first N windows (flow_index repeats across pcaps)
    counter = [0]
    orig = det.run_inference

    def wrapped():
        for p in det.queue:
            if counter[0] < a.n_tensors:
                t = tensors_from_arrays(p.window["meta"][None], p.window["mask"][None], p.window["payload"][None],
                                        p.window["payload_valid_len"][None], det.norm, det.kinds)
                tensors[counter[0]] = {k: v.numpy()[0] for k, v in t.items()}
            counter[0] += 1
        return orig()

    det.run_inference = wrapped

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    events = []
    nframes = 0
    with open(out / "frames.txt", "w") as fr:
        for pi, pcap in enumerate(a.pcap):
            if pi:
                events += det.finish()
                det.reset_capture()
            rd = RawPcapReader(pcap)
            lt = getattr(rd, "linktype", 1)
            nano = bool(getattr(rd, "nano", False))
            n = 0
            for data, meta in rd:
                if n >= a.max_frames:
                    break
                ts = _timestamp(meta, nano)
                buf = bytes(data[:a.snaplen])
                wl = int(getattr(meta, "wirelen", len(data)))
                fr.write(f"{ts:.6f} {lt} {wl} {buf.hex()}\n")
                events += det.feed_raw(buf, lt, ts, wl)
                n += 1
                nframes += 1
            rd.close()
            fr.write("END\n")          # capture boundary: the Kotlin side calls finish() + reset here
    events += det.finish()

    with open(out / "windows.txt", "w") as w:
        for i, s in enumerate(det.scores):
            key = (s["flow_index"], s["start"], s["k"])
            w.write(f"W {key[0]} {key[1]} {key[2]} {s['logit']:.9g}\n")
            t = tensors.get(i)
            if t is not None:
                w.write("M " + fl(t["meta"].transpose(1, 0)) + "\n")
                w.write("A " + fl(t["agg"]) + "\n")
                w.write("P " + " ".join(str(int(x)) for x in t["payload"].transpose(1, 0).ravel()) + "\n")
                w.write("MASK " + fl(t["mask"]) + "\n")
    with open(out / "events.txt", "w") as e:
        for ev in events:
            if ev["event"] == "device_alarm":
                e.write(f"device_alarm {ev['ts']:.6f}\n")
            else:
                e.write(f"flow_alert {ev['flow_index']} {ev['reason']} {ev['ts']:.6f} {ev['window_logit']}\n")
    (out / "runtime_config.txt").write_text(runtime_config(det, sc, sc.cooldown_s))
    n_alert = sum(1 for e in events if e["event"] == "flow_alert")
    n_dev = sum(1 for e in events if e["event"] == "device_alarm")
    print(f"frames {nframes} | scored windows {len(det.scores)} | tensors saved {len(tensors)} | flow alerts {n_alert} | device alarms {n_dev} -> {out}")


if __name__ == "__main__":
    main()
