"""Run the real-time detector.

    # replay one or more PCAPs (also works with a named pipe: mkfifo /tmp/cap; tcpdump -U -w /tmp/cap)
    python -m src.stream.run --run-dir runs/<ds>/packet_bert_s0 --pcap a.pcap b.pcap --alerts alerts.jsonl

    # live capture from an interface on this machine (needs capture privileges, e.g. sudo / root on a rooted phone)
    sudo python -m src.stream.run --run-dir runs/<ds>/packet_bert_s0 --iface en0 --alerts alerts.jsonl

The threshold comes from `<run-dir>/stream/stream_metrics.json` (selected on VAL by src.stream.replay_eval)
unless --threshold is given. Detection and logging only: nothing is blocked, nothing is uploaded.
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import List, Optional

from src.stream.engine import StreamConfig, StreamDetector
from src.utils.logging_utils import setup_logging

log = logging.getLogger("stream.run")

# scapy class name -> pcap link type understood by src.data.parse_pcap
_LINKTYPES = {"Ether": 1, "CookedLinux": 113, "CookedLinuxV2": 276, "IP": 101, "IPv6": 101, "Raw": 101}


def _threshold_from_run(run_dir: Path) -> Optional[float]:
    f = run_dir / "stream" / "stream_metrics.json"
    if f.is_file():
        return float(json.loads(f.read_text())["threshold_selection"]["threshold_logit"])
    return None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pcap", nargs="+")
    src.add_argument("--iface", help="live capture interface (scapy); 'any' on Linux/Android")
    ap.add_argument("--alerts", default="alerts.jsonl")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--checkpoints", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--min-hits", type=int, default=2)
    ap.add_argument("--bpf", default=None, help="optional BPF filter for live capture, e.g. 'not port 22'")
    ap.add_argument("--stats-every", type=float, default=30.0, help="seconds between stats lines (live)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args(argv)
    setup_logging(a.log_level)
    run_dir = Path(a.run_dir)
    thr = a.threshold if a.threshold is not None else _threshold_from_run(run_dir)
    if thr is None:
        log.error("no threshold: run src.stream.replay_eval first (it selects one on validation) or pass --threshold")
        return 2
    sc = StreamConfig(checkpoints=tuple(a.checkpoints), min_hits=a.min_hits, threshold=thr)
    sink = open(a.alerts, "a", buffering=1)
    det = StreamDetector(str(run_dir / "best.pt"), stream_cfg=sc, device=a.device, alert_sink=sink, config_path=a.data_config)
    log.info("model=%s threshold(logit)=%.3f checkpoints=%s -> alerts: %s", det.ck["model_name"], thr, sc.checkpoints, a.alerts)
    try:
        if a.pcap:
            for p in a.pcap:
                t0 = time.perf_counter()
                n0 = det.stats["packets"]
                alerts = det.feed_pcap(p)
                dt = time.perf_counter() - t0
                log.info("%s: %d packets, %d alerts, %.0f packets/s", p, det.stats["packets"] - n0, len(alerts),
                         (det.stats["packets"] - n0) / max(dt, 1e-9))
            print(json.dumps(det.summary(), indent=2))
            return 0
        from scapy.all import AsyncSniffer  # imported lazily: only live capture needs it
        last = [time.time()]
        stop = {"flag": False}

        def on_pkt(p) -> None:
            lt = _LINKTYPES.get(type(p).__name__)
            if lt is None:
                return
            det.feed_raw(bytes(p), lt, float(p.time), len(p))
            if time.time() - last[0] > a.stats_every:
                last[0] = time.time()
                log.info("stats %s", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in det.summary().items()}))

        def _sig(*_):
            stop["flag"] = True
        signal.signal(signal.SIGINT, _sig)
        signal.signal(signal.SIGTERM, _sig)
        sn = AsyncSniffer(iface=a.iface, prn=on_pkt, store=False, filter=a.bpf)
        sn.start()
        log.info("capturing on %s ... Ctrl-C to stop", a.iface)
        while not stop["flag"]:
            time.sleep(0.5)
            det.run_inference()          # flush the micro-batch even when traffic is quiet
        sn.stop()
        det.finish()
        print(json.dumps(det.summary(), indent=2))
        return 0
    finally:
        sink.close()


if __name__ == "__main__":
    sys.exit(main())
