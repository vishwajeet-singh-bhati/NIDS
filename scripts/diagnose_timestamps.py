"""Explain WHY a capture has out-of-order timestamps (read-only diagnostic).

    python scripts/diagnose_timestamps.py data/raw/CIC-AndMal2017/Adware/Dowgin/<file>.pcap
    python scripts/diagnose_timestamps.py <file.pcap> --max-packets 300000

Reports how many packets are earlier than a previously seen packet, by how much, and where the
first few occur, so we can tell jitter (harmless, handled by the window clock) from something
structural (e.g. several interleaved capture clocks).
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.parse_pcap import ParseStats, iter_packets  # noqa: E402
from src.utils.config import ParseConfig  # noqa: E402

BUCKETS = [(1e-6, "<1us"), (1e-3, "<1ms"), (1e-2, "<10ms"), (1e-1, "<100ms"), (1.0, "<1s"),
           (60.0, "<1min"), (float("inf"), ">=1min")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pcap")
    ap.add_argument("--max-packets", type=int, default=0)
    ap.add_argument("--show", type=int, default=8)
    a = ap.parse_args()
    st = ParseStats()
    cfg = ParseConfig(payload_bytes_kept=0, max_packets_per_capture=a.max_packets)
    mx = None
    n = back = 0
    sizes, examples = Counter(), []
    prev_ts = None
    run_len = cur_run = 0
    for i, p in enumerate(iter_packets(a.pcap, cfg, st)):
        n += 1
        if mx is not None and p.ts < mx:
            back += 1
            d = mx - p.ts
            for lim, name in BUCKETS:
                if d < lim:
                    sizes[name] += 1
                    break
            if len(examples) < a.show:
                examples.append((i, p.ts, mx, d, f"{p.src_ip}:{p.sport}->{p.dst_ip}:{p.dport}"))
            cur_run += 1
            run_len = max(run_len, cur_run)
        else:
            cur_run = 0
        mx = p.ts if mx is None else max(mx, p.ts)
        prev_ts = p.ts
    print(f"file: {a.pcap}\nIP packets parsed: {n}  file records: {st.total_records}  format: {st.file_format}")
    print(f"packets earlier than the running max: {back} ({(back / n if n else 0):.2%})")
    print(f"longest run of consecutive backward packets: {run_len}")
    print("size of the backward step (running max - ts):")
    for _, name in BUCKETS:
        print(f"  {name:8s} {sizes.get(name, 0)}")
    print("first examples (packet#, ts, running_max, delta_s, flow):")
    for e in examples:
        print(f"  #{e[0]:7d} ts={e[1]:.6f} max={e[2]:.6f} delta={e[3]:.6f}s {e[4]}")
    print(f"non_monotonic_timestamp counter in ParseStats: {st.non_monotonic_timestamp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
