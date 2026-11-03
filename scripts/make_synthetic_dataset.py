"""Create a tiny synthetic dataset for smoke-testing the pipeline.

    python scripts/make_synthetic_dataset.py --out data/raw/SYNTHETIC

Layout (directory names deliberately match the *template* rules in configs/labels.yaml.synthetic):
    SYNTHETIC/benign/app_a/capture1.pcap
    SYNTHETIC/benign/app_b/capture1.pcap
    SYNTHETIC/malware_like/family_x/capture1.pcap
    SYNTHETIC/malware_like/family_y/capture1.pcap

SYNTHETIC DATA: the "malware_like" captures are just periodic request/response shapes. They have
no relation to real malware and say nothing about detector quality.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.synthetic import (beacon_shaped_capture, benign_shaped_capture, make_andmal_like,  # noqa: E402
                                 write_pcap)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/SYNTHETIC")
    ap.add_argument("--andmal-like", action="store_true", help="larger AndMal2017-like layout for split tests")
    ap.add_argument("--n-benign", type=int, default=40)
    ap.add_argument("--n-families", type=int, default=8)
    ap.add_argument("--per-family", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out)
    if a.andmal_like:
        make_andmal_like(out, a.n_benign, a.n_families, a.per_family)
        print(f"Wrote AndMal-like synthetic dataset to {out}")
        return 0
    write_pcap(out / "benign/app_a/capture1.pcap", benign_shaped_capture(seed=1))
    write_pcap(out / "benign/app_b/capture1.pcap", benign_shaped_capture(t0=1_700_050_000.0, seed=2))
    write_pcap(out / "malware_like/family_x/capture1.pcap", beacon_shaped_capture(seed=1))
    write_pcap(out / "malware_like/family_y/capture1.pcap", beacon_shaped_capture(t0=1_700_200_000.0, seed=2))
    print(f"Wrote synthetic dataset to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
