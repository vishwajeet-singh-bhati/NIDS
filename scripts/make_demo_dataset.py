"""Build a small SYNTHETIC CIC-AndMal2017-like dataset end-to-end (raw pcaps -> windows -> split -> consolidated).
For plumbing/smoke tests of training, streaming and evaluation only. It says nothing about real detection quality.

    python scripts/make_demo_dataset.py --out data/demo [--strategy group_random|family_holdout]
Outputs: <out>/raw (pcaps), <out>/processed, <out>/manifest.csv, <out>/splits/<name>.csv, <out>/ds (consolidated).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/demo")
    ap.add_argument("--strategy", default="group_random", choices=["group_random", "family_holdout"])
    ap.add_argument("--n-benign", type=int, default=40)
    ap.add_argument("--n-families", type=int, default=8)
    ap.add_argument("--per-family", type=int, default=4)
    a = ap.parse_args(argv)
    from src.data import build_sequences, consolidate, splits
    from src.utils.synthetic import make_andmal_like
    out = Path(a.out)
    raw, proc, man, sp = out / "raw", out / "processed", out / "manifest.csv", out / "splits"
    make_andmal_like(raw, n_benign=a.n_benign, n_families=a.n_families, per_family=a.per_family)
    rc = build_sequences.main(["--config", str(ROOT / "configs/data.yaml"), "--label-config",
                               str(ROOT / "configs/labels.yaml"), "--raw-root", str(raw), "--processed-root", str(proc),
                               "--manifest", str(man), "--overwrite"])
    rc = rc or splits.main(["--manifest", str(man), "--name", "demo", "--strategy", a.strategy, "--seed", "0",
                            "--out-dir", str(sp)])
    rc = rc or consolidate.main(["--manifest", str(man), "--processed-root", str(proc), "--splits",
                                 str(sp / "demo.csv"), "--out", str(out / "ds")])
    print("consolidated dataset:", out / "ds")
    return rc


if __name__ == "__main__":
    sys.exit(main())
