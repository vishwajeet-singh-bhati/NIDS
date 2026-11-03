"""Does a window's score depend on which other windows share its batch?  (matters because the live engine batches
whatever windows are ready, and the QAT activation scale is computed per batch)

    python scripts/check_batch_invariance.py --bundle models/nids/nids_v1 --dataset data/datasets/full_family_holdout_s0

Scores N test windows (a) one at a time, (b) in batches of 32, (c) in batches of 1024 and reports the largest and mean
absolute logit difference, plus how many windows would flip the flow-alert decision at the bundle's threshold.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.detection.nids.data.normalization import Normalizer  # noqa: E402
from src.detection.nids.models.model_factory import load_checkpoint  # noqa: E402
from src.detection.nids.training.data import SplitData  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    b = Path(a.bundle)
    model, ck = load_checkpoint(b / "model.pt", "cpu")
    thr = json.loads((b / "config.json").read_text())["threshold_logit"]
    d = SplitData(a.dataset, a.split, Normalizer(ck["normalization"]), max_examples=a.n, seed=0)
    kinds = set(model.inputs)
    rows = d.sel
    print(f"model {ck['model_name']}, QAT active: {bool(int(ck['state_dict'].get('qat_active_buf', torch.tensor(0))) == 1)}, windows {len(rows)}")

    def run(bs):
        out = []
        with torch.no_grad():
            for i in range(0, len(rows), bs):
                out.append(model({k: v for k, v in d.batch(rows[i:i + bs], kinds).items() if k not in ("y", "idx")}).numpy())
        return np.concatenate(out)
    s1, s32, s1024 = run(1), run(32), run(1024)
    for name, s in (("batch 32", s32), ("batch 1024", s1024)):
        diff = np.abs(s - s1)
        flips = int(((s >= thr) != (s1 >= thr)).sum())
        print(f"{name:10s} vs one-at-a-time: max |dlogit| {diff.max():.4f}, mean {diff.mean():.5f}, alert decision flips {flips} of {len(s)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
