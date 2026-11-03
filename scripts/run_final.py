"""THE pipeline for the single deployable model: train -> choose thresholds on val -> test by replay -> export bundle.

    python scripts/run_final.py --raw-root data/raw/CIC-AndMal2017 --dataset data/datasets/full_family_holdout_s0 --seeds 0 1 2

Model: hybrid_resnet (payload + metadata + aggregate features, INT8 quantisation-aware training, prefix-augmented so it
decides after 8/16/32 packets). Chosen because it matched the BERT-style model in the first real runs while being
smaller, simpler to INT8-export and needing no pre-training.

What it does, per seed:
  1. train (val-only model selection) -> runs/<dataset>/final_s<seed>/
  2. replay val captures (pick flow threshold + device rule) and test captures (apply unchanged), through the live engine
Then: pick the seed with the best VAL flow ROC-AUC (never looks at test), export it to --bundle-out, write
artifacts/final_report.txt with every seed's test metrics (mean +/- std shows how stable the numbers are).

Datasets: validate generalisation on a *family_holdout* dataset (test families never seen in training).
For the model you actually deploy, run it again on a *group_random* dataset so every family is in training (its test
numbers are then optimistic because families are seen - quote the family_holdout numbers as the honest ones).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--dataset", required=True, help="consolidated dataset dir, e.g. data/datasets/full_family_holdout_s0")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--max-steps-per-epoch", type=int, default=0)
    ap.add_argument("--target-fpr", type=float, default=0.01)
    ap.add_argument("--device-fa", type=float, default=0.05)
    ap.add_argument("--capture-stride", type=int, default=1)
    ap.add_argument("--bundle-out", default="models/nids/nids_v1")
    ap.add_argument("--device", default="auto", help="training device: auto|cpu|cuda")
    ap.add_argument("--skip-training-if-exists", action="store_true", help="reuse runs/<ds>/final_s<seed>/best.pt")
    a = ap.parse_args()
    from src.detection.nids.stream import replay_eval
    from src.detection.nids.stream.bundle import export_bundle
    from src.detection.nids.stream.engine import StreamConfig
    from src.detection.nids.training.train import train
    from src.common.logger import setup_logging
    setup_logging("INFO")
    ds = a.dataset
    name = Path(ds).name
    man, sp = replay_eval.locate_inputs(ds, ROOT)
    for f in (man, sp, Path(ds) / "normalization.json"):
        if not Path(f).exists():
            print(f"missing {f}; run scripts/run_phase34.py first")
            return 2
    rows = []
    for seed in a.seeds:
        rd = ROOT / "runs" / name / f"final_s{seed}"
        if not (a.skip_training_if_exists and (rd / "best.pt").exists()):
            cfg = {"seed": seed, "epochs": a.epochs, "prefix_aug": 0.5}
            if a.max_steps_per_epoch:
                cfg["max_steps_per_epoch"] = a.max_steps_per_epoch
            train(ds, "hybrid_resnet", rd, cfg, device=a.device)
        res = replay_eval.run(str(rd), str(man), str(sp), a.raw_root, str(rd / "stream"), a.target_fpr, "fpr", StreamConfig(),
                              "cpu", 0, a.capture_stride, device_fa=a.device_fa)
        rows.append({"seed": seed, "run": str(rd), "val_roc": res["val"]["flow_level"]["roc_auc_of_max_window_logit"],
                     "test_flow": res["test"]["flow_level"], "test_device": res["device_level"]["test"],
                     "rule": res["device_level"]["selected"]})
    best = max(rows, key=lambda r: (r["val_roc"] if r["val_roc"] == r["val_roc"] else -1))
    out = export_bundle(best["run"], ROOT / a.bundle_out)

    def ms(key, sub):
        v = np.array([r[sub][key] for r in rows], dtype=float)
        return f"{np.nanmean(v):.3f} +/- {np.nanstd(v):.3f}"
    L = [f"FINAL MODEL REPORT - hybrid_resnet on {name}", f"seeds {a.seeds}; bundle seed {best['seed']} chosen by VAL flow ROC-AUC only", "",
         "per seed (test split, thresholds/rule chosen on val):",
         "seed | flow F1 | flow P | flow R | flow FPR | device rule | device acc | device P | device R | device F1 | benign-capture FA"]
    for r in rows:
        f, d = r["test_flow"], r["test_device"]
        L.append(f"{r['seed']} | {f['f1']:.3f} | {f['precision']:.3f} | {f['recall']:.3f} | {f['fpr']:.3f} | "
                 f"{r['rule']['min_alerts']}/{r['rule']['window_s']:g}s | {d['accuracy']:.3f} | {d['precision']:.3f} | {d['recall']:.3f} | "
                 f"{d['f1']:.3f} | {d['benign_capture_false_alarm_rate']:.3f}")
    if len(rows) > 1:
        L += ["", "mean +/- std over seeds: flow F1 " + ms("f1", "test_flow") + "; flow recall " + ms("recall", "test_flow") +
              "; device acc " + ms("accuracy", "test_device") + "; device recall " + ms("recall", "test_device") +
              "; device benign FA " + ms("benign_capture_false_alarm_rate", "test_device")]
    L += ["", f"bundle: {out}", "", (out / "MODEL_CARD.md").read_text()]
    rep = ROOT / "artifacts" / "final_report.txt"
    rep.parent.mkdir(exist_ok=True)
    rep.write_text("\n".join(L))
    print("\n".join(L))
    print(f"\nDone. Send me: {rep}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
