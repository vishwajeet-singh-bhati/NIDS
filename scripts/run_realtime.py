"""Pre-train, fine-tune (QAT, prefix-augmented), evaluate offline AND by streaming replay; write ONE report.

    python scripts/run_realtime.py --raw-root data/raw/CIC-AndMal2017                       # all datasets from run_phase34
    python scripts/run_realtime.py --raw-root ... --datasets data/datasets/p34_full_family_holdout_s0 --epochs 25
    python scripts/run_realtime.py --raw-root ... --quick                                    # tiny budget, to check it runs

For each consolidated dataset <tag>_<strategy>_s<seed> (made by scripts/run_phase34.py) it runs
  1. pretrain  PacketBERT (label-free masked-packet modelling, train split only)
  2. finetune  packet_bert from the pretrained encoder   (run name: pb_pre)
  3. finetune  packet_bert from scratch                  (pb_scratch  - ablation: does pre-training help?)
  4. finetune  hybrid_resnet with prefix augmentation    (hybrid     - the existing QAT CNN baseline)
  5. stream replay of VAL (threshold selection) and TEST captures for every run
and collects everything into artifacts/realtime_report.txt (send me that file).
Manifest/splits are located from the dataset name (data/manifests/p34_<tag>.csv, data/splits/<name>.csv).
"""
import argparse
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MAX_CHARS = 20000


def run(title, cmd, out, keep_err=("WARNING", "ERROR", "Traceback", "Error", "epoch")):
    out.write(f"\n{'=' * 78}\n## {title}\n$ {' '.join(map(str, cmd))}\n{'=' * 78}\n")
    try:
        r = subprocess.run([str(c) for c in cmd], cwd=ROOT, capture_output=True, text=True, errors="replace")
    except Exception as exc:
        out.write(f"FAILED TO RUN: {exc}\n")
        print(f"{title}: FAILED {exc}")
        return -1
    err = [l for l in (r.stderr or "").splitlines() if any(k in l for k in keep_err)]
    out.write((r.stdout or "")[-MAX_CHARS:] + "\n--- log ---\n" + "\n".join(err[-30:]) + f"\n[exit code {r.returncode}]\n")
    out.flush()
    print(f"{title}: exit {r.returncode}")
    return r.returncode


def locate(dataset: str):
    name = Path(dataset).name
    m = re.match(r"^(?P<tag>.+)_(group_random|family_holdout(?:_noinfra)?)_s\d+$", name)
    tag = m.group("tag") if m else name
    man = ROOT / "data/manifests" / f"p34_{tag}.csv"
    sp = ROOT / "data/splits" / f"{name}.csv"
    return man, sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", required=True, help="raw dataset directory (manifest capture paths are relative to it)")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--pretrain-epochs", type=int, default=20)
    ap.add_argument("--max-steps-per-epoch", type=int, default=None)
    ap.add_argument("--target-fpr", type=float, default=0.01)
    ap.add_argument("--capture-stride", type=int, default=1, help="replay every K-th val/test capture (speed)")
    ap.add_argument("--no-ablations", action="store_true", help="only pb_pre")
    ap.add_argument("--with-hybrid", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-tests", action="store_true")
    a = ap.parse_args()
    py = sys.executable
    if a.quick:
        a.epochs, a.pretrain_epochs, a.max_steps_per_epoch, a.capture_stride = 3, 2, 50, max(a.capture_stride, 10)
    dsets = a.datasets or sorted(str(p.parent.relative_to(ROOT)) for p in (ROOT / "data/datasets").glob("*/normalization.json"))
    if not dsets:
        print("No consolidated datasets in data/datasets/. Run scripts/run_phase34.py first.")
        return 2
    rep = ROOT / "artifacts" / "realtime_report.txt"
    rep.parent.mkdir(exist_ok=True)
    with open(rep, "w", encoding="utf-8") as out:
        try:
            import torch
            env = f"torch {torch.__version__} cuda={torch.cuda.is_available()} mps={getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available()}"
        except ImportError:
            env = "torch NOT INSTALLED"
        out.write(f"python {sys.version}\nplatform {platform.platform()}\n{env}\ndatasets {dsets}\nargs {vars(a)}\n")
        if not a.skip_tests:
            run("0. unit tests", [py, "-m", "pytest", "-q"], out)
        for d in dsets:
            name = Path(d).name
            man, sp = locate(d)
            if not man.exists() or not sp.exists():
                out.write(f"\n!! skipping {name}: missing {man if not man.exists() else sp}\n")
                continue
            pre_dir = ROOT / "runs" / name / "pretrain_s0"
            cmd = [py, "-m", "src.training.pretrain", "--dataset", d, "--out", pre_dir, "--epochs", a.pretrain_epochs]
            if a.max_steps_per_epoch:
                cmd += ["--max-steps-per-epoch", a.max_steps_per_epoch]
            run(f"1. pretrain {name}", cmd, out)
            plan = [("pb_pre", "packet_bert", ["--init-from", pre_dir / "encoder.pt"])]
            if not a.no_ablations:
                plan.append(("pb_scratch", "packet_bert", []))
            if a.with_hybrid:
                plan.append(("hybrid", "hybrid_resnet", []))
            for tag, model, extra in plan:
                rd = ROOT / "runs" / name / f"{tag}_s0"
                if tag == "pb_pre" and not (pre_dir / "encoder.pt").exists():
                    out.write("\n!! pretraining produced no encoder; skipping pb_pre\n")
                    continue
                cmd = [py, "-m", "src.training.train", "--dataset", d, "--model", model, "--run-dir", rd,
                       "--epochs", a.epochs, "--prefix-aug", 0.5] + extra
                if a.max_steps_per_epoch:
                    cmd += ["--max-steps-per-epoch", a.max_steps_per_epoch]
                if run(f"2. finetune {name} / {tag}", cmd, out) != 0:
                    continue
                run(f"3. stream replay {name} / {tag}",
                    [py, "-m", "src.stream.replay_eval", "--run-dir", rd, "--manifest", man, "--splits", sp,
                     "--raw-root", a.raw_root, "--target-fpr", a.target_fpr, "--capture-stride", a.capture_stride], out,
                    keep_err=("WARNING", "ERROR", "Traceback", "Error"))
        out.write("\n" + "=" * 78 + "\n## summary of stream metrics\n" + "=" * 78 + "\n")
        for f in sorted((ROOT / "runs").rglob("stream_metrics.json")):
            try:
                m = json.loads(f.read_text())
                t = m["test"]
                fl, cl = t["flow_level"], t["capture_level"]
                out.write(f"{f.parent.parent.parent.name}/{f.parent.parent.name}: thr={m['threshold_selection']['threshold_logit']:.2f} "
                          f"flow acc={fl['accuracy']:.3f} P={fl['precision']:.3f} R={fl['recall']:.3f} F1={fl['f1']:.3f} FPR={fl['fpr']:.3f} | "
                          f"capture R={cl['recall']:.3f} benign-capture-FA={cl['benign_capture_false_alarm_rate']:.3f} "
                          f"alerts/benign-h={cl['alerts_per_benign_hour']:.1f}\n")
            except Exception as exc:  # noqa: BLE001
                out.write(f"{f}: unreadable ({exc})\n")
        for f in sorted((ROOT / "runs").rglob("stream_report.md")):
            out.write(f"\n--- {f.relative_to(ROOT)} ---\n{f.read_text()[:6000]}\n")
    print(f"\nDone. Send me: {rep}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
