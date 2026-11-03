"""Train + evaluate models on your consolidated datasets and collect ONE report file to send back.

    # after scripts/run_phase34.py has created data/datasets/<name>/ :
    python scripts/run_phase56.py                       # all datasets found, models mlp + cnn_meta
    python scripts/run_phase56.py --datasets data/datasets/trial20_family_holdout_s0 --epochs 8
    python scripts/run_phase56.py --with-payload        # also Model C (slower)
    python scripts/run_phase56.py --seeds 0 1 2         # repeat for mean/std

Steps: unit tests -> for each (dataset, model, seed): train + evaluate -> aggregate table.
Output: artifacts/phase56_report.txt (send me that file). Nothing is uploaded anywhere.
Runs are written to runs/<dataset>/<model>_s<seed>/ (metrics.json, history.csv, best.pt, predictions).
"""
import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MAX_CHARS = 30000


def run(title, cmd, out, tail=12):
    out.write(f"\n{'=' * 78}\n## {title}\n$ {' '.join(map(str, cmd))}\n{'=' * 78}\n")
    try:
        r = subprocess.run([str(c) for c in cmd], cwd=ROOT, capture_output=True, text=True, errors="replace")
    except Exception as exc:
        out.write(f"FAILED TO RUN: {exc}\n")
        print(f"{title}: FAILED {exc}")
        return -1
    err = [l for l in (r.stderr or "").splitlines() if any(k in l for k in ("WARNING", "ERROR", "Traceback", "Error", "epoch"))]
    text = (r.stdout or "")[-MAX_CHARS:] + "\n--- log (epochs/warnings/errors) ---\n" + "\n".join(err[-40:])
    out.write(text + f"\n[exit code {r.returncode}]\n")
    out.flush()
    print(f"{title}: exit {r.returncode}")
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None, help="consolidated dirs (default: every data/datasets/*)")
    ap.add_argument("--models", nargs="+", default=["mlp", "cnn_meta"])
    ap.add_argument("--with-payload", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-steps-per-epoch", type=int, default=None)
    ap.add_argument("--max-train-examples", type=int, default=None)
    ap.add_argument("--sampler", choices=["none", "balanced"], default=None)
    ap.add_argument("--skip-tests", action="store_true")
    a = ap.parse_args()
    py = sys.executable
    dsets = a.datasets or sorted(str(p.parent.relative_to(ROOT)) for p in (ROOT / "data/datasets").glob("*/normalization.json"))
    if not dsets:
        print("No consolidated datasets found under data/datasets/. Run scripts/run_phase34.py first.")
        return 2
    models = list(a.models) + (["cnn_payload"] if a.with_payload and "cnn_payload" not in a.models else [])
    rep = ROOT / "artifacts" / "phase56_report.txt"
    rep.parent.mkdir(exist_ok=True)
    with open(rep, "w", encoding="utf-8") as out:
        try:
            import torch
            env = f"torch {torch.__version__} cuda={torch.cuda.is_available()} threads={torch.get_num_threads()}"
        except ImportError:
            env = "torch NOT INSTALLED (pip install torch)"
        out.write(f"python {sys.version}\nplatform {platform.platform()}\n{env}\ndatasets {dsets}\nmodels {models} seeds {a.seeds}\n")
        if not a.skip_tests:
            run("1. unit tests", [py, "-m", "pytest", "-q"], out)
        for d in dsets:
            for m in models:
                for s in a.seeds:
                    cmd = [py, "-m", "src.training.train", "--dataset", d, "--model", m, "--seed", s]
                    for flag, v in (("--epochs", a.epochs), ("--max-steps-per-epoch", a.max_steps_per_epoch),
                                    ("--max-train-examples", a.max_train_examples), ("--sampler", a.sampler)):
                        if v is not None:
                            cmd += [flag, v]
                    run(f"2. train+evaluate {Path(d).name} / {m} / seed {s}", cmd, out)
        run("3. aggregate", [py, "scripts/aggregate_results.py"], out)
        from src.training.report import describe_run, load_runs
        out.write("\n" + "=" * 78 + "\n## 4. per-run digests\n" + "=" * 78 + "\n")
        for mj in load_runs(ROOT / "runs"):
            if Path(mj["run"]["dataset_dir"]).name in {Path(d).name for d in dsets}:
                out.write(describe_run(mj) + "\n\n")
        for hist in sorted((ROOT / "runs").rglob("history.csv"))[:12]:
            out.write(f"--- {hist.relative_to(ROOT)} ---\n{hist.read_text()[-1500:]}\n")
    print(f"\nDone. Send me: {rep}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
