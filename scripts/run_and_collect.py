"""One command that runs every Phase 2 check and writes ONE report file to send back.

    python scripts/run_and_collect.py --root data/raw/CIC-AndMal2017

Steps: pytest -> dataset audit -> label dry-run -> trial build (first N captures) ->
inspect processed tensors. Output: artifacts/phase2_report.txt (send me that file).
Cross-platform (Windows/Linux/macOS). Nothing is uploaded anywhere.
"""
import argparse
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_CHARS = 60000  # per step, keeps the report sendable


def step(title, cmd, out, tail_only=False):
    out.write(f"\n{'=' * 78}\n## {title}\n$ {' '.join(cmd)}\n{'=' * 78}\n")
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, errors="replace")
        text = (r.stdout or "") + (("\n--- stderr ---\n" + r.stderr) if r.stderr.strip() else "")
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS // 2] + "\n...[truncated]...\n" + text[-MAX_CHARS // 2:]
        out.write(text + f"\n[exit code {r.returncode}]\n")
        out.flush()
        print(f"{title}: exit {r.returncode}")
        return r.returncode
    except Exception as exc:
        out.write(f"FAILED TO RUN: {exc}\n")
        print(f"{title}: FAILED TO RUN {exc}")
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="raw dataset directory")
    ap.add_argument("--limit", type=int, default=5, help="captures for the trial build")
    ap.add_argument("--probe-packets", type=int, default=2000)
    ap.add_argument("--max-packets", type=int, default=200000,
                    help="cap packets per capture in the trial build (keeps it fast)")
    a = ap.parse_args()
    py = sys.executable
    (ROOT / "artifacts").mkdir(exist_ok=True)
    rep = ROOT / "artifacts" / "phase2_report.txt"
    with open(rep, "w", encoding="utf-8") as out:
        out.write(f"python {sys.version}\nplatform {platform.platform()}\nroot {a.root}\n")
        step("1. unit tests", [py, "-m", "pytest", "-q"], out)
        if step("2. dataset audit", [py, "-m", "src.data.inspect_dataset", "--root", a.root,
                                     "--out", "artifacts/reports/audit", "--probe-packets",
                                     str(a.probe_packets)], out) == 0:
            md = ROOT / "artifacts/reports/audit.md"
            if md.exists():
                out.write("\n--- audit.md (first 400 lines) ---\n")
                out.write("".join(md.read_text(encoding="utf-8").splitlines(True)[:400]))
        step("3. label dry-run", [py, "-m", "src.data.build_sequences", "--config", "configs/data.yaml",
                                  "--raw-root", a.root, "--dry-run-labels"], out)
        step("4. trial build", [py, "-m", "src.data.build_sequences", "--config", "configs/data.yaml",
                                "--raw-root", a.root, "--limit", str(a.limit),
                                "--max-packets-per-capture", str(a.max_packets), "--overwrite",
                                "--processed-root", "data/processed/trial",
                                "--manifest", "data/manifests/trial.csv"], out)
        step("5. inspect tensors", [py, "-m", "src.data.inspect_processed", "--manifest",
                                    "data/manifests/trial.csv", "--processed-root",
                                    "data/processed/trial", "--show", "3"], out)
        log = ROOT / "data/manifests/processing_log.jsonl"
        if log.exists():
            bad = [l for l in log.read_text(encoding="utf-8").splitlines()
                   if '"status": "ok"' not in l and '"shard_exists_manifest_rebuilt"' not in l]
            out.write("\n## non-ok processing_log lines\n" + ("\n".join(bad[:50]) or "none") + "\n")
    print(f"\nDone. Send me: {rep}")


if __name__ == "__main__":
    main()
