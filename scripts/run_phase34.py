"""Run Phases 2-4 on your real dataset and collect ONE report file to send back.

    # quick trial (~5% of captures spread over all classes; minutes):
    python scripts/run_phase34.py --root data/raw/CIC-AndMal2017

    # everything (long; uses a per-capture cap so it fits in RAM/disk):
    python scripts/run_phase34.py --root data/raw/CIC-AndMal2017 --full --workers 4

Steps: unit tests -> build windows -> invariant checks -> label audit -> splits (random-capture,
family-holdout, family-holdout without local infra) -> consolidate two of them (train-only
normaliser) -> summary. Output: artifacts/phase34_report.txt  (send me that file).
Nothing is uploaded anywhere. Names/paths are relative to the repo root; run from there.
"""
import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_CHARS = 40000


def step(title, cmd, out, keep="head+tail"):
    out.write(f"\n{'=' * 78}\n## {title}\n$ {' '.join(map(str, cmd))}\n{'=' * 78}\n")
    try:
        r = subprocess.run([str(c) for c in cmd], cwd=ROOT, capture_output=True, text=True, errors="replace")
    except Exception as exc:
        out.write(f"FAILED TO RUN: {exc}\n")
        print(f"{title}: FAILED TO RUN {exc}")
        return -1
    text = r.stdout or ""
    err = r.stderr or ""
    # stderr is mostly INFO progress lines: keep only warnings/errors/tracebacks + last lines
    keep_err = [l for l in err.splitlines() if any(k in l for k in ("WARNING", "ERROR", "Traceback", "Error", "FAIL"))]
    tail = err.splitlines()[-8:]
    text += ("\n--- stderr (warnings/errors) ---\n" + "\n".join(keep_err[:60]) if keep_err else "")
    text += "\n--- stderr (last lines) ---\n" + "\n".join(tail)
    if len(text) > MAX_CHARS:
        text = text[: MAX_CHARS // 2] + "\n...[truncated]...\n" + text[-MAX_CHARS // 2:]
    out.write(text + f"\n[exit code {r.returncode}]\n")
    out.flush()
    print(f"{title}: exit {r.returncode}")
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="raw dataset directory")
    ap.add_argument("--full", action="store_true", help="process all captures")
    ap.add_argument("--stride", type=int, default=20, help="trial mode: take every K-th capture")
    ap.add_argument("--cap-examples", type=int, default=400, help="max windows kept per capture")
    ap.add_argument("--max-packets", type=int, default=1_000_000, help="max packets read per capture")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-payload", action="store_true", help="skip payload arrays (much smaller/faster)")
    a = ap.parse_args()
    py = sys.executable
    tag = "full" if a.full else f"trial{a.stride}"
    proc, man = f"data/processed/p34_{tag}", f"data/manifests/p34_{tag}.csv"
    rep = ROOT / "artifacts" / "phase34_report.txt"
    rep.parent.mkdir(exist_ok=True)
    with open(rep, "w", encoding="utf-8") as out:
        out.write(f"python {sys.version}\nplatform {platform.platform()}\nroot {a.root}\nmode {tag}\n")
        step("1. unit tests", [py, "-m", "pytest", "-q"], out)
        build = [py, "-m", "src.data.build_sequences", "--config", "configs/data.yaml", "--raw-root", a.root,
                 "--processed-root", proc, "--manifest", man, "--overwrite", "--workers", a.workers,
                 "--max-examples-per-capture", a.cap_examples, "--max-packets-per-capture", a.max_packets]
        if not a.full:
            build += ["--capture-stride", a.stride]
        if a.no_payload:
            build += ["--no-payload"]
        if step("2. build windows", build, out) not in (0, 1):
            out.write("\nBuild failed hard; stopping.\n")
            return
        if not a.no_payload:
            step("3. tensor invariants (loads everything into RAM; skipped in --full)",
                 [py, "-m", "src.data.inspect_processed", "--manifest", man, "--processed-root", proc, "--show", 2]
                 if not a.full else [py, "-c", "print('skipped in --full mode')"], out)
        step("4. label audit", [py, "-m", "src.data.label_audit", "--manifest", man, "--out", f"artifacts/reports/label_audit_{tag}"], out)
        rep_md = ROOT / f"artifacts/reports/label_audit_{tag}.md"
        if rep_md.exists():
            out.write("\n--- label_audit.md ---\n" + rep_md.read_text(encoding="utf-8")[:30000])
        names = {}
        for name, extra in (("group_random", ["--strategy", "group_random"]),
                            ("family_holdout", ["--strategy", "family_holdout"]),
                            ("family_holdout_noinfra", ["--strategy", "family_holdout", "--exclude-local-infra"])):
            full_name = f"{tag}_{name}_s{a.seed}"
            names[name] = full_name
            step(f"5. split {name}", [py, "-m", "src.data.splits", "--manifest", man, "--name", full_name,
                                      "--seed", a.seed, "--allow-degenerate"] + extra, out)
            md = ROOT / "data/splits" / f"{full_name}.md"
            if md.exists():
                out.write("\n--- " + md.name + " ---\n" + md.read_text(encoding="utf-8")[:6000])
        for name in ("group_random", "family_holdout"):
            cmd = [py, "-m", "src.data.consolidate", "--manifest", man, "--processed-root", proc,
                   "--splits", f"data/splits/{names[name]}.csv", "--out", f"data/datasets/{names[name]}"]
            if a.no_payload:
                cmd.append("--no-payload")
            step(f"6. consolidate {name}", cmd, out)
            card = ROOT / "data/datasets" / names[name] / "dataset_card.json"
            if card.exists():
                c = json.loads(card.read_text())
                out.write("\n--- dataset_card (splits + normalizer fit) ---\n"
                          + json.dumps({"splits": c["splits"], "fit": c["normalization_fit"]}, indent=1) + "\n")
        pl = ROOT / "data/manifests/processing_log.jsonl"
        if pl.exists():
            bad = [l for l in pl.read_text(encoding="utf-8").splitlines()
                   if '"status": "ok"' not in l and "shard_exists_manifest_rebuilt" not in l]
            out.write("\n## non-ok processing_log lines (all runs so far)\n" + ("\n".join(bad[-30:]) or "none") + "\n")
    print(f"\nDone. Send me: {rep}")


if __name__ == "__main__":
    main()
