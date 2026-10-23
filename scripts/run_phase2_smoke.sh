#!/usr/bin/env bash
# End-to-end smoke test on SYNTHETIC captures: audit -> label dry run -> build -> inspect.
# Verifies plumbing only. Says nothing about detector quality.
set -euo pipefail
cd "$(dirname "$0")/.."
RAW=data/raw/SYNTHETIC
PROC=data/processed/synthetic
MAN=data/manifests/synthetic_manifest.csv
LBL=configs/labels.synthetic.yaml

python scripts/make_synthetic_dataset.py --out "$RAW"
python -m src.data.inspect_dataset --root "$RAW" --out artifacts/reports/synthetic_audit
python -m src.data.build_sequences --config configs/data.yaml --raw-root "$RAW" \
    --processed-root "$PROC" --manifest "$MAN" --label-config "$LBL" --dry-run-labels
python -m src.data.build_sequences --config configs/data.yaml --raw-root "$RAW" \
    --processed-root "$PROC" --manifest "$MAN" --label-config "$LBL" --overwrite
python -m src.data.inspect_processed --manifest "$MAN" --processed-root "$PROC" --show 2
