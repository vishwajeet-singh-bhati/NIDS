## Smoke test (synthetic data; plumbing only)
```
pip install -r requirements.txt
python -m pytest -q                      # expect 136 passed
bash scripts/run_phase2_smoke.sh         # Phase 2 synthetic end-to-end
```
## Real data (CIC-AndMal2017)
```
python scripts/run_phase34.py --root data/raw/CIC-AndMal2017          # -> artifacts/phase34_report.txt
python scripts/run_phase56.py                                          # -> artifacts/phase56_report.txt
```
Rebuild shards with --overwrite if they were built before spec v1.1. Send back both report files.
