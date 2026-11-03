#!/usr/bin/env bash
set -e
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "Setup complete. Run: python -m pytest -q"
