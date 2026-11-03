# NIDS - on-device network intrusion detection (research prototype)

One model, one pipeline, one deployable folder.

**Model:** `hybrid_resnet` (payload + packet metadata + flow aggregates, INT8 quantisation-aware training, prefix-augmented).
**Live detector:** packets -> flows -> decision after 8/16/32 packets -> flow alert -> *device alarm* when >= N flow alerts happen within T seconds.
Detection and logging only; nothing is blocked and nothing leaves the machine.

## Quick start (Linux / WSL)

```bash
python -m venv nids && source nids/bin/activate
pip install -r requirements.txt
python -m pytest -q                                                       # 159 passed (synthetic data)

# 1. data: PCAP -> windows -> leakage-safe splits -> normalised datasets  (hours; add --workers N)
python scripts/run_phase34.py --root data/raw/CIC-AndMal2017 --full --workers 4

# 2. train + choose thresholds on val + test by replaying unseen captures + export the bundle
python scripts/run_final.py --raw-root data/raw/CIC-AndMal2017 \
       --dataset data/datasets/full_family_holdout_s0 --seeds 0 1 2       # -> models/nids/nids_v1/ and artifacts/final_report.txt

# 3. run the detector (bundle = model + thresholds + device rule)
python -m src.detection.nids.stream.run --bundle models/nids/nids_v1 --pcap some.pcap --alerts alerts.jsonl        # replay
sudo python -m src.detection.nids.stream.run --bundle models/nids/nids_v1 --iface eth0 --alerts alerts.jsonl       # live (Linux / rooted Android)
```

Output of step 2: `models/nids/nids_v1/{model.pt, config.json, MODEL_CARD.md}`. The model card carries the held-out metrics
(accuracy / precision / recall / F1 at flow and device level, false alarms per benign hour, time to detect) and the limits.

Deployment model: run step 2 once on a `*_family_holdout_*` dataset (honest numbers on unseen families), then again on a
`*_group_random_*` dataset so every family is in training; quote the first set of numbers, ship the second model.

## Layout

```
src/detection/nids/data/      PCAP parsing, flow tracking, window features (spec v1.1), labels, splits, normaliser, consolidation
src/detection/nids/models/    hybrid_resnet (the model) + research baselines: mlp, cnn_meta, cnn_payload, packet_bert(_pay)
src/detection/nids/training/  train (prefix-aug, QAT), pretrain (label-free, optional), evaluate, metrics
src/detection/nids/stream/    engine (live detector), policy + device_rule (alert logic), replay_eval, bundle, run (CLI)
scripts/       run_phase34 (data), run_final (the model), run_realtime (model comparison), helpers
docs/PIPELINE_DETAILS.md   feature spec, manifest schema, data-processing details, troubleshooting
```

## What the real runs showed (CIC-AndMal2017, held-out families, weak labels)

* Payload-reading models reached val ROC-AUC ~0.75-0.78; metadata-only ~0.63-0.67.
* PacketBERT with payload (pretrained) tied the hybrid ResNet, so the simpler, easier-to-quantise hybrid is the single model.
* Single flow alert: precision ~0.97, recall ~0.29 at 1 % FPR. Device alarm: accuracy 0.97-0.98, precision 0.88-0.95,
  recall 0.94-1.00, benign-capture false alarms 1-4 %, 0.06-0.3 alarms per benign hour (~105 captures, 23 malware, one seed).

## Limits

Labels are per capture (benign background inside malware captures counts as malicious); benign and malware captures were
recorded on different dates (possible shortcut); payload is mostly ciphertext; small device-level test set; one seed so far;
live capture on a real interface and INT8 export (TFLite/LiteRT) to Android are not done; not a production-grade detector.
