# android-nids — on-device network intrusion detection (research prototype)

Status: **Phases 1–6 implemented, plus a real-time streaming pipeline (section 3a)** (repo, dataset audit, PCAP→flows→windows, labels, leakage-safe splits,
normalisation, models, training, evaluation). **Phases 7–8 (INT8 quantisation, Android app) are not started.**
Everything was validated on *synthetic* captures only (157 tests). **No real-data detection result is
claimed anywhere in this repo**; results appear only after you run it on CIC-AndMal2017.

The first classifier is a **malware-associated traffic classifier trained on weak capture-level labels**.
It is not a flow-level intrusion detector and not a C&C detector. Detection and logging only — nothing is
blocked, and no traffic leaves the device.

## 1. Layout

```
configs/   data.yaml (spec v1.1)  labels.yaml (real AndMal rules)  labels.synthetic.yaml
           model.yaml  train.yaml  experiments.yaml
src/data/      inspect_dataset  parse_pcap  flow_tracker  feature_extraction  labeling  build_sequences
               inspect_processed  dataset_loader  splits  normalization  consolidate  label_audit  flow_tags
src/models/    mlp_baseline (A)  cnn_1d (B meta, C payload+meta)  hybrid_resnet (D, QAT)  packet_bert (E, QAT)  model_factory
src/stream/    engine (live detector)  policy (flow alert rule)  replay_eval (flow/capture metrics)  run (CLI)
src/training/  data  losses  train  pretrain  evaluate  report
src/utils/     config  logging_utils  reproducibility  synthetic  metrics
scripts/       run_phase34.py  run_phase56.py  run_realtime.py  make_demo_dataset.py  aggregate_results.py  diagnose_timestamps.py
               make_synthetic_dataset.py  run_phase2_smoke.sh  run_and_collect.py
tests/         157 tests (pytest)
```

## 2. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# optional, smaller CPU-only torch:  pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pytest -q          # all tests pass (~2 min)
```

## 3. Run order on real data (from repo root)

```bash
# Phases 3-4: PCAP -> windows -> splits -> normalisation -> consolidated datasets (writes artifacts/phase34_report.txt)
python scripts/run_phase34.py --root data/raw/CIC-AndMal2017            # trial subset first
python scripts/run_phase34.py --root data/raw/CIC-AndMal2017 --full --workers 4

# Phases 5-6: train + evaluate on every consolidated dataset (writes artifacts/phase56_report.txt)
python scripts/run_phase56.py                                            # models mlp + cnn_meta
python scripts/run_phase56.py --with-payload --seeds 0 1 2 --epochs 15   # add Model C, 3 seeds
python scripts/aggregate_results.py                                      # table across runs
```

**Important:** if you built shards with an earlier version, rebuild with `--overwrite` (the v1.1 window clock
fixes the `FAIL I4: rel_time decreases` error). Send back `artifacts/phase34_report.txt` and `artifacts/phase56_report.txt`.

Single runs:

```bash
python -m src.training.train --dataset data/datasets/<name> --model {mlp,cnn_meta,cnn_payload} [--seed 0 --epochs 15 --sampler balanced]
python -m src.training.evaluate --run-dir runs/<name>/cnn_meta_s0 [--eval-dataset data/datasets/<other>]
```

## 3a. Real-time pipeline (streaming detection, pretrained encoder, QAT)

Why: fixed 32-packet windows are only a *context cap*. A deployed detector must not wait for them.

| Piece | What it does |
|---|---|
| `src/stream/engine.py` | Packets in → same `FlowTracker` as offline → at 8/16/32 packets of a window the first *k* packets are encoded exactly like training and scored → per-flow EMA + hysteresis → JSONL alert. Alerts can fire after the 8th packet. Also scores the last partial window when a flow ends. Micro-batched inference, memory guard, alert cooldown. Detection/logging only. |
| `--prefix-aug 0.5` (train) | 50 % of training windows are shown as random packet prefixes (4/8/16/32), so one model works at every checkpoint. Model selection pools val windows at all prefix lengths. |
| `src/models/packet_bert.py` | Compact BERT-style encoder (≈0.25M params, ReLU FFN, all Linear/Conv are INT8 fake-quant layers). Not `bert-base`: no text/traffic checkpoint transfers to this feature spec and 110M params is not an on-device model. |
| `src/training/pretrain.py` | Label-free masked-packet-modelling on the TRAIN split only; reports reconstruction vs a zero baseline. |
| QAT | `qat_start_epoch` switches on per-channel INT8 weight / per-tensor INT8 activation fake-quant with STE; metrics are measured with quantisation active. |
| `src/stream/device_rule.py` | Device alarm: ≥N flow alerts within T seconds → one `device_alarm` event (the natural “flag this phone” decision). (T, N) is chosen on VAL captures (max recall at benign-capture false-alarm ≤ 5 %) and applied unchanged to TEST. |
| `src/stream/replay_eval.py` | Pushes held-out PCAPs through the engine; threshold chosen on VAL captures (max recall at FPR ≤ 1 %), applied unchanged to TEST. Reports accuracy / precision / recall / F1 / FPR at **flow** and **capture** level, alerts per benign hour, time-to-detect (s, packets), throughput, per-family recall, and results with local-infrastructure flows removed. |

Run (after `scripts/run_phase34.py` created `data/datasets/*`):

```bash
python scripts/run_realtime.py --raw-root data/raw/CIC-AndMal2017                   # hybrid_resnet + PacketBERT(payload), -> artifacts/realtime_report.txt
python scripts/run_realtime.py --raw-root data/raw/CIC-AndMal2017 --ablations        # + from-scratch and metadata-only PacketBERT
python scripts/run_realtime.py --raw-root data/raw/CIC-AndMal2017 --quick           # tiny budget, checks it runs

# individual steps
python -m src.training.pretrain --dataset data/datasets/<ds> --out runs/<ds>/pretrain_s0
python -m src.training.train --dataset data/datasets/<ds> --model packet_bert --init-from runs/<ds>/pretrain_s0/encoder.pt --prefix-aug 0.5 --run-dir runs/<ds>/pb_pre_s0
python -m src.stream.replay_eval --run-dir runs/<ds>/pb_pre_s0 --manifest data/manifests/p34_<tag>.csv --splits data/splits/<ds>.csv --raw-root data/raw/CIC-AndMal2017
python -m src.stream.run --run-dir runs/<ds>/pb_pre_s0 --pcap some.pcap --alerts alerts.jsonl       # replay
sudo python -m src.stream.run --run-dir runs/<ds>/pb_pre_s0 --iface en0 --alerts alerts.jsonl        # live capture
```

No re-processing is needed: spec v1.1 shards/datasets built earlier are used as they are.

First real-data findings (CIC-AndMal2017, held-out families): payload-reading models (hybrid ResNet, payload CNN) reach val ROC-AUC ≈ 0.75–0.78, metadata-only models ≈ 0.63–0.67. At the flow level the hybrid ResNet gives precision ≈ 0.98 / recall ≈ 0.27–0.30 at 1 % FPR; aggregating alerts per device (≥20 within 30 s) gave accuracy 0.97, precision 0.92, recall 0.96 on 107 test captures (23 malware, 6 families, weak labels) – small sample, not a production claim.

What is NOT done yet: real INT8 export (TFLite/LiteRT/ExecuTorch) and the Android service; softmax/LayerNorm/residual adds are not fake-quantised, so an exported INT8 model must be re-validated; live capture was not exercised on a real interface in the test environment (replay and raw-frame decoding are tested); no real-data results exist yet.

## 4. What Phases 5–6 do

* **Models:** A `mlp` (22 aggregate features), B `cnn_meta` (12 metadata channels + mask), C `cnn_payload`
  (adds a byte-level payload branch; an experiment, payloads are mostly ciphertext). Padding is masked
  throughout; a window prefix is a valid input (early prediction).
* **Training:** train-only normaliser; imbalance handled with `pos_weight` or an in-loop balanced sampler
  (no data duplication); best epoch chosen on **val only**; test touched once.
* **Evaluation:** thresholds from val only (FPR ≤ 1 %, ≤ 5 %, max-F1, logit ≥ 0) applied unchanged to test;
  ROC/PR-AUC; confusion matrices; per-family recall, per-benign-subset FPR, per-flow-tag results, AUC with
  local-infrastructure windows removed; capture-level alert rates; ECE before/after val-fitted temperature;
  detection delay (packets and seconds) for k = 4/8/16/32 packets.
* **Experiment plan:** `configs/experiments.yaml` (random group split, unseen-family holdout, holdout without
  local infrastructure, window 16/64). Rows without a finished run show as *pending*.

## 5. Findings on the real data so far (from your Phase 3–4 report)

* ~35 % of windows are local-infrastructure background traffic (flow tag `local_infra`).
* Benign and malware captures were recorded on largely non-overlapping dates → a **temporal confound**; a model
  may learn capture-time artifacts. Treat family-holdout and `--exclude-local-infra` results as the more honest ones.

## 6. Limitations

1. Labels are weak and capture-level; `malicious_associated` ≠ confirmed attack or C&C.
2. All code paths were validated on synthetic data; real-data behaviour and performance are unknown until you run it.
3. Memory for the largest captures is bounded only via `--max-packets-per-capture` / `--max-examples-per-capture`.
4. Pure-Python parsing speed on multi-GB data is unmeasured; use `--workers`.
5. Flow timeouts and window policy are my choices, not dataset-derived.
6. No production-grade detection claim. Quantisation (INT8) and Android integration are Phases 7–8.

## 7. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `FAIL I4: rel_time decreases` | Shards built with spec v1.0; rebuild with `--overwrite` |
| `raw_root does not exist` | Set `--root` / `--raw-root` |
| All examples `unknown` | No label rule matched; use `--dry-run-labels` |
| `train/val must each contain both classes` | Split too small/skewed; use a bigger subset or another seed |
| `ModuleNotFoundError: src` | Run from the repo root with `python -m ...` |
| torch download fails | Use the CPU wheel index shown in section 2 |

---
# Appendix: feature specification and manifest (Phase 2)

## 5. Feature specification v1.1 (frozen for Phases 5-8)

Per example, window size `W` (16/32/64; default 32), payload budget `B` (default 128):

| array | shape / dtype | meaning |
|---|---|---|
| `meta` | `[W,12]` float32 | channels: `direction, payload_len, log_iat, proto_tcp, proto_udp, proto_dns, proto_other, flag_syn, flag_fin, flag_rst, flag_psh, flag_ack` |
| `mask` | `[W]` uint8 | 1 real packet, 0 padding (padding rows are all zero) |
| `payload` | `[W,B]` uint8 | first B transport-payload bytes; **omitted entirely** with `--no-payload` |
| `payload_valid_len` | `[W]` uint16 | real bytes in `payload[i]`; separates real `0x00` from padding |
| `rel_time` | `[W]` float32 | window clock in seconds (monotone); for delay analysis, not a model input |

* `direction`: +1 initiator→responder, −1 responder→initiator.
* `payload_len`: raw declared transport payload bytes (from IP/transport length fields, so snaplen-truncated
  captures still report the true size and Ethernet padding is never counted). Normalisation is **not** applied
  here; it is fitted on training data only (Phase 4).
* Window clock (v1.1): packets keep capture-file order; `clock[i] = max(clock[i-1], ts[i]-ts[0])`; `log_iat = log1p(Δclock in ms)`; first packet is 0. Out-of-order timestamps therefore never produce negative gaps (v1.0 broke invariant I4 on real data; **rebuild shards with `--overwrite`**).
* `proto_dns`: TCP/UDP with a port in `dns_ports` (53, 5353). DoT/DoH/mDNS-on-other-ports are not detected.
* IPs and ports are saved in the manifest for audit only and are never tensors.
* Model A aggregate features (`AGG_FEATURE_NAMES`, 22 values) are derived from `meta`+`mask` only.

### Early prediction and detection delay
Because padding is explicit and `log_iat` is window-relative, the first `k` packets of a window encode
identically whether or not later packets exist (`prefix_view`, tested). An early prediction after `k`
packets is the same model applied to that masked prefix; the delay is `rel_time[k-1]` seconds and `k`
packets. `src.training.evaluate` reports `k ∈ {4, 8, 16, 32}` with packets/seconds needed. Whether short prefixes are accurate is an empirical question; nothing is claimed.

### How awkward traffic is handled

| Situation | Behaviour |
|---|---|
| IP fragments | First fragment parsed and flagged; non-first fragments counted and **skipped**; no reassembly |
| TCP segmentation/offload | Packets taken as captured; IP total length 0 → captured length used, warned |
| Retransmissions | Detected per direction (32-bit modular seq compare); dropped from sequences by default, counted |
| Snaplen truncation | True `payload_len` kept, fewer payload bytes, warning `snaplen_truncated_payload` |
| 5-tuple reuse | New pure SYN with different ISN, or traffic after close + time-wait, starts a new flow |
| Idle / very long flows | Idle timeout starts a new flow; max duration splits and keeps original initiator |
| Mid-stream capture start | Initiator = first sender (SYN+ACK-first → receiver); flagged `midstream_start` |
| Bad / non-monotonic timestamps | Counted; packets are not re-sorted; the window clock is monotone (see above) |
| Truncated / corrupt file | Packets read so far are kept; `file_truncated_or_corrupt` in warnings |
| Unsupported link type / non-IP | Counted, never silently decoded |
| TLS/QUIC | Not decrypted; payload bytes are ciphertext |

## 6. Manifest schema (`data/manifests/*.csv`, one row per example)

`example_id, source_dataset, capture_id, capture_path, app_id, malware_family, malware_category,
original_label, label, binary_label, label_source, label_reliability, label_rule, flow_id, window_index,
window_start_packet, n_packets, flow_packets_kept, flow_total_packets_seen, window_first_ts,
window_last_ts, window_duration_s, initiator_ip, initiator_port, responder_ip, responder_port, ip_proto,
flow_end_reason, shard_file, shard_row, warnings, feature_spec_version`

* `label` ∈ {`benign`, `malicious_associated`, `confirmed_attack`, `unknown`}; `original_label` is preserved verbatim.
* `binary_label`: 0 benign, 1 malicious_associated/confirmed_attack, **−1 = excluded from supervised work**.
* `label_reliability`/`label_rule` record how the label was obtained. Nothing emits `confirmed_attack`
  unless you write a rule or override for it.
* `warnings` is `;`-separated and combines capture-level and flow-level conditions.
* Grouping keys for Phase 4 splits: `capture_id`, `flow_id`, `app_id`, `malware_family`.

## 6. Manifest schema (`data/manifests/*.csv`, one row per example)

`example_id, source_dataset, capture_id, capture_path, app_id, malware_family, malware_category,
original_label, label, binary_label, label_source, label_reliability, label_rule, flow_id, window_index,
window_start_packet, n_packets, flow_packets_kept, flow_total_packets_seen, window_first_ts,
window_last_ts, window_duration_s, initiator_ip, initiator_port, responder_ip, responder_port, ip_proto,
flow_end_reason, shard_file, shard_row, warnings, feature_spec_version`

* `label` ∈ {`benign`, `malicious_associated`, `confirmed_attack`, `unknown`}; `original_label` is preserved verbatim.
* `binary_label`: 0 benign, 1 malicious_associated/confirmed_attack, **−1 = excluded from supervised work**.
* `label_reliability`/`label_rule` record how the label was obtained. Nothing emits `confirmed_attack`
  unless you write a rule or override for it.
* `warnings` is `;`-separated and combines capture-level and flow-level conditions.
* Grouping keys for Phase 4 splits: `capture_id`, `flow_id`, `app_id`, `malware_family`.

