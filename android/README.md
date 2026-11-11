# Android live pipeline (research prototype, detection + logging only)

core/   pure Kotlin port of the Python pipeline: pcap-stream reader, packet parser, flow tracker, window encoder,
        train-fitted normalisation, flow-alert policy, device-alarm rule, StreamEngine.
app/    Android app: root `tcpdump` capture -> core -> TFLite model -> alerts.jsonl; plus an on-phone self-test.

Checks, in order
1. PC: `scripts/export_golden.py` writes runtime_config.txt + golden frames/logits/events from the Python reference.
2. PC: `gradle :core:run --args="../golden"` (or `android/build_check.sh ../golden`) replays the frames through the Kotlin core with the
   Python logits and must print PARITY OK (same windows, tensors, alerts).
3. Phone: app "Self-test" runs the same frames through the TFLite model on the phone and compares logits with PyTorch.
4. Phone: "Start live capture" (needs root + a tcpdump binary at /data/local/tmp/tcpdump).
Nothing is blocked; nothing leaves the phone. Alerts are JSON lines in the app's files dir (alerts.jsonl).
