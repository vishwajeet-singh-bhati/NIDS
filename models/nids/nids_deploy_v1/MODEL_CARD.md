# Model card - hybrid_resnet

Created 2026-11-02 06:23:36.

**Task:** flag traffic that looks like malware-associated Android traffic (CIC-AndMal2017 weak labels). Detection and logging only.

## Operating point (chosen on validation, applied unchanged to test)

- flow alert threshold: logit 1.514 (early decisions after [8, 16, 32] packets)
- device alarm: >= 20 flow alerts within 300 s

## Held-out test results (replay of unseen captures through the live engine)

| level | accuracy | precision | recall | F1 | false-alarm |
|---|---|---|---|---|---|
| single flow | 0.615 | 0.977 | 0.318 | 0.480 | FPR 0.009 |
| device (capture) | 0.975 | 0.912 | 0.969 | 0.939 | 0.024 of benign captures; 0.11 alarms / benign hour |

Device-level test set: 318 captures (tp 62, fp 6, fn 2, tn 248).

Time from the first flow of a malware capture to its first device alarm: median 56.6 s, p90 1316.3 s (62 detected malware captures). The alarm needs a burst of flow alerts, so it is much slower than a single flow alert.

Time to detect (true-positive flows): median 0.417 s / 8 packets; p90 12.658 s / 16 packets.

## Limits - read before trusting it

- Labels are per capture: benign background traffic inside malware captures counts as malicious, so flow-level recall is capped.
- Benign and malware captures were recorded on largely different dates; the model may partly learn capture-time artefacts.
- Device-level numbers come from the capture counts above (a handful of malware families); seeds share the same split, so spread over seeds reflects training noise only, not split-to-split variation.
- Precision at device level depends on how common malware is: the test set is roughly one-fifth malware; with rarer malware, precision falls.
- Payload bytes are mostly ciphertext; the signal may come from handshake fields that identify apps rather than maliciousness.
- Not a production-grade detector. Not validated on other datasets or on live device traffic.
