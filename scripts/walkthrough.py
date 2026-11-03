"""Dry run of the WHOLE pipeline on ONE pcap, printing what every stage produces.

    python scripts/walkthrough.py --pcap <file.pcap> --dataset data/datasets/full_group_random_s0 \
        --model runs/full_group_random_s0/final_s0/best.pt [--bundle models/nids_deploy_v1]

Stages shown:  1 raw packets -> 2 flows -> 3 windows -> 4 tensors (model input) -> 5 normalisation (train stats)
               -> 6 model score per window -> 7 early/streaming scores (8/16/32 packets) -> 8 flow alert policy
               -> 9 device alarm rule.
Nothing here is fitted or tuned: it only applies the saved normaliser / checkpoint / thresholds.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.feature_extraction import encode_window, aggregate_features
from src.data.flow_tracker import FlowTracker
from src.data.normalization import Normalizer
from src.data.parse_pcap import ParseStats, iter_packets
from src.models.model_factory import load_checkpoint
from src.stream.device_rule import BurstParams, alarm_times
from src.stream.policy import PolicyParams, first_alert
from src.training.data import tensors_from_arrays
from src.utils.config import META_CHANNELS, FlowConfig, ParseConfig

np.set_printoptions(precision=3, suppress=True, linewidth=160)


def hdr(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap", required=True)
    ap.add_argument("--dataset", required=True, help="consolidated dataset dir (for normalization.json)")
    ap.add_argument("--model", required=True, help="stage-1 checkpoint (best.pt or bundle model.pt)")
    ap.add_argument("--threshold", type=float, default=None, help="flow alert threshold (logit); default from --bundle or 0")
    ap.add_argument("--bundle", default="")
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--show-flows", type=int, default=3)
    a = ap.parse_args()

    cfg_p, cfg_f = ParseConfig(), FlowConfig()
    thr, min_hits, margin, alpha = 0.0, 2, 3.0, 0.5
    dev = {"window_s": 300.0, "min_alerts": 20}
    if a.bundle:
        bc = json.loads((Path(a.bundle) / "config.json").read_text())
        thr, min_hits, margin, alpha = bc["threshold_logit"], bc["min_hits"], bc["hard_margin"], bc["ema_alpha"]
        dev = bc["device_rule"]
    if a.threshold is not None:
        thr = a.threshold

    # ------------------------------------------------------------------ 1. raw packets
    hdr("1. RAW PACKETS  (src/data/parse_pcap.py)  - headers decoded with struct; payload kept = first 128 bytes")
    stats = ParseStats()
    pkts = list(iter_packets(a.pcap, cfg_p, stats))
    print(f"file: {a.pcap}\npackets parsed: {len(pkts)}   parse stats: { {k: v for k, v in stats.as_dict().items() if v} }")
    print("first 5 packets (what survives parsing; IP addresses/ports are used for flow grouping only, never as features):")
    for p in pkts[:5]:
        print(f"  t={p.ts:.3f}  {p.src_ip}:{p.sport} -> {p.dst_ip}:{p.dport}  proto={p.category}  "
              f"payload_len={p.payload_len}  flags={p.tcp_flags:#04x}  payload[:8]={p.payload[:8].hex() or '-'}")

    # ------------------------------------------------------------------ 2. flows
    hdr("2. FLOWS  (src/data/flow_tracker.py)  - canonical bidirectional key, TCP handshake/timeouts, retransmits dropped")
    flows = FlowTracker(cfg_f).run(iter(pkts))
    print(f"{len(flows)} flows. Packet counts per flow (top 8): {sorted([len(f.packets) for f in flows], reverse=True)[:8]}")
    for f in sorted(flows, key=lambda f: -len(f.packets))[:a.show_flows]:
        print(f"  flow#{f.flow_index} proto={f.ip_proto} initiator={f.initiator} responder={f.responder} "
              f"packets={len(f.packets)} retrans_dropped={f.retransmissions} duration={f.last_ts - f.start_ts:.2f}s "
              f"end={f.end_reason} warnings={f.warnings}")

    # ------------------------------------------------------------------ 3. windows + 4. tensors
    flow = max(flows, key=lambda f: len(f.packets))
    hdr(f"3/4. WINDOWS -> MODEL INPUT  (src/data/feature_extraction.py)   example = longest flow (#{flow.flow_index}, {len(flow.packets)} packets)")
    W = a.window
    ex = encode_window(flow.packets[:W], W, 128)
    print(f"window = first {min(W, len(flow.packets))} packets of the flow. Tensors: meta {ex['meta'].shape} float32, mask {ex['mask'].shape}, "
          f"payload {ex['payload'].shape} uint8, payload_valid_len {ex['payload_valid_len'].shape}")
    print("\nmeta channels:", META_CHANNELS)
    print("RAW meta, first 6 packets (rows):")
    print(ex["meta"][:6])
    print("\nmask:", ex["mask"].tolist())
    print("payload row 0 (first 16 of 128 bytes):", ex["payload"][0][:16].tolist(), " valid_len:", int(ex["payload_valid_len"][0]))
    print("aggregate (22 numbers, used only by the MLP/aggregate branch):")
    print(aggregate_features(ex["meta"], ex["mask"])[0])

    # ------------------------------------------------------------------ 5. normalisation
    hdr("5. NORMALISATION  (src/data/normalization.py)  - statistics fitted on the TRAIN split only")
    norm = Normalizer.load(Path(a.dataset) / "normalization.json")
    ch = norm.spec["meta"]["channels"]
    for c in ("payload_len", "log_iat"):
        print(f"  {c}: transform={ch[c]['transform']} train mean={ch[c]['mean']:.4f} std={ch[c]['std']:.4f}")
    m1, k1 = ex["meta"][None], ex["mask"][None]
    t = tensors_from_arrays(m1, k1, ex["payload"][None], ex["payload_valid_len"][None], norm, {"meta", "agg", "payload"})
    print("NORMALISED meta, first 6 packets:")
    print(t["meta"][0][:6].numpy())
    print("payload tokens (byte+1, 0=pad) row 0 first 16:", t["payload"][0, 0, :16].tolist())

    # ------------------------------------------------------------------ 6/7. model + early scores
    hdr("6/7. MODEL SCORE  (src/models/hybrid_resnet.py)  - one logit per window; early decisions on the first 8/16/32 packets")
    model, ck = load_checkpoint(a.model, "cpu")
    model.eval()
    print(f"model: {ck['model_name']}, parameters: {sum(p.numel() for p in model.parameters()):,}, inputs: {sorted(model.inputs)}")
    kinds = set(model.inputs)
    logits_full = []
    for k in (8, 16, 32):
        if len(flow.packets) < min(k, 2):
            continue
        e = encode_window(flow.packets[:k], W, 128)
        tt = tensors_from_arrays(e["meta"][None], e["mask"][None], e["payload"][None], e["payload_valid_len"][None], norm, kinds)
        with torch.no_grad():
            z = float(model(tt))
        print(f"  after {min(k, len(flow.packets)):2d} packets (t={e['rel_time'][min(k, len(flow.packets)) - 1]:.3f}s): logit={z:+.3f}  "
              f"prob={1 / (1 + np.exp(-z)):.3f}  -> {'ABOVE' if z >= thr else 'below'} threshold {thr:.3f}")

    # all windows of this flow
    hdr("8. FLOW ALERT POLICY  (src/stream/policy.py)  - EMA of window logits + min_hits + hard margin")
    zs = []
    for s0 in range(0, len(flow.packets), W):
        chunk = flow.packets[s0:s0 + W]
        if len(chunk) < 2:
            break
        e = encode_window(chunk, W, 128)
        tt = tensors_from_arrays(e["meta"][None], e["mask"][None], e["payload"][None], e["payload_valid_len"][None], norm, kinds)
        with torch.no_grad():
            zs.append(float(model(tt)))
    pp = PolicyParams(threshold=thr, ema_alpha=alpha, min_hits=min_hits, hard_margin=margin)
    idx, why = first_alert(zs, pp)
    print(f"window logits of this flow: {np.round(zs, 2).tolist()}\nthreshold={thr:.3f} (chosen on VALIDATION), ema_alpha={alpha}, min_hits={min_hits}, hard_margin={margin}")
    print("flow alert:", "none" if idx is None else f"at window {idx} ({why})")

    # all flows -> device rule
    hdr("9. DEVICE ALARM  (src/stream/device_rule.py)  - >= N flow alerts within T seconds")
    alert_t = []
    nsc = 0
    for f in flows:
        if len(f.packets) < 2:
            continue
        zz = []
        for s0 in range(0, len(f.packets), W):
            chunk = f.packets[s0:s0 + W]
            if len(chunk) < 2:
                break
            e = encode_window(chunk, W, 128)
            tt = tensors_from_arrays(e["meta"][None], e["mask"][None], e["payload"][None], e["payload_valid_len"][None], norm, kinds)
            with torch.no_grad():
                zz.append(float(model(tt)))
        nsc += len(zz)
        i, _ = first_alert(zz, pp)
        if i is not None:
            alert_t.append(f.packets[min(i * W + W, len(f.packets)) - 1].ts - pkts[0].ts)
    al = alarm_times(alert_t, BurstParams(window_s=float(dev["window_s"]), min_alerts=int(dev["min_alerts"])))
    print(f"{nsc} windows scored over {len(flows)} flows -> {len(alert_t)} flow alerts; device rule: >= {dev['min_alerts']} alerts in {dev['window_s']} s")
    print("device alarms at t(s) =", [round(x, 1) for x in al] or "none")


if __name__ == "__main__":
    main()
