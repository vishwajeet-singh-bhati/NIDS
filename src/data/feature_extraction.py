"""Feature specification (v1.1) and window encoding.

This module is the ONE definition of how packets become tensors. The Android encoder must
reproduce `encode_window` exactly; Phase 7/8 will include a parity test against it.

Tensors produced for one example (W = window size, B = payload bytes kept, C = 12):

    meta               float32 [W, C]  channel order = config.META_CHANNELS
    mask               uint8   [W]     1 = real packet, 0 = padding (padding rows are all zero)
    payload            uint8   [W, B]  first B transport-payload bytes (ciphertext for TLS); optional
    payload_valid_len  uint16  [W]     how many bytes of `payload[i]` are real. This is what
                                       distinguishes a real 0x00 byte from padding.
    rel_time           float32 [W]     seconds since the first packet of the window (for detection-
                                       delay analysis only; NOT a model input)

Conventions
-----------
* `payload_len` (meta channel 1) is the RAW declared payload length in bytes. Normalisation
  (log1p + z-score with TRAIN-set statistics) is deliberately NOT applied here; it is fitted in
  Phase 4 so that no test statistics leak.
* TIME / OUT-OF-ORDER TIMESTAMPS (changed in v1.1, found on real CIC-AndMal2017 captures where
  timestamps go backwards inside flows): packets keep CAPTURE-FILE ORDER (never re-sorted), but
  time is measured with a monotone "window clock": clock[0] = 0 and
  clock[i] = max(clock[i-1], ts[i] - ts[0]). A packet whose timestamp is earlier than an already
  seen one therefore does not advance the clock.
      rel_time[i] = clock[i]
      log_iat[i]  = log1p((clock[i] - clock[i-1]) * 1000)      (milliseconds; 0 for i = 0)
  so sum(iat) == rel_time and rel_time is non-decreasing by construction. For captures with
  monotone timestamps this is identical to v1.0. The first packet of every window has
  log_iat = 0, which makes each window self-contained (and every prefix of a window an exact,
  valid shorter window - see `prefix_view`).
* IP addresses and ports are never part of the tensors.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.data.flow_tracker import PacketRecord
from src.data.parse_pcap import TCP_ACK, TCP_FIN, TCP_PSH, TCP_RST, TCP_SYN
from src.utils.config import (META_CHANNELS, N_META, PROTO_DNS, PROTO_OTHER, PROTO_TCP, PROTO_UDP)

_CH = {name: i for i, name in enumerate(META_CHANNELS)}
_PROTO_CH = {PROTO_TCP: _CH["proto_tcp"], PROTO_UDP: _CH["proto_udp"], PROTO_DNS: _CH["proto_dns"],
             PROTO_OTHER: _CH["proto_other"]}


def empty_window(window_size: int, payload_bytes: int, include_payload: bool) -> Dict[str, np.ndarray]:
    out = {
        "meta": np.zeros((window_size, N_META), dtype=np.float32),
        "mask": np.zeros((window_size,), dtype=np.uint8),
        "payload_valid_len": np.zeros((window_size,), dtype=np.uint16),
        "rel_time": np.zeros((window_size,), dtype=np.float32),
    }
    if include_payload:
        out["payload"] = np.zeros((window_size, payload_bytes), dtype=np.uint8)
    return out


def encode_window(packets: Sequence[PacketRecord], window_size: int, payload_bytes: int,
                  include_payload: bool = True, iat_log_unit_ms: bool = True) -> Dict[str, np.ndarray]:
    """Encode up to `window_size` packets. Extra packets are ignored (callers slice windows)."""
    if len(packets) == 0:
        raise ValueError("encode_window needs at least one packet")
    out = empty_window(window_size, payload_bytes, include_payload)
    n = min(len(packets), window_size)
    t0 = packets[0].ts
    clock = 0.0  # monotone window clock, seconds since the first packet of the window
    for i in range(n):
        p = packets[i]
        row = out["meta"][i]
        row[_CH["direction"]] = float(p.direction)
        row[_CH["payload_len"]] = float(p.payload_len)
        new_clock = max(clock, p.ts - t0) if i > 0 else 0.0
        iat = new_clock - clock
        clock = new_clock
        iat_val = iat * 1000.0 if iat_log_unit_ms else iat
        row[_CH["log_iat"]] = math.log1p(iat_val)
        row[_PROTO_CH[p.category]] = 1.0
        if p.tcp_flags:  # non-TCP packets carry 0; DNS-over-TCP keeps its real flags
            f = p.tcp_flags
            row[_CH["flag_syn"]] = 1.0 if f & TCP_SYN else 0.0
            row[_CH["flag_fin"]] = 1.0 if f & TCP_FIN else 0.0
            row[_CH["flag_rst"]] = 1.0 if f & TCP_RST else 0.0
            row[_CH["flag_psh"]] = 1.0 if f & TCP_PSH else 0.0
            row[_CH["flag_ack"]] = 1.0 if f & TCP_ACK else 0.0
        out["mask"][i] = 1
        out["rel_time"][i] = clock
        if include_payload:
            raw = p.payload[:payload_bytes]
            out["payload"][i, :len(raw)] = np.frombuffer(raw, dtype=np.uint8) if raw else 0
            out["payload_valid_len"][i] = len(raw)
        else:
            out["payload_valid_len"][i] = min(len(p.payload), payload_bytes)
    return out


def prefix_view(example: Dict[str, np.ndarray], k: int) -> Dict[str, np.ndarray]:
    """Return the example as the model would see it after only the first `k` packets arrived.

    Because (a) padding is explicit (mask=0, zero rows) and (b) log_iat is relative to the
    previous packet inside the window, the first k packets of a window encode IDENTICALLY
    whether or not later packets exist. So an early prediction is just the same model applied
    to this masked prefix; no re-encoding and no train/serve skew.
    (This also holds with out-of-order timestamps because the window clock only looks backwards.)
    Detection delay for a prefix of length k = `rel_time[k-1]` (seconds into the window) and k
    packets. Phase 6 evaluates prefix lengths such as 4, 8, 16, 32.
    """
    out = {}
    for name, arr in example.items():
        a = arr.copy()
        a[k:] = 0
        out[name] = a
    return out


# ---------------------------------------------------------------------------------------------
# Aggregate features for Model A (MLP baseline). Computed from `meta` + `mask` only, so the
# Android side needs nothing beyond the same per-packet tensor.
# ---------------------------------------------------------------------------------------------
AGG_FEATURE_NAMES: List[str] = [
    "n_packets", "frac_fwd", "n_fwd", "n_rev",
    "sum_payload_fwd", "sum_payload_rev", "mean_payload_fwd", "mean_payload_rev",
    "max_payload", "std_payload", "frac_zero_payload",
    "mean_log_iat", "max_log_iat", "std_log_iat",
    "frac_tcp", "frac_udp", "frac_dns", "frac_other",
    "n_syn", "n_fin", "n_rst", "n_psh",
]


def aggregate_features(meta: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """meta [N,W,C] (or [W,C]), mask [N,W] (or [W]) -> float32 [N, len(AGG_FEATURE_NAMES)]."""
    if meta.ndim == 2:
        meta, mask = meta[None], mask[None]
    m = mask.astype(np.float32)
    n = np.maximum(m.sum(axis=1), 1.0)
    d = meta[:, :, _CH["direction"]]
    pl = meta[:, :, _CH["payload_len"]]
    iat = meta[:, :, _CH["log_iat"]]
    fwd = (d > 0).astype(np.float32) * m
    rev = (d < 0).astype(np.float32) * m
    n_fwd, n_rev = fwd.sum(1), rev.sum(1)

    def masked_mean(x, w):
        return (x * w).sum(1) / np.maximum(w.sum(1), 1.0)

    mean_pl = masked_mean(pl, m)
    std_pl = np.sqrt(np.maximum(masked_mean(pl * pl, m) - mean_pl ** 2, 0.0))
    mean_iat = masked_mean(iat, m)
    std_iat = np.sqrt(np.maximum(masked_mean(iat * iat, m) - mean_iat ** 2, 0.0))

    def frac(ch):
        return (meta[:, :, _CH[ch]] * m).sum(1) / n

    def count(ch):
        return (meta[:, :, _CH[ch]] * m).sum(1)

    cols = [
        m.sum(1), n_fwd / n, n_fwd, n_rev,
        (pl * fwd).sum(1), (pl * rev).sum(1), masked_mean(pl, fwd), masked_mean(pl, rev),
        (pl * m).max(1), std_pl, (((pl == 0) * m).sum(1)) / n,
        mean_iat, (iat * m).max(1), std_iat,
        frac("proto_tcp"), frac("proto_udp"), frac("proto_dns"), frac("proto_other"),
        count("flag_syn"), count("flag_fin"), count("flag_rst"), count("flag_psh"),
    ]
    out = np.stack(cols, axis=1).astype(np.float32)
    assert out.shape[1] == len(AGG_FEATURE_NAMES)
    return out


def count_backward_steps(packets: Sequence[PacketRecord]) -> int:
    """Number of packets whose timestamp is earlier than the running maximum seen so far
    (i.e. out-of-order in the capture file). Used for the manifest column `n_ts_backward`."""
    n, mx = 0, None
    for p in packets:
        if mx is not None and p.ts < mx:
            n += 1
        mx = p.ts if mx is None else max(mx, p.ts)
    return n
