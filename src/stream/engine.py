"""Real-time detector: packets in -> per-flow incremental windows -> model -> smoothed flow score -> alerts.

Why this design (instead of waiting for full 32-packet windows):
  * Decisions are made while a flow is still growing: after `checkpoints` packets of a window (default
    8, 16, 32) the first k packets are encoded EXACTLY as in training (`encode_window`, spec v1.1) and
    scored. A flow can raise an alert after its 8th packet. The models are trained on such prefixes
    (`--prefix-aug`), so one model serves every checkpoint.
  * The flow tracker is the same one used offline (`FlowTracker`), so flow splitting, retransmission
    handling and packet caps are identical to training.
  * Each flow keeps an exponential moving average of its window logits; a flow is flagged only after
    `min_hits` scored windows above the threshold (hysteresis), and alerts are de-duplicated.
  * Window tensors are produced by `tensors_from_arrays`, the same function offline evaluation uses.

Detection and logging only: nothing is blocked, nothing leaves the process. Alerts are JSON lines.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, TextIO, Tuple

import numpy as np
import torch

from src.data.feature_extraction import encode_window
from src.data.flow_tracker import Flow, FlowTracker
from src.data.normalization import Normalizer
from src.data.parse_pcap import ParsedPacket, ParseStats, decode_packet, iter_packets
from src.models.model_factory import load_checkpoint
from src.training.data import tensors_from_arrays
from src.utils.config import FlowConfig, ParseConfig, PipelineConfig, load_config
from src.stream.policy import PolicyParams, PolicyState, step
from src.utils.metrics import sigmoid

log = logging.getLogger("stream")


@dataclass
class StreamConfig:
    checkpoints: Tuple[int, ...] = (8, 16, 32)   # packets into a window at which the flow is scored
    final_min_packets: int = 4                   # when a flow ends, score its last partial window if >= this many packets
    threshold: float = 0.0                       # logit; choose it on validation (src.stream.replay_eval)
    ema_alpha: float = 0.5                       # weight of the newest window logit in the flow score
    min_hits: int = 2                            # consecutive scored windows above threshold before alerting
    hard_margin: float = 3.0                     # a single window this far above threshold alerts immediately
    max_scored_windows: int = 20                 # same cap as training (sequence.max_windows_per_flow)
    batch_size: int = 128                        # micro-batch size for the model
    max_active_flows: int = 50_000               # memory guard; oldest flows are evicted (and finalised) beyond it
    cooldown_s: float = 60.0                     # suppress repeat alerts for the same (initiator ip, responder ip:port, proto)
    drain_every: int = 2000                      # packets between housekeeping passes


@dataclass
class FlowState:
    pol: PolicyState = field(default_factory=PolicyState)
    scored_at: set = field(default_factory=set)       # (start, k) pairs already scored (no double scoring)


@dataclass
class _Pending:
    flow: Flow
    start: int
    k: int
    window: Dict[str, np.ndarray]
    ts: float
    final: bool = False


class StreamDetector:
    def __init__(self, checkpoint: str, cfg: Optional[PipelineConfig] = None, stream_cfg: Optional[StreamConfig] = None,
                 device: str = "cpu", alert_sink: Optional[TextIO] = None, config_path: str = "configs/data.yaml"):
        self.model, self.ck = load_checkpoint(checkpoint, device)
        self.device = device
        self.norm = Normalizer(self.ck["normalization"])
        spec = self.ck["feature_spec"]
        self.W = int(spec["window_size"])
        self.payload_bytes = int(spec.get("payload_bytes") or spec.get("payload_bytes_kept") or 0)
        self.kinds = set(self.model.inputs)
        self.include_payload = "payload" in self.kinds
        if self.include_payload and not self.payload_bytes:
            raise ValueError("model needs payload bytes but the checkpoint's feature spec has no payload size")
        self.cfg = cfg or load_config(config_path)
        if self.cfg.sequence.window_size != self.W:
            self.cfg.sequence.window_size = self.W
        if self.payload_bytes:
            self.cfg.parse.payload_bytes_kept = self.payload_bytes
        self.cfg.sequence.stride = int(spec.get("stride", self.cfg.sequence.stride))
        self.S = self.cfg.sequence.stride
        self.sc = stream_cfg or StreamConfig()
        self.sc.checkpoints = tuple(sorted({min(int(k), self.W) for k in self.sc.checkpoints} | {self.W}))
        self.tracker = FlowTracker(self.cfg.flow)
        self.parse_stats = ParseStats()
        self.state: Dict[int, FlowState] = {}
        self.queue: List[_Pending] = []
        self.sink = alert_sink
        self.alerts: List[Dict[str, Any]] = []
        self.scores: List[Dict[str, Any]] = []        # optional score trace (replay evaluation)
        self.record_scores = False
        self._last_alert: Dict[tuple, float] = {}
        self._since_drain = 0
        self.stats = {"packets": 0, "packets_kept": 0, "flows_seen": 0, "inferences": 0, "alerts": 0,
                      "alerts_suppressed_cooldown": 0, "inference_seconds": 0.0, "batches": 0}
        self.now = 0.0

    # ------------------------------------------------------------------ input
    def feed_raw(self, buf: bytes, linktype: int, ts: float, wirelen: Optional[int] = None) -> List[Dict[str, Any]]:
        pkt = decode_packet(buf, linktype, ts, wirelen or len(buf), self.cfg.parse, self.parse_stats)
        return self.feed(pkt) if pkt is not None else []

    def feed(self, pkt: ParsedPacket) -> List[Dict[str, Any]]:
        """Process one parsed packet. Returns alerts emitted by this call (usually empty)."""
        self.stats["packets"] += 1
        self.now = max(self.now, pkt.ts)
        flow = self.tracker.process(pkt)
        if flow is not None:
            self.stats["packets_kept"] += 1
            self._after_append(flow, pkt.ts)
        self._since_drain += 1
        out: List[Dict[str, Any]] = []
        if len(self.queue) >= self.sc.batch_size:
            out += self.run_inference()
        if self._since_drain >= self.sc.drain_every:
            out += self._housekeeping()
        return out

    def feed_pcap(self, path: str | Path, finish: bool = True) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for pkt in iter_packets(path, self.cfg.parse, self.parse_stats):
            out += self.feed(pkt)
        return out + (self.finish() if finish else self.run_inference())

    # ------------------------------------------------------------------ windows
    def _after_append(self, flow: Flow, ts: float) -> None:
        st = self.state.get(flow.flow_index)
        if st is None:
            st = self.state[flow.flow_index] = FlowState()
            self.stats["flows_seen"] += 1
        n = len(flow.packets)
        for k in self.sc.checkpoints:
            s = n - k
            if s < 0 or s % self.S:
                continue
            if s // self.S >= self.sc.max_scored_windows:
                continue
            self._enqueue(flow, st, s, k, ts, final=False)

    def _enqueue(self, flow: Flow, st: FlowState, s: int, k: int, ts: float, final: bool) -> None:
        if (s, k) in st.scored_at:
            return
        st.scored_at.add((s, k))
        win = encode_window(flow.packets[s:s + k], self.W, self.payload_bytes, self.include_payload,
                            self.cfg.sequence.iat_log_unit_ms)
        self.queue.append(_Pending(flow, s, k, win, ts, final))

    def _finalise(self, flow: Flow) -> None:
        """Flow ended: score the last partial window if it has not been scored at that exact length."""
        st = self.state.get(flow.flow_index)
        if st is None:
            return
        n = len(flow.packets)
        if n >= 2:
            s = ((n - 1) // self.S) * self.S
            k = n - s
            if k >= self.sc.final_min_packets and s // self.S < self.sc.max_scored_windows:
                self._enqueue(flow, st, s, k, flow.last_ts, final=True)

    # ------------------------------------------------------------------ inference + policy
    def run_inference(self) -> List[Dict[str, Any]]:
        if not self.queue:
            return []
        q, self.queue = self.queue, []
        out: List[Dict[str, Any]] = []
        for i in range(0, len(q), self.sc.batch_size):
            chunk = q[i:i + self.sc.batch_size]
            meta = np.stack([p.window["meta"] for p in chunk])
            mask = np.stack([p.window["mask"] for p in chunk])
            pay = np.stack([p.window["payload"] for p in chunk]) if self.include_payload else None
            pvl = np.stack([p.window["payload_valid_len"] for p in chunk]) if self.include_payload else None
            batch = tensors_from_arrays(meta, mask, pay, pvl, self.norm, self.kinds)
            t0 = time.perf_counter()
            with torch.no_grad():
                logits = self.model({k: v.to(self.device) for k, v in batch.items()}).cpu().numpy().astype(np.float64)
            self.stats["inference_seconds"] += time.perf_counter() - t0
            self.stats["inferences"] += len(chunk)
            self.stats["batches"] += 1
            for p, z in zip(chunk, logits):
                out += self._apply_policy(p, float(z))
        return out

    def _apply_policy(self, p: _Pending, z: float) -> List[Dict[str, Any]]:
        st = self.state.get(p.flow.flow_index)
        if st is None:
            return []
        sc = self.sc
        reason = step(st.pol, z, PolicyParams(sc.threshold, sc.ema_alpha, sc.min_hits, sc.hard_margin))
        if self.record_scores:
            self.scores.append({"flow_index": p.flow.flow_index, "start": p.start, "k": p.k, "ts": p.ts,
                                "logit": z, "n_packets": len(p.flow.packets), "final": p.final,
                                "initiator_ip": p.flow.initiator[0], "responder_ip": p.flow.responder[0],
                                "responder_port": p.flow.responder[1], "ip_proto": p.flow.ip_proto,
                                "flow_start_ts": p.flow.start_ts})
        return self._emit(p, st, z, reason) if reason else []

    def _emit(self, p: _Pending, st: FlowState, z: float, reason: str) -> List[Dict[str, Any]]:
        f = p.flow
        dedupe = (f.initiator[0], f.responder[0], f.responder[1], f.ip_proto)
        last = self._last_alert.get(dedupe)
        if last is not None and p.ts - last < self.sc.cooldown_s:
            self.stats["alerts_suppressed_cooldown"] += 1
            return []
        self._last_alert[dedupe] = p.ts
        ev = {"ts": p.ts, "flow_index": f.flow_index, "initiator": f"{f.initiator[0]}:{f.initiator[1]}",
              "responder": f"{f.responder[0]}:{f.responder[1]}", "ip_proto": f.ip_proto, "reason": reason,
              "window_logit": round(z, 4), "flow_score": round(st.pol.ema, 4),
              "window_prob": round(float(sigmoid(np.array([z]))[0]), 4), "scored_windows": st.pol.scored,
              "packets_seen": len(f.packets), "flow_start_ts": f.start_ts,
              "seconds_since_flow_start": round(p.ts - f.start_ts, 4), "decided_on_prefix_k": p.k,
              "model": self.ck["model_name"],
              "note": "malicious-associated traffic pattern (weak-label model); detection/log only, nothing is blocked"}
        self.stats["alerts"] += 1
        self.alerts.append(ev)
        if self.sink is not None:
            self.sink.write(json.dumps(ev) + "\n")
            self.sink.flush()
        return [ev]

    # ------------------------------------------------------------------ lifecycle
    def _housekeeping(self) -> List[Dict[str, Any]]:
        self._since_drain = 0
        if self.tracker._active and len(self.tracker._active) > self.sc.max_active_flows:  # noqa: SLF001
            self.tracker.evict_oldest(self.sc.max_active_flows)
        done = self.tracker.drain_finished()
        for f in done:
            self._finalise(f)
        out = self.run_inference()
        self._forget_finished(done)
        return out

    def _forget_finished(self, flows: Iterable[Flow]) -> None:
        for f in flows:
            self.state.pop(f.flow_index, None)

    def finish(self) -> List[Dict[str, Any]]:
        """End of input: finalise every flow, score what is pending, forget all flow state."""
        for f in self.tracker.flush():
            self._finalise(f)
        out = self.run_inference()
        self.state.clear()
        return out

    def reset_capture(self) -> None:
        """Start a fresh, independent input (e.g. the next PCAP in a replay): new flow table, no alert history."""
        self.tracker = FlowTracker(self.cfg.flow)
        self.parse_stats = ParseStats()
        self.state, self.queue, self._last_alert, self._since_drain, self.now = {}, [], {}, 0, 0.0

    def summary(self) -> Dict[str, Any]:
        s = dict(self.stats)
        s["mean_inference_ms_per_window"] = 1000 * s["inference_seconds"] / max(s["inferences"], 1)
        s["active_flows"] = len(self.tracker._active)  # noqa: SLF001
        return s
