"""Bidirectional flow tracking with explicit connection lifecycle handling.

Flow identity
-------------
A flow key is the canonical (ip_proto, endpoint_a, endpoint_b) with the two (ip, port)
endpoints sorted, so both directions map to the same key. A key is NOT a flow identity by
itself: repeated connections reusing a 5-tuple become separate `Flow` objects (`flow_index`).

When a new flow starts for an existing key
------------------------------------------
* TCP: a pure SYN (SYN without ACK) arrives and the existing flow is closed, already past its
  handshake with a different ISN, or the SYN comes from the other endpoint.
  (A SYN retransmission with the same ISN from the same side stays in the flow.)
* TCP: the existing flow was closed (FIN seen both ways, or RST) and the next packet arrives
  after `tcp_time_wait_s` -> late packets within time-wait still join the old flow.
* Any protocol: idle gap > idle timeout, or flow age > max_flow_duration_s.
  Idle-expired => new flow, initiator re-guessed from its first packet (warning `midstream_start`).
  Max-duration => the continuation keeps the original initiator (warning `continued_after_max_duration`).

Initiator
---------
* TCP flow beginning with SYN (no ACK): sender of the SYN.
* TCP flow beginning with SYN+ACK (capture started mid-handshake): the RECEIVER of that packet.
* Otherwise: sender of the first packet seen, flagged `midstream_start` for TCP flows.
  UDP/other flows have no handshake, so "initiator = first sender" is the definition and is
  not flagged.

Retransmissions (TCP)
---------------------
Per direction we track the highest sequence number already covered. A data segment whose end
is not beyond it (32-bit modular compare) is a retransmission: it is counted in
`flow.retransmissions` and, if `drop_tcp_retransmissions`, not added to the packet sequence.
Pure ACKs / SYN / FIN / RST are never classified as retransmissions.

Memory
------
Only the first `max_packets_per_flow` packets of a flow are stored; later ones are counted in
`flow.packets_dropped_cap` and the flow is flagged `packet_cap_reached`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from src.data.parse_pcap import TCP_ACK, TCP_FIN, TCP_RST, TCP_SYN, ParsedPacket
from src.utils.config import PROTO_DNS, PROTO_TCP, PROTO_UDP, FlowConfig

log = logging.getLogger(__name__)

Endpoint = Tuple[str, int]
FlowKey = Tuple[int, Endpoint, Endpoint]


@dataclass(slots=True)
class PacketRecord:
    ts: float
    direction: int       # +1 initiator->responder, -1 responder->initiator
    category: int
    payload_len: int
    payload: bytes
    tcp_flags: int
    warnings: Tuple[str, ...] = ()


@dataclass
class Flow:
    flow_index: int
    key: FlowKey
    initiator: Endpoint
    responder: Endpoint
    ip_proto: int
    category: int
    start_ts: float
    last_ts: float
    packets: List[PacketRecord] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    retransmissions: int = 0
    packets_dropped_cap: int = 0
    total_packets_seen: int = 0
    syn_seen: bool = False
    syn_seq: Optional[int] = None
    fin_dir: set = field(default_factory=set)
    closed: bool = False
    closed_ts: float = 0.0
    max_end_seq: Dict[int, int] = field(default_factory=dict)  # direction -> highest covered seq
    end_reason: str = ""

    def add_warning(self, w: str) -> None:
        if w not in self.warnings:
            self.warnings.append(w)


def make_key(p: ParsedPacket) -> FlowKey:
    a: Endpoint = (p.src_ip, p.sport)
    b: Endpoint = (p.dst_ip, p.dport)
    return (p.ip_proto, a, b) if a <= b else (p.ip_proto, b, a)


def _seq_le(a: int, b: int) -> bool:
    """a <= b in 32-bit modular sequence space."""
    return ((b - a) & 0xFFFFFFFF) < 0x80000000


class FlowTracker:
    def __init__(self, cfg: FlowConfig):
        self.cfg = cfg
        self._active: Dict[FlowKey, Flow] = {}
        self._finished: List[Flow] = []
        self._next_index = 0
        self._last_sweep_ts: Optional[float] = None
        self.stats = {"flows_started": 0, "flows_finished": 0, "retransmissions_dropped": 0,
                      "packets_dropped_cap": 0}

    # ------------------------------------------------------------------ public API
    def process(self, pkt: ParsedPacket) -> Optional[Flow]:
        """Add one packet. Returns the Flow the packet was APPENDED to, or None if it was not kept
        (dropped retransmission / per-flow packet cap). Offline callers ignore the return value;
        the streaming engine uses it to react to a growing flow."""
        self._maybe_sweep(pkt.ts)
        key = make_key(pkt)
        flow = self._active.get(key)
        if flow is not None:
            reason = self._should_split(flow, pkt)
            if reason:
                self._finish(flow, reason)
                del self._active[key]
                flow = self._start_flow(key, pkt, continued=(reason == "max_duration"), prev=flow)
        else:
            flow = self._start_flow(key, pkt, continued=False, prev=None)
        n_before = len(flow.packets)
        self._add_packet(flow, pkt)
        return flow if len(flow.packets) > n_before else None

    def drain_finished(self) -> List[Flow]:
        """Streaming use: return (and forget) flows finished so far (idle timeout, new SYN, max duration,
        eviction). Keeps memory bounded on a never-ending capture."""
        done, self._finished = self._finished, []
        return done

    def evict_oldest(self, keep: int) -> int:
        """Finish the least-recently-active flows until at most `keep` remain active (memory guard)."""
        extra = len(self._active) - keep
        if extra <= 0:
            return 0
        for k in sorted(self._active, key=lambda k: self._active[k].last_ts)[:extra]:
            self._finish(self._active.pop(k), "evicted")
        return extra

    def flush(self) -> List[Flow]:
        """Finish every still-open flow (end of capture) and return ALL flows ordered by
        creation index (deterministic output order)."""
        for flow in list(self._active.values()):
            if flow.ip_proto == 6 and not flow.closed:  # UDP/other have no close event to miss
                flow.add_warning("incomplete_at_capture_end")
            self._finish(flow, "capture_end")
        self._active.clear()
        flows = sorted(self._finished, key=lambda f: f.flow_index)
        self._finished = []
        return flows

    def run(self, packets: Iterable[ParsedPacket]) -> List[Flow]:
        for p in packets:
            self.process(p)
        return self.flush()

    # ------------------------------------------------------------------ internals
    def _idle_timeout(self, category: int) -> float:
        if category == PROTO_TCP:
            return self.cfg.tcp_idle_timeout_s
        if category in (PROTO_UDP, PROTO_DNS):
            return self.cfg.udp_idle_timeout_s
        return self.cfg.other_idle_timeout_s

    def _should_split(self, flow: Flow, pkt: ParsedPacket) -> Optional[str]:
        if pkt.ts - flow.last_ts > self._idle_timeout(flow.category):
            return "idle_timeout"
        if pkt.ts - flow.start_ts > self.cfg.max_flow_duration_s:
            return "max_duration"
        if flow.category == PROTO_TCP or pkt.ip_proto == 6:
            pure_syn = bool(pkt.tcp_flags & TCP_SYN) and not (pkt.tcp_flags & TCP_ACK)
            if pure_syn:
                same_side = (pkt.src_ip, pkt.sport) == flow.initiator
                is_retx_of_first = same_side and flow.syn_seq is not None and pkt.seq == flow.syn_seq
                if not is_retx_of_first:
                    if flow.closed or flow.syn_seen or len(flow.packets) > 0:
                        return "new_syn"
            if flow.closed and (pkt.ts - flow.closed_ts) > self.cfg.tcp_time_wait_s:
                return "after_close"
        return None

    def _start_flow(self, key: FlowKey, pkt: ParsedPacket, continued: bool,
                    prev: Optional[Flow]) -> Flow:
        src: Endpoint = (pkt.src_ip, pkt.sport)
        dst: Endpoint = (pkt.dst_ip, pkt.dport)
        warnings: List[str] = []
        if continued and prev is not None:
            initiator, responder = prev.initiator, prev.responder
            warnings.append("continued_after_max_duration")
        else:
            initiator, responder = src, dst
            if pkt.ip_proto == 6:
                syn = bool(pkt.tcp_flags & TCP_SYN)
                ack = bool(pkt.tcp_flags & TCP_ACK)
                if syn and ack:
                    initiator, responder = dst, src  # SYN+ACK: the receiver opened the connection
                elif not syn:
                    warnings.append("midstream_start")
            if prev is not None:
                warnings.append("restarted_same_5tuple")
        flow = Flow(flow_index=self._next_index, key=key, initiator=initiator, responder=responder,
                    ip_proto=pkt.ip_proto, category=pkt.category, start_ts=pkt.ts, last_ts=pkt.ts,
                    warnings=warnings)
        self._next_index += 1
        self._active[key] = flow
        self.stats["flows_started"] += 1
        return flow

    def _add_packet(self, flow: Flow, pkt: ParsedPacket) -> None:
        flow.total_packets_seen += 1
        flow.last_ts = max(flow.last_ts, pkt.ts)
        direction = 1 if (pkt.src_ip, pkt.sport) == flow.initiator else -1
        extra: List[str] = list(pkt.warnings)

        if pkt.ip_proto == 6:
            flags = pkt.tcp_flags
            if flags & TCP_SYN and not (flags & TCP_ACK) and not flow.syn_seen:
                flow.syn_seen = True
                flow.syn_seq = pkt.seq
            if flags & TCP_FIN:
                flow.fin_dir.add(direction)
            if flags & TCP_RST:
                self._close(flow, pkt.ts)
            elif len(flow.fin_dir) == 2:
                self._close(flow, pkt.ts)
            # retransmission detection (data segments only)
            if pkt.payload_len > 0 and not (flags & (TCP_SYN | TCP_RST)):
                end = (pkt.seq + pkt.payload_len) & 0xFFFFFFFF
                prev_end = flow.max_end_seq.get(direction)
                if prev_end is not None and _seq_le(end, prev_end):
                    flow.retransmissions += 1
                    flow.add_warning("tcp_retransmissions_present")
                    if self.cfg.drop_tcp_retransmissions:
                        self.stats["retransmissions_dropped"] += 1
                        return
                    extra.append("retransmission")
                if prev_end is None or not _seq_le(end, prev_end):
                    flow.max_end_seq[direction] = end
            elif flags & TCP_SYN:
                flow.max_end_seq[direction] = (pkt.seq + 1) & 0xFFFFFFFF

        if len(flow.packets) >= self.cfg.max_packets_per_flow:
            flow.packets_dropped_cap += 1
            self.stats["packets_dropped_cap"] += 1
            flow.add_warning("packet_cap_reached")
            return
        flow.packets.append(PacketRecord(ts=pkt.ts, direction=direction, category=pkt.category,
                                         payload_len=pkt.payload_len, payload=pkt.payload,
                                         tcp_flags=pkt.tcp_flags, warnings=tuple(extra)))

    def _close(self, flow: Flow, ts: float) -> None:
        if not flow.closed:
            flow.closed = True
            flow.closed_ts = ts

    def _finish(self, flow: Flow, reason: str) -> None:
        flow.end_reason = reason
        if reason == "idle_timeout" and not flow.closed:
            flow.add_warning("ended_by_idle_timeout")
        elif reason == "max_duration":
            flow.add_warning("split_by_max_duration")
        elif reason == "new_syn" and not flow.closed:
            flow.add_warning("ended_by_new_syn_without_close")
        self._finished.append(flow)
        self.stats["flows_finished"] += 1

    def _maybe_sweep(self, now: float) -> None:
        if self._last_sweep_ts is None:
            self._last_sweep_ts = now
            return
        if now - self._last_sweep_ts < self.cfg.expire_check_interval_s:
            return
        self._last_sweep_ts = now
        expired = [k for k, f in self._active.items()
                   if now - f.last_ts > self._idle_timeout(f.category)]
        for k in expired:
            f = self._active.pop(k)
            self._finish(f, "idle_timeout")
