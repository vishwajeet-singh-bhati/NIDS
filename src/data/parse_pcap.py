"""Streaming PCAP/PCAPNG packet parser.

Design notes
------------
* Scapy is used ONLY to read the capture container (`RawPcapReader`, which also handles
  pcapng). Headers are decoded here with `struct`, because we need exact control over:
    - payload length (derived from the IP/transport length fields, NOT from captured bytes,
      so snaplen-truncated captures still report the true payload size),
    - Ethernet trailer padding (never counted as payload),
    - truncation / malformed detection.
* Supported link types: Ethernet (+up to 2 VLAN tags), Linux cooked SLL / SLL2, raw IP,
  BSD loopback (NULL). Anything else is counted and reported, never silently decoded.
* IPv4 and IPv6 (hop-by-hop / routing / destination / fragment / AH extension headers).
* Handling of awkward traffic (also documented in README):
    - IP fragments: the FIRST fragment is parsed (flagged); non-first fragments have no
      transport header and are COUNTED AND SKIPPED. No reassembly is attempted.
    - TCP segmentation / offload: segments are treated as captured. If the IP total length is 0
      (TSO artefact), the captured length is used and a warning is recorded.
    - Retransmissions are detected in flow_tracker (needs per-flow sequence state).
    - Missing/zero timestamps: replaced by the previous packet's timestamp and counted.
    - Out-of-order timestamps: kept in file order, counted; inter-arrival time is clamped >= 0.
    - Truncated capture file / mid-file read errors: packets read so far are kept and the
      stats record `file_truncated_or_corrupt`.
* Encrypted payloads (TLS/QUIC) are NOT decrypted; payload bytes are ciphertext.
"""
from __future__ import annotations

import logging
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

from scapy.utils import RawPcapReader

from src.utils.config import (PROTO_DNS, PROTO_OTHER, PROTO_TCP, PROTO_UDP, ParseConfig)

log = logging.getLogger(__name__)

# libpcap link-layer header types
DLT_NULL = 0
DLT_EN10MB = 1
DLT_RAW = 101
DLT_RAW_12 = 12
DLT_RAW_14 = 14
DLT_LINUX_SLL = 113
DLT_IPV4 = 228
DLT_IPV6 = 229
DLT_LINUX_SLL2 = 276
SUPPORTED_LINKTYPES = {DLT_NULL, DLT_EN10MB, DLT_RAW, DLT_RAW_12, DLT_RAW_14, DLT_LINUX_SLL,
                       DLT_IPV4, DLT_IPV6, DLT_LINUX_SLL2}

TCP_FIN, TCP_SYN, TCP_RST, TCP_PSH, TCP_ACK = 0x01, 0x02, 0x04, 0x08, 0x10

IPPROTO_TCP, IPPROTO_UDP = 6, 17
_V6_SKIP_EXT = {0, 43, 60}  # hop-by-hop, routing, destination options (length in 8-byte units)
_V6_FRAG, _V6_AH, _V6_NONE = 44, 51, 59


@dataclass(slots=True)
class ParsedPacket:
    ts: float
    ip_version: int
    ip_proto: int            # IANA protocol number (6 TCP, 17 UDP, ...)
    src_ip: str
    dst_ip: str
    sport: int               # 0 for protocols without ports
    dport: int
    category: int            # PROTO_TCP / PROTO_UDP / PROTO_DNS / PROTO_OTHER
    tcp_flags: int           # 0 for non-TCP
    seq: int                 # TCP sequence number, 0 otherwise
    ack: int
    payload_len: int         # transport payload length declared by IP/transport headers
    payload: bytes           # captured payload bytes, truncated to cfg.payload_bytes_kept
    wirelen: int
    caplen: int
    warnings: Tuple[str, ...] = ()


@dataclass
class ParseStats:
    path: str = ""
    linktype: Optional[int] = None
    file_format: str = ""
    total_records: int = 0
    parsed: int = 0
    non_ip: int = 0
    unsupported_linktype: int = 0
    malformed: int = 0
    ip_fragment_first: int = 0
    ip_fragment_nonfirst_skipped: int = 0
    truncated_by_snaplen: int = 0
    ip_total_length_zero: int = 0
    bad_timestamp: int = 0
    non_monotonic_timestamp: int = 0
    ipv4: int = 0
    ipv6: int = 0
    tcp: int = 0
    udp: int = 0
    dns: int = 0
    other: int = 0
    file_truncated_or_corrupt: bool = False
    open_error: str = ""
    hit_packet_limit: bool = False
    first_ts: Optional[float] = None
    last_ts: Optional[float] = None

    def as_dict(self) -> Dict[str, object]:
        return dict(self.__dict__)

    def warning_tags(self) -> Tuple[str, ...]:
        tags = []
        if self.open_error:
            tags.append("open_error")
        if self.file_truncated_or_corrupt:
            tags.append("file_truncated_or_corrupt")
        if self.unsupported_linktype:
            tags.append("unsupported_linktype")
        if self.malformed:
            tags.append("malformed_packets_present")
        if self.ip_fragment_nonfirst_skipped:
            tags.append("ip_fragments_skipped")
        if self.truncated_by_snaplen:
            tags.append("snaplen_truncation_present")
        if self.bad_timestamp:
            tags.append("bad_timestamps_present")
        # non_monotonic_timestamp is deliberately NOT a per-row tag: on real captures it is nearly
        # universal. It is recorded per window (manifest `n_ts_backward`) and in processing_log.
        if self.hit_packet_limit:
            tags.append("packet_limit_reached")
        return tuple(tags)


def _ip_str_v4(b: bytes) -> str:
    return f"{b[0]}.{b[1]}.{b[2]}.{b[3]}"


def _ip_str_v6(b: bytes) -> str:
    groups = struct.unpack("!8H", b)
    return ":".join(f"{g:x}" for g in groups)  # uncompressed, canonical enough for keys


def _strip_link_layer(buf: bytes, linktype: int) -> Optional[Tuple[int, int]]:
    """Return (ip_version, offset_of_ip_header) or None when the frame is not IP."""
    n = len(buf)
    if linktype == DLT_EN10MB:
        if n < 14:
            return None
        off = 12
        ethertype = struct.unpack_from("!H", buf, off)[0]
        off += 2
        for _ in range(2):  # up to two VLAN tags (802.1Q / 802.1ad)
            if ethertype in (0x8100, 0x88A8, 0x9100):
                if n < off + 4:
                    return None
                ethertype = struct.unpack_from("!H", buf, off + 2)[0]
                off += 4
        if ethertype == 0x0800:
            return 4, off
        if ethertype == 0x86DD:
            return 6, off
        return None
    if linktype == DLT_LINUX_SLL:
        if n < 16:
            return None
        proto = struct.unpack_from("!H", buf, 14)[0]
        off = 16
    elif linktype == DLT_LINUX_SLL2:
        if n < 20:
            return None
        proto = struct.unpack_from("!H", buf, 0)[0]
        off = 20
    elif linktype == DLT_NULL:
        if n < 4:
            return None
        fam_le = struct.unpack_from("<I", buf, 0)[0]
        fam_be = struct.unpack_from(">I", buf, 0)[0]
        fam = fam_le if fam_le < 256 else fam_be
        off = 4
        proto = 0x0800 if fam == 2 else 0x86DD if fam in (10, 23, 24, 28, 30) else 0
    elif linktype in (DLT_RAW, DLT_RAW_12, DLT_RAW_14, DLT_IPV4, DLT_IPV6):
        if n < 1:
            return None
        ver = buf[0] >> 4
        return (ver, 0) if ver in (4, 6) else None
    else:
        return None
    if proto == 0x0800:
        return 4, off
    if proto == 0x86DD:
        return 6, off
    return None


class _Malformed(Exception):
    pass


def decode_packet(buf: bytes, linktype: int, ts: float, wirelen: int, cfg: ParseConfig,
                  stats: ParseStats) -> Optional[ParsedPacket]:
    """Decode one captured frame. Returns None if the frame should not become a packet record
    (non-IP, malformed, non-first fragment). Every None path increments a counter."""
    located = _strip_link_layer(buf, linktype)
    if located is None:
        stats.non_ip += 1
        return None
    ver, ip_off = located
    caplen = len(buf)
    warnings = []
    try:
        if ver == 4:
            if caplen < ip_off + 20:
                raise _Malformed("short_ipv4")
            vihl = buf[ip_off]
            ihl = (vihl & 0x0F) * 4
            if (vihl >> 4) != 4 or ihl < 20:
                raise _Malformed("bad_ipv4_header")
            total_len = struct.unpack_from("!H", buf, ip_off + 2)[0]
            fl_off = struct.unpack_from("!H", buf, ip_off + 6)[0]
            frag_off = fl_off & 0x1FFF
            more_frag = bool(fl_off & 0x2000)
            proto = buf[ip_off + 9]
            src = _ip_str_v4(buf[ip_off + 12:ip_off + 16])
            dst = _ip_str_v4(buf[ip_off + 16:ip_off + 20])
            if total_len == 0:
                stats.ip_total_length_zero += 1
                warnings.append("ip_total_length_zero")
                declared_end = caplen
            elif total_len < ihl:
                raise _Malformed("ipv4_total_len_lt_ihl")
            else:
                declared_end = ip_off + total_len
            l4_off = ip_off + ihl
            if frag_off > 0:
                stats.ip_fragment_nonfirst_skipped += 1
                return None
            if more_frag:
                stats.ip_fragment_first += 1
                warnings.append("ip_fragment_first")
        else:
            if caplen < ip_off + 40:
                raise _Malformed("short_ipv6")
            if (buf[ip_off] >> 4) != 6:
                raise _Malformed("bad_ipv6_header")
            plen = struct.unpack_from("!H", buf, ip_off + 4)[0]
            nxt = buf[ip_off + 6]
            src = _ip_str_v6(buf[ip_off + 8:ip_off + 24])
            dst = _ip_str_v6(buf[ip_off + 24:ip_off + 40])
            declared_end = ip_off + 40 + plen if plen else caplen
            if plen == 0:
                stats.ip_total_length_zero += 1
                warnings.append("ip_total_length_zero")
            l4_off = ip_off + 40
            for _ in range(8):  # bounded extension-header walk
                if nxt in _V6_SKIP_EXT:
                    if caplen < l4_off + 2:
                        raise _Malformed("short_ipv6_ext")
                    nxt, hl = buf[l4_off], (buf[l4_off + 1] + 1) * 8
                    l4_off += hl
                elif nxt == _V6_AH:
                    if caplen < l4_off + 2:
                        raise _Malformed("short_ipv6_ah")
                    nxt, hl = buf[l4_off], (buf[l4_off + 1] + 2) * 4
                    l4_off += hl
                elif nxt == _V6_FRAG:
                    if caplen < l4_off + 8:
                        raise _Malformed("short_ipv6_frag")
                    fo = struct.unpack_from("!H", buf, l4_off + 2)[0]
                    if (fo >> 3) > 0:
                        stats.ip_fragment_nonfirst_skipped += 1
                        return None
                    stats.ip_fragment_first += 1
                    warnings.append("ip_fragment_first")
                    nxt = buf[l4_off]
                    l4_off += 8
                else:
                    break
            proto = nxt
            if proto == _V6_NONE:
                proto = 59
        declared_end = max(declared_end, l4_off)
        sport = dport = 0
        flags = seq = ack = 0
        if proto == IPPROTO_TCP:
            if caplen < l4_off + 20:
                raise _Malformed("short_tcp")
            sport, dport, seq, ack, off_flags = struct.unpack_from("!HHIIH", buf, l4_off)
            doff = (off_flags >> 12) * 4
            if doff < 20:
                raise _Malformed("bad_tcp_offset")
            flags = off_flags & 0xFF
            hdr_end = l4_off + doff
            category = PROTO_TCP
        elif proto == IPPROTO_UDP:
            if caplen < l4_off + 8:
                raise _Malformed("short_udp")
            sport, dport = struct.unpack_from("!HH", buf, l4_off)
            hdr_end = l4_off + 8
            category = PROTO_UDP
        else:
            hdr_end = l4_off  # "other": everything after the IP header counts as payload
            category = PROTO_OTHER
        if category in (PROTO_TCP, PROTO_UDP) and (sport in cfg.dns_ports or dport in cfg.dns_ports):
            category = PROTO_DNS
        payload_len = max(0, declared_end - hdr_end)
        cap_end = min(caplen, declared_end)
        if declared_end > caplen:
            stats.truncated_by_snaplen += 1
            warnings.append("snaplen_truncated")
        keep = min(cfg.payload_bytes_kept, payload_len, max(0, cap_end - hdr_end))
        payload = bytes(buf[hdr_end:hdr_end + keep]) if keep > 0 else b""
        if proto == IPPROTO_TCP and hdr_end > cap_end and declared_end <= caplen:
            raise _Malformed("tcp_header_exceeds_packet")
    except _Malformed:
        stats.malformed += 1
        return None
    except struct.error:
        stats.malformed += 1
        return None

    if ver == 4:
        stats.ipv4 += 1
    else:
        stats.ipv6 += 1
    if category == PROTO_TCP:
        stats.tcp += 1
    elif category == PROTO_UDP:
        stats.udp += 1
    elif category == PROTO_DNS:
        stats.dns += 1
    else:
        stats.other += 1
    return ParsedPacket(ts=ts, ip_version=ver, ip_proto=proto, src_ip=src, dst_ip=dst, sport=sport,
                        dport=dport, category=category, tcp_flags=flags, seq=seq, ack=ack,
                        payload_len=payload_len, payload=payload, wirelen=wirelen, caplen=caplen,
                        warnings=tuple(warnings))


def _timestamp(meta, nano: bool) -> float:
    """Convert scapy's raw-reader metadata to float seconds (pcap and pcapng)."""
    if hasattr(meta, "tshigh"):  # pcapng
        raw = (int(meta.tshigh) << 32) | int(meta.tslow)
        return raw / float(meta.tsresol or 1_000_000)
    return float(meta.sec) + float(meta.usec) / (1e9 if nano else 1e6)


def iter_packets(path: str | Path, cfg: ParseConfig, stats: Optional[ParseStats] = None
                 ) -> Iterator[ParsedPacket]:
    """Yield ParsedPacket objects in capture-file order. `stats` is filled as a side effect and is
    complete once the generator is exhausted."""
    path = Path(path)
    stats = stats if stats is not None else ParseStats()
    stats.path = str(path)
    try:
        reader = RawPcapReader(str(path))
    except Exception as exc:  # scapy raises Scapy_Exception, OSError, struct.error, ...
        stats.open_error = f"{type(exc).__name__}: {exc}"
        log.error("Cannot open %s: %s", path, stats.open_error)
        return
    is_ng = hasattr(reader, "if_tsresol") or type(reader).__name__ == "RawPcapNgReader"
    stats.file_format = "pcapng" if is_ng else "pcap"
    nano = bool(getattr(reader, "nano", False))
    file_linktype = getattr(reader, "linktype", None)
    stats.linktype = file_linktype
    prev_ts: Optional[float] = None
    warned_linktypes = set()
    try:
        while True:
            try:
                data, meta = reader._read_packet()  # noqa: SLF001 (public API lacks raw access)
            except EOFError:
                break
            except Exception as exc:
                stats.file_truncated_or_corrupt = True
                log.warning("%s: read error after %d records: %s", path.name, stats.total_records, exc)
                break
            stats.total_records += 1
            if cfg.max_packets_per_capture and stats.total_records > cfg.max_packets_per_capture:
                stats.hit_packet_limit = True
                break
            linktype = getattr(meta, "linktype", None) if is_ng else file_linktype
            if linktype is None:
                linktype = DLT_EN10MB
            if stats.linktype is None:
                stats.linktype = linktype
            if linktype not in SUPPORTED_LINKTYPES:
                stats.unsupported_linktype += 1
                if linktype not in warned_linktypes:
                    warned_linktypes.add(linktype)
                    log.warning("%s: unsupported link type %s; those packets are skipped", path.name,
                                linktype)
                continue
            ts = _timestamp(meta, nano)
            if ts <= 0 or ts != ts:
                stats.bad_timestamp += 1
                ts = prev_ts if prev_ts is not None else 0.0
            elif prev_ts is not None and ts < prev_ts:
                stats.non_monotonic_timestamp += 1
            pkt = decode_packet(data, linktype, ts, int(getattr(meta, "wirelen", len(data))), cfg, stats)
            if pkt is None:
                prev_ts = ts if ts > 0 else prev_ts
                continue
            prev_ts = ts
            if stats.first_ts is None:
                stats.first_ts = ts
            stats.last_ts = ts
            stats.parsed += 1
            yield pkt
    finally:
        try:
            reader.close()
        except Exception:  # pragma: no cover
            pass
