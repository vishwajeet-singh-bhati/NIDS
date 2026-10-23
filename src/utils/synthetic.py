"""Synthetic PCAP generation for PIPELINE TESTING ONLY.

These captures are hand-built (scapy) with explicit timestamps. They exercise the parser, flow
tracker and encoder. They are NOT malware traffic and must never be presented as real-world
validation of any detector.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional

from scapy.all import DNS, DNSQR, Ether, IP, IPv6, TCP, UDP, ICMP, Raw, wrpcap  # type: ignore

CLIENT = "10.0.0.2"
SERVER = "93.184.216.34"
DNS_SERVER = "8.8.8.8"
MAC_A, MAC_B = "02:00:00:00:00:01", "02:00:00:00:00:02"


def _eth(ip_layer, t: float):
    p = Ether(src=MAC_A, dst=MAC_B) / ip_layer
    p.time = t
    return p


def _rev_eth(ip_layer, t: float):
    p = Ether(src=MAC_B, dst=MAC_A) / ip_layer
    p.time = t
    return p


def tcp_session(t0: float, sport: int, dport: int = 443, n_exchanges: int = 5,
                client: str = CLIENT, server: str = SERVER, isn_c: int = 1000, isn_s: int = 5000,
                payload_c: int = 100, payload_s: int = 400, gap: float = 0.05, close: bool = True,
                retransmit_first_request: bool = False) -> List:
    pk, t = [], t0
    pk.append(_eth(IP(src=client, dst=server) / TCP(sport=sport, dport=dport, flags="S", seq=isn_c), t))
    t += gap
    pk.append(_rev_eth(IP(src=server, dst=client) / TCP(sport=dport, dport=sport, flags="SA", seq=isn_s, ack=isn_c + 1), t))
    t += gap
    seq_c, seq_s = isn_c + 1, isn_s + 1
    pk.append(_eth(IP(src=client, dst=server) / TCP(sport=sport, dport=dport, flags="A", seq=seq_c, ack=seq_s), t))
    for i in range(n_exchanges):
        t += gap
        data = bytes([(i * 7 + j) % 251 + 1 for j in range(payload_c)])  # never 0x00: easy to spot padding bugs
        req = _eth(IP(src=client, dst=server) / TCP(sport=sport, dport=dport, flags="PA", seq=seq_c, ack=seq_s) / Raw(data), t)
        pk.append(req)
        if i == 0 and retransmit_first_request:
            t += gap
            dup = _eth(IP(src=client, dst=server) / TCP(sport=sport, dport=dport, flags="PA", seq=seq_c, ack=seq_s) / Raw(data), t)
            pk.append(dup)
        seq_c += payload_c
        t += gap
        resp = bytes([(i * 13 + j) % 251 + 1 for j in range(payload_s)])
        pk.append(_rev_eth(IP(src=server, dst=client) / TCP(sport=dport, dport=sport, flags="PA", seq=seq_s, ack=seq_c) / Raw(resp), t))
        seq_s += payload_s
    if close:
        t += gap
        pk.append(_eth(IP(src=client, dst=server) / TCP(sport=sport, dport=dport, flags="FA", seq=seq_c, ack=seq_s), t))
        t += gap
        pk.append(_rev_eth(IP(src=server, dst=client) / TCP(sport=dport, dport=sport, flags="FA", seq=seq_s, ack=seq_c + 1), t))
        t += gap
        pk.append(_eth(IP(src=client, dst=server) / TCP(sport=sport, dport=dport, flags="A", seq=seq_c + 1, ack=seq_s + 1), t))
    return pk


def udp_session(t0: float, sport: int, dport: int = 5000, n: int = 6, client: str = CLIENT,
                server: str = SERVER, size: int = 60, gap: float = 0.1) -> List:
    pk, t = [], t0
    for i in range(n):
        pk.append(_eth(IP(src=client, dst=server) / UDP(sport=sport, dport=dport) / Raw(bytes([i + 1]) * size), t))
        t += gap
        pk.append(_rev_eth(IP(src=server, dst=client) / UDP(sport=dport, dport=sport) / Raw(bytes([i + 100]) * size), t))
        t += gap
    return pk


def dns_exchange(t0: float, sport: int, name: str = "example.com", server: str = DNS_SERVER) -> List:
    DNS_SERVER_ = server
    q = _eth(IP(src=CLIENT, dst=server) / UDP(sport=sport, dport=53) / DNS(rd=1, qd=DNSQR(qname=name)), t0)
    r = _rev_eth(IP(src=server, dst=CLIENT) / UDP(sport=53, dport=sport) / DNS(qr=1, qd=DNSQR(qname=name)), t0 + 0.02)
    return [q, r]


def ipv6_tcp_session(t0: float, sport: int = 40000) -> List:
    c, s = "2001:db8::2", "2001:db8::99"
    pk, t = [], t0
    pk.append(_eth(IPv6(src=c, dst=s) / TCP(sport=sport, dport=443, flags="S", seq=1), t)); t += .05
    pk.append(_rev_eth(IPv6(src=s, dst=c) / TCP(sport=443, dport=sport, flags="SA", seq=9, ack=2), t)); t += .05
    pk.append(_eth(IPv6(src=c, dst=s) / TCP(sport=sport, dport=443, flags="PA", seq=2, ack=10) / Raw(b"\x01" * 50), t))
    return pk


def icmp_echo(t0: float) -> List:
    return [_eth(IP(src=CLIENT, dst=SERVER) / ICMP() / Raw(b"\x05" * 32), t0),
            _rev_eth(IP(src=SERVER, dst=CLIENT) / ICMP(type=0) / Raw(b"\x05" * 32), t0 + 0.03)]


def benign_shaped_capture(t0: float = 1_700_000_000.0, seed: int = 0) -> List:
    """Many short-ish TCP sessions + DNS: just a mix to exercise the pipeline."""
    rng = random.Random(seed)
    pk: List = []
    t = t0
    for i in range(6):
        pk += dns_exchange(t, 50000 + i)
        pk += tcp_session(t + 0.1, 41000 + i, 443, n_exchanges=rng.randint(3, 8), payload_c=rng.randint(50, 200),
                          payload_s=rng.randint(200, 1200))
        t += 1.0
    pk += icmp_echo(t)
    return sorted(pk, key=lambda p: float(p.time))


def beacon_shaped_capture(t0: float = 1_700_100_000.0, seed: int = 1) -> List:
    """Periodic small request/response 'beacon'-like pattern. Synthetic shape only."""
    pk: List = []
    t = t0
    for i in range(12):
        pk += tcp_session(t, 42000 + i, 8080, n_exchanges=1, payload_c=40, payload_s=40, gap=0.02)
        t += 5.0
    return sorted(pk, key=lambda p: float(p.time))


def write_pcap(path: str | Path, packets: List, snaplen_cut: Optional[int] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wrpcap(str(path), packets)
    if snaplen_cut:
        _truncate_records(path, snaplen_cut)
    return path


def _truncate_records(path: Path, cut: int) -> None:
    """Rewrite a classic little-endian pcap so each record is cut to `cut` captured bytes
    (wirelen preserved) to simulate a small snaplen."""
    import struct
    data = path.read_bytes()
    gh, off, out = data[:24], 24, bytearray(data[:24])
    while off + 16 <= len(data):
        sec, usec, caplen, wirelen = struct.unpack_from("<IIII", data, off)
        body = data[off + 16: off + 16 + caplen]
        newlen = min(caplen, cut)
        out += struct.pack("<IIII", sec, usec, newlen, wirelen) + body[:newlen]
        off += 16 + caplen
    path.write_bytes(bytes(out))


def mdns_announce(t0: float, n: int = 2) -> List:
    return [_eth(IP(src=CLIENT, dst="224.0.0.251") / UDP(sport=5353, dport=5353) / Raw(b"\x00" * 30), t0 + i * 0.5)
            for i in range(n)]


def varied_capture(kind: str, seed: int, t0: float) -> List:
    """Synthetic capture for the AndMal-like layout. kind in {"benign","malware"}.
    Both kinds contain the same local background (gateway DNS, mDNS) on purpose, so that the label
    audit has something to find. Shapes are arbitrary and say nothing about real malware."""
    rng = random.Random(seed)
    pk: List = []
    t = t0
    for i in range(rng.randint(3, 6)):
        pk += dns_exchange(t, 50000 + i, server="10.0.0.1")
        t += 0.3
    pk += mdns_announce(t)
    n_sess = rng.randint(6, 14)
    for i in range(n_sess):
        if kind == "malware":
            pk += tcp_session(t, 42000 + i, 8080 + rng.choice([0, 1]), n_exchanges=rng.randint(1, 3),
                              payload_c=rng.randint(20, 60), payload_s=rng.randint(20, 80), gap=0.02,
                              server=f"45.33.32.{rng.randint(1, 8)}")
            t += rng.uniform(3, 7)
        else:
            pk += tcp_session(t, 41000 + i, 443, n_exchanges=rng.randint(3, 12), payload_c=rng.randint(50, 300),
                              payload_s=rng.randint(200, 1400), server=f"93.184.216.{rng.randint(1, 40)}")
            t += rng.uniform(0.2, 1.5)
    return sorted(pk, key=lambda p: float(p.time))


def make_andmal_like(out: Path, n_benign: int, n_families: int, per_family: int) -> None:
    """CIC-AndMal2017-LIKE layout (so the real configs/labels.yaml rules apply):
        Benign/2015/<date>-be-2015-[OK-n-]<package>.pcap
        <Category>/<Family>/<date>-<xx>-<family>-<vendor>-<md5>.pcap
    Some benign apps get two captures (same package) to exercise app-level grouping."""
    import hashlib
    cats = [("Adware", "ad"), ("Ransomware", "ra"), ("Scareware", "sc"), ("SMSMalware", "sms")]
    t = 1_700_000_000.0
    for i in range(n_benign):
        pkg = f"com.example.app{i // 2:03d}"       # two captures per package
        tag = "OK-1-" if i % 2 == 0 else "OK-3-"
        write_pcap(out / f"Benign/2015/08_0{4 + i % 3}_2017-be-2015-{tag}{pkg}.pcap",
                   varied_capture("benign", seed=i, t0=t + i * 1000))
    for f in range(n_families):
        cat, short = cats[f % len(cats)]
        fam = f"Fam{f:02d}"
        for k in range(per_family):
            md5 = hashlib.md5(f"{fam}-{k}".encode()).hexdigest()
            write_pcap(out / f"{cat}/{fam}/06_{14 + k % 5}_2017-{short}-{fam.lower()}-vendor-{md5}.pcap",
                       varied_capture("malware", seed=1000 + f * 50 + k, t0=t + 5_000_000 + (f * 50 + k) * 1000))
