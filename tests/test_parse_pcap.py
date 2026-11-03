import struct

import pytest
from scapy.all import DNS, DNSQR, Ether, ICMP, IP, IPv6, IPv6ExtHdrHopByHop, TCP, UDP, Raw, PcapNgWriter, wrpcap

from src.data.parse_pcap import ParseStats, iter_packets
from src.utils.config import PROTO_DNS, PROTO_OTHER, PROTO_TCP, PROTO_UDP, ParseConfig
from src.utils.synthetic import _truncate_records, tcp_session, write_pcap


def parse(path, cfg=None):
    stats = ParseStats()
    pkts = list(iter_packets(path, cfg or ParseConfig(), stats))
    return pkts, stats


def stamp(pkt, t):
    pkt.time = t
    return pkt


def test_tcp_fields_and_payload(tmp_path):
    p = stamp(Ether() / IP(src="1.1.1.1", dst="2.2.2.2") / TCP(sport=1234, dport=80, flags="PA", seq=77, ack=5)
              / Raw(b"hello"), 1700000000.25)
    f = write_pcap(tmp_path / "a.pcap", [p])
    pkts, st = parse(f)
    assert len(pkts) == 1
    k = pkts[0]
    assert (k.src_ip, k.dst_ip, k.sport, k.dport, k.ip_proto) == ("1.1.1.1", "2.2.2.2", 1234, 80, 6)
    assert k.category == PROTO_TCP and k.seq == 77 and k.ack == 5
    assert k.tcp_flags == 0x18  # PSH|ACK
    assert k.payload_len == 5 and k.payload == b"hello"
    assert k.ts == pytest.approx(1700000000.25, abs=1e-5)
    assert st.parsed == 1 and st.tcp == 1 and st.ipv4 == 1


def test_ethernet_trailer_padding_is_not_payload(tmp_path):
    # An empty ACK is 54 bytes; Ethernet pads frames to 60. Padding must not count as payload.
    p = stamp(Ether() / IP(src="1.1.1.1", dst="2.2.2.2") / TCP(sport=1, dport=2, flags="A"), 1.0e9 + 1)
    raw = bytes(p) + b"\x00" * 6
    q = Ether(raw)
    q.time = p.time
    f = write_pcap(tmp_path / "pad.pcap", [q])
    pkts, _ = parse(f)
    assert pkts[0].payload_len == 0 and pkts[0].payload == b""


def test_payload_truncated_to_budget_but_len_is_true(tmp_path):
    p = stamp(Ether() / IP(src="1.1.1.1", dst="2.2.2.2") / TCP(sport=1, dport=2, flags="PA") / Raw(b"A" * 500), 1.0e9)
    f = write_pcap(tmp_path / "big.pcap", [p])
    pkts, _ = parse(f, ParseConfig(payload_bytes_kept=128))
    assert pkts[0].payload_len == 500 and len(pkts[0].payload) == 128
    pkts, _ = parse(f, ParseConfig(payload_bytes_kept=0))
    assert pkts[0].payload_len == 500 and pkts[0].payload == b""


def test_snaplen_truncation_keeps_true_payload_len_and_flags(tmp_path):
    p = stamp(Ether() / IP(src="1.1.1.1", dst="2.2.2.2") / TCP(sport=1, dport=2, flags="PA") / Raw(b"B" * 400), 1.0e9)
    f = write_pcap(tmp_path / "snap.pcap", [p], snaplen_cut=100)
    pkts, st = parse(f)
    k = pkts[0]
    assert k.payload_len == 400                   # declared, not captured
    assert len(k.payload) == 100 - 14 - 20 - 20   # only what survived the cut
    assert "snaplen_truncated" in k.warnings and st.truncated_by_snaplen == 1
    assert "snaplen_truncation_present" in st.warning_tags()


def test_udp_dns_icmp_categories(tmp_path):
    pk = [
        stamp(Ether() / IP(src="1.1.1.1", dst="2.2.2.2") / UDP(sport=5000, dport=6000) / Raw(b"x" * 10), 1.0e9),
        stamp(Ether() / IP(src="1.1.1.1", dst="8.8.8.8") / UDP(sport=5001, dport=53) / DNS(rd=1, qd=DNSQR(qname="a.com")), 1.0e9 + 1),
        stamp(Ether() / IP(src="1.1.1.1", dst="2.2.2.2") / ICMP() / Raw(b"y" * 8), 1.0e9 + 2),
    ]
    f = write_pcap(tmp_path / "m.pcap", pk)
    pkts, st = parse(f)
    assert [k.category for k in pkts] == [PROTO_UDP, PROTO_DNS, PROTO_OTHER]
    assert pkts[2].sport == 0 and pkts[2].dport == 0 and pkts[2].ip_proto == 1
    assert pkts[0].payload_len == 10
    assert (st.udp, st.dns, st.other) == (1, 1, 1)


def test_ipv6_with_extension_header(tmp_path):
    p = stamp(Ether() / IPv6(src="2001:db8::1", dst="2001:db8::2") / IPv6ExtHdrHopByHop()
              / TCP(sport=1000, dport=443, flags="PA", seq=9) / Raw(b"z" * 30), 1.0e9)
    f = write_pcap(tmp_path / "v6.pcap", [p])
    pkts, st = parse(f)
    assert len(pkts) == 1 and pkts[0].ip_version == 6
    assert pkts[0].sport == 1000 and pkts[0].dport == 443 and pkts[0].payload_len == 30
    assert st.ipv6 == 1


def test_non_first_fragment_skipped_first_fragment_flagged(tmp_path):
    first = stamp(Ether() / IP(src="1.1.1.1", dst="2.2.2.2", flags="MF", frag=0, id=7) / UDP(sport=1, dport=2) / Raw(b"a" * 40), 1.0e9)
    second = stamp(Ether() / IP(src="1.1.1.1", dst="2.2.2.2", flags=0, frag=5, id=7, proto=17) / Raw(b"b" * 40), 1.0e9 + 1)
    f = write_pcap(tmp_path / "frag.pcap", [first, second])
    pkts, st = parse(f)
    assert len(pkts) == 1 and "ip_fragment_first" in pkts[0].warnings
    assert st.ip_fragment_nonfirst_skipped == 1 and st.ip_fragment_first == 1
    assert "ip_fragments_skipped" in st.warning_tags()


def test_non_ip_and_malformed_counted_not_dropped_silently(tmp_path):
    arp = stamp(Ether(type=0x0806) / Raw(b"\x00" * 28), 1.0e9)
    short_tcp = stamp(Ether() / IP(src="1.1.1.1", dst="2.2.2.2", proto=6) / Raw(b"\x00" * 6), 1.0e9 + 1)
    good = stamp(Ether() / IP(src="1.1.1.1", dst="2.2.2.2") / UDP(sport=1, dport=2), 1.0e9 + 2)
    f = write_pcap(tmp_path / "x.pcap", [arp, short_tcp, good])
    pkts, st = parse(f)
    assert len(pkts) == 1
    assert st.non_ip == 1 and st.malformed == 1 and st.total_records == 3


def test_pcapng_is_read(tmp_path):
    p = stamp(Ether() / IP(src="1.1.1.1", dst="2.2.2.2") / TCP(sport=1, dport=2, flags="S"), 1700000123.5)
    path = tmp_path / "a.pcapng"
    w = PcapNgWriter(str(path))
    w.write(p)
    w.close()
    pkts, st = parse(path)
    assert len(pkts) == 1 and st.file_format == "pcapng"
    assert pkts[0].ts == pytest.approx(1700000123.5, abs=1e-4)


def test_truncated_file_is_reported_and_partial_results_kept(tmp_path):
    pk = tcp_session(1.7e9, 40000, n_exchanges=3)
    f = write_pcap(tmp_path / "t.pcap", pk)
    data = f.read_bytes()
    f.write_bytes(data[: len(data) - 30])  # chop the last record
    pkts, st = parse(f)
    assert 0 < len(pkts) < len(pk)
    assert st.total_records >= len(pkts)


def test_garbage_file_reports_open_error(tmp_path):
    f = tmp_path / "junk.pcap"
    f.write_bytes(b"this is not a pcap file at all" * 3)
    pkts, st = parse(f)
    assert pkts == [] and st.open_error
    assert "open_error" in st.warning_tags()


def test_bad_and_non_monotonic_timestamps_counted(tmp_path):
    a = stamp(Ether() / IP(src="1.1.1.1", dst="2.2.2.2") / UDP(sport=1, dport=2), 1.0e9 + 10)
    b = stamp(Ether() / IP(src="1.1.1.1", dst="2.2.2.2") / UDP(sport=1, dport=2), 1.0e9 + 5)
    f = write_pcap(tmp_path / "ts.pcap", [a, b])
    pkts, st = parse(f)
    assert st.non_monotonic_timestamp == 1 and len(pkts) == 2


def test_packet_limit(tmp_path):
    f = write_pcap(tmp_path / "l.pcap", tcp_session(1.7e9, 40000, n_exchanges=5))
    pkts, st = parse(f, ParseConfig(max_packets_per_capture=4))
    assert len(pkts) == 4 and st.hit_packet_limit
