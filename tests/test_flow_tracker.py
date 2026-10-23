from src.data.flow_tracker import FlowTracker, make_key
from src.data.parse_pcap import TCP_ACK, TCP_FIN, TCP_RST, TCP_SYN
from src.utils.config import FlowConfig, PROTO_TCP
from tests.conftest import mk

C, S = "10.0.0.2", "9.9.9.9"
SYN, SYNACK, ACK, PSHACK, FINACK = TCP_SYN, TCP_SYN | TCP_ACK, TCP_ACK, TCP_ACK | 0x08, TCP_FIN | TCP_ACK


def handshake(t, sport=40000, dport=443, isn_c=100, isn_s=900):
    return [mk(t, C, sport, S, dport, flags=SYN, seq=isn_c),
            mk(t + .01, S, dport, C, sport, flags=SYNACK, seq=isn_s, ack=isn_c + 1),
            mk(t + .02, C, sport, S, dport, flags=ACK, seq=isn_c + 1, ack=isn_s + 1)]


def run(pkts, **kw):
    return FlowTracker(FlowConfig(**kw)).run(pkts)


def test_key_is_direction_independent():
    a = mk(0, C, 1, S, 2)
    b = mk(0, S, 2, C, 1)
    assert make_key(a) == make_key(b)


def test_direction_relative_to_initiator():
    pk = handshake(0) + [mk(.1, C, 40000, S, 443, flags=PSHACK, seq=101, plen=10),
                         mk(.2, S, 443, C, 40000, flags=PSHACK, seq=901, plen=20)]
    (f,) = run(pk)
    assert f.initiator == (C, 40000) and f.responder == (S, 443)
    assert [p.direction for p in f.packets] == [1, -1, 1, 1, -1]
    assert "midstream_start" not in f.warnings


def test_synack_first_makes_receiver_the_initiator():
    pk = [mk(0, S, 443, C, 40000, flags=SYNACK, seq=900, ack=101), mk(.01, C, 40000, S, 443, flags=ACK, seq=101, ack=901)]
    (f,) = run(pk)
    assert f.initiator == (C, 40000)
    assert [p.direction for p in f.packets] == [-1, 1]


def test_midstream_tcp_flagged_first_sender_is_initiator():
    pk = [mk(0, C, 40000, S, 443, flags=PSHACK, seq=5000, plen=10), mk(.1, S, 443, C, 40000, flags=ACK, seq=1, ack=5010)]
    (f,) = run(pk)
    assert "midstream_start" in f.warnings and f.initiator == (C, 40000)


def test_udp_and_tcp_with_same_ports_are_separate_flows():
    pk = [mk(0, C, 5000, S, 5000, proto=6, flags=PSHACK, seq=1, plen=5),
          mk(.1, C, 5000, S, 5000, proto=17, plen=5),
          mk(.2, S, 5000, C, 5000, proto=17, plen=5)]
    flows = run(pk)
    assert len(flows) == 2
    assert sorted(f.ip_proto for f in flows) == [6, 17]
    assert sorted(len(f.packets) for f in flows) == [1, 2]


def test_different_ports_same_hosts_are_separate_flows():
    pk = handshake(0, sport=40000) + handshake(0.5, sport=40001)
    assert len(run(pk)) == 2


def test_5tuple_reuse_after_close_creates_new_flow():
    first = handshake(0) + [mk(.1, C, 40000, S, 443, flags=FINACK, seq=101, ack=901),
                            mk(.11, S, 443, C, 40000, flags=FINACK, seq=901, ack=102),
                            mk(.12, C, 40000, S, 443, flags=ACK, seq=102, ack=902)]
    second = handshake(10.0, isn_c=7777, isn_s=3333)  # same 5-tuple, new ISN, long after close
    flows = run(first + second)
    assert len(flows) == 2
    assert len(flows[0].packets) == 6 and len(flows[1].packets) == 3
    assert "restarted_same_5tuple" in flows[1].warnings


def test_new_syn_with_different_isn_splits_even_without_close():
    flows = run(handshake(0, isn_c=100) + handshake(1.0, isn_c=99999))
    assert len(flows) == 2 and "ended_by_new_syn_without_close" in flows[0].warnings


def test_syn_retransmission_stays_in_same_flow():
    pk = [mk(0, C, 40000, S, 443, flags=SYN, seq=100), mk(1.0, C, 40000, S, 443, flags=SYN, seq=100)] + \
         [mk(1.1, S, 443, C, 40000, flags=SYNACK, seq=900, ack=101)]
    assert len(run(pk)) == 1


def test_late_packets_inside_time_wait_join_old_flow():
    pk = handshake(0) + [mk(.1, C, 40000, S, 443, flags=FINACK, seq=101, ack=901),
                         mk(.11, S, 443, C, 40000, flags=FINACK, seq=901, ack=102),
                         mk(.12, C, 40000, S, 443, flags=ACK, seq=102, ack=902),
                         mk(1.0, S, 443, C, 40000, flags=ACK, seq=902, ack=102)]  # 0.89 s after close < 5 s
    flows = run(pk)
    assert len(flows) == 1 and len(flows[0].packets) == 7


def test_rst_closes_flow_and_later_traffic_after_timewait_is_new_flow():
    pk = handshake(0) + [mk(.1, S, 443, C, 40000, flags=TCP_RST, seq=901)] + \
         [mk(20.0, C, 40000, S, 443, flags=PSHACK, seq=500, plen=5)]
    flows = run(pk)
    assert len(flows) == 2 and flows[0].closed


def test_tcp_idle_timeout_splits_flow():
    pk = handshake(0) + [mk(200.0, C, 40000, S, 443, flags=PSHACK, seq=101, plen=5)]
    flows = run(pk, tcp_idle_timeout_s=120.0)
    assert len(flows) == 2 and "ended_by_idle_timeout" in flows[0].warnings
    assert "midstream_start" in flows[1].warnings


def test_udp_idle_timeout_is_separate_from_tcp():
    pk = [mk(0, C, 5000, S, 6000, proto=17, plen=5), mk(30.0, C, 5000, S, 6000, proto=17, plen=5)]
    assert len(run(pk, udp_idle_timeout_s=60.0)) == 1
    assert len(run(pk, udp_idle_timeout_s=20.0)) == 2


def test_sweep_expires_idle_flows_but_results_are_identical():
    # an unrelated busy flow advances the clock so the sweeper retires the idle one early
    pk = [mk(0, C, 5000, S, 6000, proto=17, plen=5)]
    pk += [mk(t, C, 7000, S, 8000, proto=17, plen=5) for t in range(1, 200, 5)]
    flows = run(pk, udp_idle_timeout_s=30.0, expire_check_interval_s=1.0)
    assert len(flows) == 2
    assert [f.flow_index for f in flows] == sorted(f.flow_index for f in flows)


def test_max_duration_splits_and_keeps_initiator():
    pk = handshake(0) + [mk(t, C, 40000, S, 443, flags=PSHACK, seq=101 + 10 * i, plen=10)
                         for i, t in enumerate([50, 100, 150, 200, 250])] + \
         [mk(t, S, 443, C, 40000, flags=ACK, seq=901, ack=1) for t in (260, 270)]
    flows = run(pk, max_flow_duration_s=120.0, tcp_idle_timeout_s=1000.0)
    assert len(flows) >= 2
    assert all(f.initiator == (C, 40000) for f in flows)
    assert "continued_after_max_duration" in flows[1].warnings and "split_by_max_duration" in flows[0].warnings


def test_retransmission_dropped_and_counted():
    data = lambda t, seq: mk(t, C, 40000, S, 443, flags=PSHACK, seq=seq, plen=10)
    pk = handshake(0) + [data(.1, 101), data(.2, 111), data(.3, 101)]  # last one is a retransmission
    (f,) = run(pk)
    assert f.retransmissions == 1 and len(f.packets) == 5
    (g,) = run(pk, drop_tcp_retransmissions=False)
    assert len(g.packets) == 6 and any("retransmission" in p.warnings for p in g.packets)


def test_retransmission_check_handles_sequence_wraparound():
    near = 0xFFFFFFF0
    pk = [mk(0, C, 1, S, 2, flags=PSHACK, seq=near, plen=0x20),            # ends at 0x10 (wrapped)
          mk(.1, C, 1, S, 2, flags=PSHACK, seq=0x10, plen=10),             # new data after wrap
          mk(.2, C, 1, S, 2, flags=PSHACK, seq=near, plen=0x20)]           # retransmission
    (f,) = run(pk)
    assert f.retransmissions == 1 and len(f.packets) == 2


def test_pure_acks_never_flagged_as_retransmissions():
    pk = handshake(0) + [mk(.1 + i * .01, C, 40000, S, 443, flags=ACK, seq=101, plen=0) for i in range(5)]
    (f,) = run(pk)
    assert f.retransmissions == 0 and len(f.packets) == 8


def test_packet_cap_is_reported():
    pk = [mk(i * .01, C, 5000, S, 6000, proto=17, plen=5) for i in range(30)]
    (f,) = run(pk, max_packets_per_flow=10)
    assert len(f.packets) == 10 and f.packets_dropped_cap == 20 and f.total_packets_seen == 30
    assert "packet_cap_reached" in f.warnings


def test_incomplete_tcp_flow_flagged_at_capture_end_udp_not():
    tcp = handshake(0)
    udp = [mk(0, C, 5000, S, 6000, proto=17, plen=5)]
    flows = run(tcp + udp)
    by_proto = {f.ip_proto: f for f in flows}
    assert "incomplete_at_capture_end" in by_proto[6].warnings
    assert "incomplete_at_capture_end" not in by_proto[17].warnings
