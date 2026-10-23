import numpy as np
import pytest

from src.data.build_sequences import build_examples_for_flows, window_starts
from src.data.feature_extraction import (AGG_FEATURE_NAMES, aggregate_features, empty_window, encode_window,
                                         prefix_view)
from src.data.flow_tracker import FlowTracker, PacketRecord
from src.utils.config import (META_CHANNELS, PROTO_DNS, PROTO_TCP, PROTO_UDP, PipelineConfig, load_config)
from tests.conftest import mk

CH = {n: i for i, n in enumerate(META_CHANNELS)}


def rec(ts, direction=1, plen=0, payload=b"", flags=0, cat=PROTO_UDP):
    return PacketRecord(ts=ts, direction=direction, category=cat, payload_len=plen, payload=payload, tcp_flags=flags)


def test_window_shapes_mask_and_padding_are_explicit():
    pk = [rec(0.0, 1, 3, b"\x01\x02\x03"), rec(0.5, -1, 2, b"\x00\x00")]
    w = encode_window(pk, 32, 128)
    assert w["meta"].shape == (32, len(META_CHANNELS)) and w["payload"].shape == (32, 128)
    assert w["mask"].tolist() == [1, 1] + [0] * 30
    assert np.abs(w["meta"][2:]).sum() == 0 and w["payload"][2:].sum() == 0
    # a REAL zero-byte payload is distinguishable from padding via payload_valid_len
    assert w["payload_valid_len"][1] == 2 and w["payload"][1, :2].tolist() == [0, 0]
    assert w["payload_valid_len"][2] == 0


def test_payload_truncation_to_B():
    pk = [rec(0.0, 1, 500, bytes(range(1, 129)))]  # parser already cut to 128
    w = encode_window(pk, 16, 64)
    assert w["payload"].shape == (16, 64) and w["payload_valid_len"][0] == 64
    assert w["payload"][0].tolist() == list(range(1, 65))


def test_channels_direction_iat_and_flags():
    pk = [rec(10.0, 1, 0, flags=0x02, cat=PROTO_TCP), rec(10.1, -1, 0, flags=0x12, cat=PROTO_TCP),
          rec(10.1, 1, 7, flags=0x18 | 0x01, cat=PROTO_TCP)]
    w = encode_window(pk, 16, 8)["meta"]
    assert w[:, CH["direction"]][:3].tolist() == [1, -1, 1]
    assert w[0, CH["log_iat"]] == 0
    assert w[1, CH["log_iat"]] == pytest.approx(np.log1p(100.0), rel=1e-5)
    assert w[2, CH["log_iat"]] == 0
    assert (w[0, CH["flag_syn"]], w[0, CH["flag_ack"]]) == (1, 0)
    assert (w[1, CH["flag_syn"]], w[1, CH["flag_ack"]]) == (1, 1)
    assert (w[2, CH["flag_psh"]], w[2, CH["flag_fin"]], w[2, CH["flag_ack"]]) == (1, 1, 1)
    assert w[:3, CH["proto_tcp"]].tolist() == [1, 1, 1] and w[:3, CH["proto_udp"]].sum() == 0


def test_negative_iat_is_clamped():
    w = encode_window([rec(5.0), rec(4.0)], 16, 8)
    assert w["meta"][1, CH["log_iat"]] == 0 and w["rel_time"][1] == 0


def test_include_payload_false_omits_payload_array():
    w = encode_window([rec(0.0, 1, 3, b"abc")], 16, 8, include_payload=False)
    assert "payload" not in w and set(w) == {"meta", "mask", "payload_valid_len", "rel_time"}


def test_prefix_view_equals_encoding_only_the_first_k_packets():
    pk = [rec(i * 0.1, 1 if i % 2 == 0 else -1, 10 + i, bytes([i + 1]) * 5) for i in range(20)]
    full = encode_window(pk, 32, 16)
    for k in (1, 4, 8, 16):
        pre = prefix_view(full, k)
        direct = encode_window(pk[:k], 32, 16)
        for name in direct:
            assert np.array_equal(pre[name], direct[name]), (k, name)


def test_encode_empty_raises():
    with pytest.raises(ValueError):
        encode_window([], 32, 8)


@pytest.mark.parametrize("n,W,S,expected", [
    (5, 32, 32, [0]), (32, 32, 32, [0]), (33, 32, 32, [0]),          # tail of 1 packet (<8) dropped
    (40, 32, 32, [0, 32]), (64, 32, 32, [0, 32]), (100, 32, 32, [0, 32, 64]),
    (40, 32, 16, [0, 16]), (80, 32, 16, [0, 16, 32, 48]),
])
def test_window_starts(n, W, S, expected):
    cfg = PipelineConfig()
    cfg.sequence.window_size, cfg.sequence.stride = W, S
    starts, _, _ = window_starts(n, cfg)
    assert starts == expected


def test_window_cap_and_tail_counters():
    cfg = PipelineConfig()
    cfg.sequence.max_windows_per_flow = 2
    starts, tail, cap = window_starts(200, cfg)
    assert starts == [0, 32] and cap == 1
    starts, tail, cap = window_starts(35, cfg)
    assert starts == [0] and tail == 1


def _udp_flow(n):
    return FlowTracker(PipelineConfig().flow).run(
        [mk(i * .01, "1.1.1.1", 5, "2.2.2.2", 6, proto=17, plen=10, payload=b"\x07" * 10) for i in range(n)])


def test_long_flow_yields_multiple_windows_same_flow_id_and_ordering():
    cfg = PipelineConfig()
    counters = {}
    ex, rows = build_examples_for_flows(_udp_flow(75), cfg, "cap", counters)
    assert len(ex) == 3 and [r["window_start_packet"] for r in rows] == [0, 32, 64]
    assert [r["n_packets"] for r in rows] == [32, 32, 11]
    assert len({r["flow_id"] for r in rows}) == 1 and len({r["example_id"] for r in rows}) == 3
    assert counters["windows_dropped_tail"] == 0
    # 70 packets: the trailing 6 (< min_tail_packets=8) are NOT turned into a near-empty example,
    # and the drop is counted rather than silent.
    counters = {}
    ex, rows = build_examples_for_flows(_udp_flow(70), cfg, "cap", counters)
    assert len(rows) == 2 and counters["windows_dropped_tail"] == 1


def test_short_flow_is_padded_and_tiny_flow_skipped():
    cfg = PipelineConfig()
    c = {}
    ex, rows = build_examples_for_flows(_udp_flow(10), cfg, "cap", c)
    assert len(ex) == 1 and rows[0]["n_packets"] == 10 and "padded_short_window" in rows[0]["warnings"]
    c = {}
    ex, rows = build_examples_for_flows(_udp_flow(1), cfg, "cap", c)
    assert ex == [] and c["flows_skipped_short"] == 1


def test_aggregate_features_shape_and_values():
    pk = [rec(0.0, 1, 100), rec(0.1, -1, 300), rec(0.2, 1, 0)]
    w = encode_window(pk, 16, 8)
    f = aggregate_features(w["meta"], w["mask"])
    d = dict(zip(AGG_FEATURE_NAMES, f[0]))
    assert f.shape == (1, len(AGG_FEATURE_NAMES))
    assert d["n_packets"] == 3 and d["n_fwd"] == 2 and d["n_rev"] == 1
    assert d["sum_payload_fwd"] == 100 and d["sum_payload_rev"] == 300 and d["max_payload"] == 300
    assert d["frac_zero_payload"] == pytest.approx(1 / 3) and d["frac_udp"] == 1.0
    # padding must not influence aggregates
    w2 = encode_window(pk, 64, 8)
    f2 = aggregate_features(w2["meta"], w2["mask"])
    assert np.allclose(f, f2)


def test_config_validation_and_unknown_keys(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("sequence: {window_size: 20}\n")
    with pytest.raises(ValueError):
        load_config(p)
    p.write_text("sequence: {window_sizee: 32}\n")
    with pytest.raises(ValueError):
        load_config(p)
    p.write_text("sequence: {window_size: 16, stride: 8}\n")
    cfg = load_config(p, {"sequence.window_size": 64})
    assert cfg.sequence.window_size == 64 and cfg.sequence.stride == 8


def test_out_of_order_timestamps_use_monotone_window_clock():
    pk = [rec(0.0), rec(1.0), rec(0.5), rec(2.0), rec(1.9)]
    w = encode_window(pk, 16, 8)
    assert w["rel_time"][:5].tolist() == [0.0, 1.0, 1.0, 2.0, 2.0]
    iat_ms = np.expm1(w["meta"][:5, CH["log_iat"]])
    assert np.allclose(iat_ms, [0, 1000, 0, 1000, 0], atol=1e-3)
    assert np.isclose(iat_ms.sum() / 1000.0, w["rel_time"][4])          # sum(iat) == rel_time
    assert (np.diff(w["rel_time"][:5]) >= 0).all()
    # first packet later than its successors: clock simply stays at 0 until time catches up
    w2 = encode_window([rec(5.0), rec(4.0), rec(6.0)], 16, 8)
    assert w2["rel_time"][:3].tolist() == [0.0, 0.0, 1.0]


def test_prefix_equivalence_holds_with_out_of_order_timestamps():
    ts = [0.0, 0.3, 0.2, 0.9, 0.5, 1.4, 1.3, 2.0]
    pk = [rec(t, 1 if i % 2 == 0 else -1, 10 + i, bytes([i + 1])) for i, t in enumerate(ts)]
    full = encode_window(pk, 16, 4)
    for k in (2, 4, 6):
        pre = prefix_view(full, k)
        direct = encode_window(pk[:k], 16, 4)
        for name in direct:
            assert np.array_equal(pre[name], direct[name]), (k, name)


def test_count_backward_steps():
    from src.data.feature_extraction import count_backward_steps
    assert count_backward_steps([rec(0), rec(1), rec(2)]) == 0
    assert count_backward_steps([rec(0), rec(2), rec(1), rec(3), rec(2.5)]) == 2


def test_sampling_cap_is_deterministic_and_order_preserving():
    cfg = PipelineConfig()
    cfg.sequence.max_examples_per_capture = 5
    flows = FlowTracker(cfg.flow).run(
        [mk(i * .01 + f * 100, "1.1.1.1", 5 + f, "2.2.2.2", 6, proto=17, plen=10) for f in range(20) for i in range(10)])
    c1, c2, c3 = {}, {}, {}
    _, r1 = build_examples_for_flows(flows, cfg, "capA", c1)
    _, r2 = build_examples_for_flows(flows, cfg, "capA", c2)
    _, r3 = build_examples_for_flows(flows, cfg, "capB", c3)
    assert len(r1) == 5 and c1["windows_dropped_sampling"] == 15
    assert [r["example_id"] for r in r1] == [r["example_id"] for r in r2]      # deterministic
    assert [r["example_id"].split(":", 1)[1] for r in r1] != [r["example_id"].split(":", 1)[1] for r in r3]
    idx = [int(r["flow_id"].split(":f")[1]) for r in r1]
    assert idx == sorted(idx)                                                  # original order kept
    cfg.sequence.max_examples_per_capture = 0
    _, rall = build_examples_for_flows(flows, cfg, "capA", {})
    assert len(rall) == 20


def test_flow_packet_cap_must_cover_window_plan():
    cfg = PipelineConfig()
    cfg.validate()                                   # defaults: 19*32+32 = 640 <= 1500
    cfg.sequence.window_size = cfg.sequence.stride = 64
    cfg.validate()                                   # 19*64+64 = 1280 <= 1500
    cfg.flow.max_packets_per_flow = 1000
    with pytest.raises(ValueError, match="max_packets_per_flow"):
        cfg.validate()
