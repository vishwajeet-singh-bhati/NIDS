"""Streaming engine, policy, prefix augmentation, pretraining and replay evaluation (synthetic data only)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.normalization import Normalizer
from src.models.model_factory import build_model, save_checkpoint
from src.stream.engine import StreamConfig, StreamDetector
from src.stream.policy import PolicyParams, PolicyState, first_alert, step
from src.training.data import SplitData
from src.training.evaluate import predict
from tests.conftest import mk

SMALL = dict(d_model=32, heads=2, layers=2, ff_mult=2, dropout=0.0, use_payload=True, byte_channels=[8, 8],
             byte_embed_dim=4, head_hidden=16, qat=True, qat_start_epoch=1)


@pytest.fixture(scope="module")
def ckpt(consolidated_dataset, tmp_path_factory):
    d = tmp_path_factory.mktemp("ck")
    card = json.loads((Path(consolidated_dataset) / "dataset_card.json").read_text())
    norm = Normalizer.load(Path(consolidated_dataset) / "normalization.json")
    torch.manual_seed(0)
    m = build_model("packet_bert", SMALL)
    for p in m.parameters():                       # make logits non-trivial (untrained net is near-constant)
        if p.ndim > 1:
            torch.nn.init.normal_(p, std=0.3)
    p = d / "best.pt"
    save_checkpoint(p, m, "packet_bert", SMALL, card["feature_spec"], norm.spec, {"seed": 0}, 1, {},
                    {"dataset_dir": str(consolidated_dataset)})
    return p


# ------------------------------------------------------------------ policy
def test_policy_hysteresis_and_hard_margin():
    p = PolicyParams(threshold=0.0, ema_alpha=0.5, min_hits=2, hard_margin=3.0)
    assert first_alert([1.0, -1.0, 1.0, -1.0], p) == (None, None)          # never two in a row
    assert first_alert([1.0, 1.0], p)[0] == 1                               # two hits
    assert first_alert([-1.0, 3.5], p) == (1, "single_window_high")        # one very high window is enough
    st = PolicyState()
    assert step(st, 5.0, p) == "single_window_high"
    assert step(st, 5.0, p) is None                                         # one alert per flow


def test_policy_ema_blocks_after_negative_history():
    p = PolicyParams(threshold=0.0, ema_alpha=0.3, min_hits=2, hard_margin=10)
    assert first_alert([-8.0, 0.5, 0.5], p) == (None, None)                 # EMA still below threshold


# ------------------------------------------------------------------ prefix-aug batches
def test_per_row_prefix_masks(consolidated_dataset):
    norm = Normalizer.load(Path(consolidated_dataset) / "normalization.json")
    d = SplitData(str(consolidated_dataset), "train", norm)
    rows = np.arange(4)
    b = d.batch(rows, {"meta", "payload"}, np.array([2, 5, 8, 32]))
    n_real = np.asarray(d.mask_raw[rows]).sum(1)
    got = b["mask"].numpy().sum(1)
    assert (got == np.minimum([2, 5, 8, 32], n_real)).all()
    assert float(b["meta"][0, 2:].abs().sum()) == 0 and int((b["payload"][0, 2:] != 0).sum()) == 0


def test_prefix_aug_changes_training_batches(consolidated_dataset):
    norm = Normalizer.load(Path(consolidated_dataset) / "normalization.json")
    d = SplitData(str(consolidated_dataset), "train", norm)
    rng = np.random.default_rng(0)
    full = next(d.iter_train_batches(64, {"meta"}, rng, "none", 1, 0.0))["mask"].sum(1)
    rng = np.random.default_rng(0)
    aug = next(d.iter_train_batches(64, {"meta"}, rng, "none", 1, 1.0, (4,)))["mask"].sum(1)
    assert aug.max() <= 4 and full.max() > 4


# ------------------------------------------------------------------ engine
def _long_flow(n=40, t0=100.0):
    pk = [mk(t0, "10.0.0.2", 40000, "8.8.8.8", 443, flags=0x02, seq=100)]
    for i in range(1, n):
        fwd = i % 2 == 0
        pk.append(mk(t0 + 0.01 * i, "10.0.0.2" if fwd else "8.8.8.8", 40000 if fwd else 443,
                     "8.8.8.8" if fwd else "10.0.0.2", 443 if fwd else 40000, flags=0x18,
                     seq=101 + 50 * i, plen=50, payload=bytes([i % 256]) * 50))
    return pk


class _Const(torch.nn.Module):
    inputs = {"meta"}

    def __init__(self, v):
        super().__init__()
        self.v = v

    def forward(self, b):
        return torch.full((b["mask"].shape[0],), float(self.v))


def test_alert_raised_after_eighth_packet_not_after_full_window(ckpt):
    det = StreamDetector(str(ckpt), stream_cfg=StreamConfig(threshold=0.0, min_hits=1), config_path="configs/data.yaml")
    det.model = _Const(5.0)
    det.kinds = {"meta"}
    det.include_payload = False
    alerts = []
    for p in _long_flow():
        alerts += det.feed(p)
        alerts += det.run_inference()
        if alerts:
            break
    assert len(alerts) == 1
    a = alerts[0]
    assert a["decided_on_prefix_k"] == 8 and a["packets_seen"] == 8      # early: 8 of 32 packets
    assert a["reason"] == "single_window_high" and a["seconds_since_flow_start"] == pytest.approx(0.07, abs=1e-6)


def test_no_alert_below_threshold_and_scores_each_checkpoint_once(ckpt):
    det = StreamDetector(str(ckpt), stream_cfg=StreamConfig(threshold=1.0, min_hits=1), config_path="configs/data.yaml")
    det.model, det.kinds, det.include_payload = _Const(-5.0), {"meta"}, False
    det.record_scores = True
    for p in _long_flow(70):
        det.feed(p)
    det.finish()
    assert det.stats["alerts"] == 0
    ks = [(s["start"], s["k"]) for s in det.scores]
    assert len(ks) == len(set(ks))                                          # no double scoring
    assert (0, 8) in ks and (0, 16) in ks and (0, 32) in ks and (32, 8) in ks and (32, 32) in ks
    assert any(s["final"] for s in det.scores)                              # the 6-packet tail window... or final partial


def test_cooldown_suppresses_duplicate_alerts(ckpt):
    det = StreamDetector(str(ckpt), stream_cfg=StreamConfig(threshold=0.0, min_hits=1, cooldown_s=60), config_path="configs/data.yaml")
    det.model, det.kinds, det.include_payload = _Const(5.0), {"meta"}, False
    for sport in (40000, 40001):                                            # two flows, same destination
        for p in _long_flow(10):
            p.sport = sport if p.src_ip == "10.0.0.2" else p.sport
            p.dport = sport if p.dst_ip == "10.0.0.2" else p.dport
            det.feed(p)
        det.run_inference()
    assert det.stats["alerts"] == 1 and det.stats["alerts_suppressed_cooldown"] == 1


def test_engine_logits_match_offline_logits(ckpt, consolidated_dataset, andmal_like_manifest):
    """Train/serve parity. Synthetic flows are short, so: (a) the engine's final partial window must equal the
    offline full window of that flow, (b) the engine's 8-packet prefix must equal the offline model applied to
    the first 8 packets of the same window (`prefix_k=8`)."""
    ds = Path(consolidated_dataset)
    norm = Normalizer.load(ds / "normalization.json")
    from src.models.model_factory import load_checkpoint
    model, _ = load_checkpoint(ckpt)
    d = SplitData(str(ds), "test", norm)
    idx = d.sel_index.assign(full=predict(model, d, 256, "cpu"), pre8=predict(model, d, 256, "cpu", 8))
    man = pd.read_csv(andmal_like_manifest["manifest"], usecols=["example_id", "capture_path"])
    idx = idx.merge(man, on="example_id")
    det = StreamDetector(str(ckpt), stream_cfg=StreamConfig(checkpoints=(8, 32), final_min_packets=2, threshold=1e9),
                         config_path="configs/data.yaml")
    det.record_scores = True
    n_final = n_pre = 0
    for cp, g in list(idx.groupby("capture_path"))[:6]:
        det.reset_capture()
        det.scores = []
        det.feed_pcap(Path(andmal_like_manifest["raw"]) / cp)
        for s in det.scores:
            hit = g[g["example_id"] == f"{g['capture_id'].iloc[0]}:f{s['flow_index']}:w{s['start'] // 32}"]
            if not len(hit):
                continue
            if s["final"]:
                assert float(hit["full"].iloc[0]) == pytest.approx(s["logit"], abs=1e-4)
                n_final += 1
            elif s["k"] == 8:
                assert float(hit["pre8"].iloc[0]) == pytest.approx(s["logit"], abs=1e-4)
                n_pre += 1
    assert n_final > 0 and n_pre > 0


def test_feed_raw_decodes_ethernet_frames(ckpt):
    from scapy.all import IP, TCP, Ether
    det = StreamDetector(str(ckpt), config_path="configs/data.yaml")
    for i in range(5):
        fr = bytes(Ether() / IP(src="10.0.0.2", dst="1.2.3.4") / TCP(sport=5555, dport=443, flags="S" if i == 0 else "A", seq=i))
        det.feed_raw(fr, 1, 10.0 + i * 0.01)
    assert det.stats["packets_kept"] == 5 and det.stats["flows_seen"] == 1


def test_memory_guard_evicts_oldest(ckpt):
    det = StreamDetector(str(ckpt), stream_cfg=StreamConfig(max_active_flows=3, drain_every=5), config_path="configs/data.yaml")
    for i in range(10):
        det.feed(mk(1.0 + i, "10.0.0.2", 41000 + i, "9.9.9.9", 443, flags=0x02, seq=1))
    assert len(det.tracker._active) <= 6


# ------------------------------------------------------------------ pretraining + finetune with QAT
def test_pretrain_then_finetune_with_qat(consolidated_dataset, tmp_path):
    from src.training.pretrain import pretrain
    from src.training.train import train
    r = pretrain(str(consolidated_dataset), tmp_path / "pre", {"epochs": 2, "batch_size": 64, "max_steps_per_epoch": 5},
                 model_cfg=SMALL, device="cpu")
    assert Path(r["encoder"]).exists() and r["final_ratio_to_zero_baseline"] < 1.2
    out = train(str(consolidated_dataset), "packet_bert", tmp_path / "ft",
                {"epochs": 2, "batch_size": 64, "max_steps_per_epoch": 5, "prefix_aug": 0.5, "init_from": r["encoder"],
                 "patience": 5}, model_cfg=SMALL, device="cpu")
    ck = torch.load(tmp_path / "ft" / "best.pt", weights_only=False)
    assert int(ck["state_dict"]["qat_active_buf"]) == 1                     # QAT was switched on (epoch >= 1)
    m = json.loads((tmp_path / "ft" / "metrics.json").read_text())
    assert "test" in m["splits"] and out["best_epoch"] >= 1


def test_init_from_rejects_mismatched_encoder(consolidated_dataset, tmp_path):
    from src.training.pretrain import load_pretrained_encoder, pretrain
    r = pretrain(str(consolidated_dataset), tmp_path / "pre", {"epochs": 1, "batch_size": 64, "max_steps_per_epoch": 2},
                 model_cfg=SMALL, device="cpu")
    other = build_model("packet_bert", {**SMALL, "d_model": 48})
    with pytest.raises(ValueError, match="does not match"):
        load_pretrained_encoder(other, r["encoder"])


# ------------------------------------------------------------------ replay evaluation
def test_replay_eval_end_to_end(ckpt, consolidated_dataset, andmal_like_manifest, tmp_path):
    from src.stream import replay_eval
    run = tmp_path / "run"
    run.mkdir()
    (run / "best.pt").write_bytes(Path(ckpt).read_bytes())
    sp = Path(consolidated_dataset).parent / "splits" / "t.csv"
    res = replay_eval.run(str(run), andmal_like_manifest["manifest"], str(sp), str(andmal_like_manifest["raw"]),
                          str(run / "stream"), target_fpr=0.05, max_captures=8)
    assert (run / "stream" / "stream_report.md").exists()
    t = res["test"]
    for k in ("flow_level", "capture_level", "throughput", "flow_level_excluding_local_infra"):
        assert k in t
    assert 0.0 <= t["flow_level"]["accuracy"] <= 1.0
    # the val threshold is applied unchanged to test
    assert res["threshold_selection"]["threshold_logit"] == t["policy"]["threshold"] == res["val"]["policy"]["threshold"]
    assert res["val"]["flow_level"]["fpr"] <= 0.05 or "note" in res["threshold_selection"] and res["threshold_selection"]["note"]


def test_choose_threshold_respects_target_fpr():
    from src.stream.replay_eval import choose_threshold
    rng = np.random.default_rng(0)
    rows = []
    for i in range(200):
        y = int(i % 2 == 0)
        z = rng.normal(2.0 if y else -2.0, 1.5, size=3)
        rows.append({"y": y, "logits": z, "ts": np.arange(3) * 0.1, "k": np.array([8, 16, 32]), "npk": np.array([8, 16, 32]),
                     "flow_start_ts": 0.0})
    ch = choose_threshold(pd.DataFrame(rows), PolicyParams(0.0, 0.5, 2, 3.0), 0.05, "fpr")
    assert ch["val"]["fpr"] <= 0.05 and ch["val"]["recall"] > 0.5


# ------------------------------------------------------------------ device-level rule
def test_burst_rule_counts_alerts_inside_window_and_cools_down():
    from src.stream.device_rule import BurstParams, alarm_times
    p = BurstParams(window_s=10, min_alerts=3)
    assert alarm_times([0, 20, 40], p) == []                       # spread out: never 3 within 10 s
    assert alarm_times([0, 1, 2], p) == [2]
    assert alarm_times([0, 1, 2, 3, 4, 5, 6], p) == [2]            # cooldown (= window) suppresses repeats
    assert alarm_times([0, 1, 2, 13, 14, 15], p) == [2, 15]        # a second burst later alarms again
    assert alarm_times([5, 0, 1, 2], p) == [2]                     # input order does not matter


def test_engine_emits_device_alarm_after_burst(ckpt):
    det = StreamDetector(str(ckpt), stream_cfg=StreamConfig(threshold=0.0, min_hits=1, device_window_s=60, device_min_alerts=3,
                                                              cooldown_s=0), config_path="configs/data.yaml")
    det.model, det.kinds, det.include_payload = _Const(5.0), {"meta"}, False
    ev = []
    for i in range(4):                                             # four different flows, each alerting at its 8th packet
        for p in _long_flow(10, t0=100.0 + i):
            for q in (p,):
                q.sport = 41000 + i if q.src_ip == "10.0.0.2" else q.sport
                q.dport = 41000 + i if q.dst_ip == "10.0.0.2" else q.dport
            ev += det.feed(p)
        ev += det.run_inference()
    kinds = [e["event"] for e in ev]
    assert kinds.count("flow_alert") == 4 and kinds.count("device_alarm") == 1


def test_device_level_selects_on_val_and_applies_to_test():
    from src.stream.replay_eval import device_level
    rng = np.random.default_rng(1)

    def mk_split(prefix):
        rows, caps, cs = [], [], []
        for i in range(30):
            y = int(i % 3 == 0)
            cid = f"{prefix}{i}"
            caps.append({"capture_id": cid, "binary_label": y})
            cs.append({"capture_id": cid, "duration_s": 600.0})
            for j in range(40):
                hot = rng.random() < (0.6 if y else 0.02)
                rows.append({"capture_id": cid, "y": y, "logits": np.array([5.0 if hot else -5.0]), "ts": np.array([j * 5.0]),
                             "k": np.array([8]), "npk": np.array([8]), "flow_start_ts": j * 5.0})
        return {"flows": pd.DataFrame(rows), "caps": pd.DataFrame(caps), "captures": pd.DataFrame(cs)}
    R = {"val": mk_split("v"), "test": mk_split("t")}
    d = device_level(R, PolicyParams(0.0, 0.5, 1, 3.0), "test", target_fa=0.05)
    assert d["val"]["benign_capture_false_alarm_rate"] <= 0.05
    assert d["test"]["recall"] > 0.9 and d["test"]["benign_capture_false_alarm_rate"] <= 0.1
    assert d["selected"]["min_alerts"] >= 3


def test_payload_packetbert_pretrain_and_finetune(consolidated_dataset, tmp_path):
    from src.training.pretrain import pretrain
    from src.training.train import train
    cfg = {**SMALL}
    r = pretrain(str(consolidated_dataset), tmp_path / "p", {"epochs": 1, "batch_size": 64, "max_steps_per_epoch": 3},
                 model_cfg=cfg, device="cpu", model_name="packet_bert_pay")
    assert torch.load(r["encoder"], weights_only=False)["model_name"] == "packet_bert_pay"
    train(str(consolidated_dataset), "packet_bert_pay", tmp_path / "f", {"epochs": 1, "batch_size": 64, "max_steps_per_epoch": 3,
          "init_from": r["encoder"], "prefix_aug": 0.5}, model_cfg=cfg, device="cpu")
    assert (tmp_path / "f" / "best.pt").exists()
