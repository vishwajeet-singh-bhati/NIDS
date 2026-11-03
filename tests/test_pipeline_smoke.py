"""End-to-end smoke tests on SYNTHETIC captures (pipeline plumbing only, not detector validation)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import build_sequences, inspect_dataset, inspect_processed
from src.data.dataset_loader import load_arrays, load_manifest, mlp_features
from src.data.feature_extraction import AGG_FEATURE_NAMES
from src.utils.synthetic import (beacon_shaped_capture, benign_shaped_capture, tcp_session, udp_session, write_pcap)

LABELS = Path(__file__).resolve().parents[1] / "configs" / "labels.synthetic.yaml"
DATA_CFG = Path(__file__).resolve().parents[1] / "configs" / "data.yaml"


@pytest.fixture
def raw(tmp_path):
    root = tmp_path / "raw"
    write_pcap(root / "benign/app_a/c1.pcap", benign_shaped_capture(seed=1))
    write_pcap(root / "benign/app_b/c1.pcap", benign_shaped_capture(t0=1_700_050_000.0, seed=2))
    write_pcap(root / "malware_like/family_x/c1.pcap", beacon_shaped_capture(seed=1))
    return root


def run_build(raw, tmp_path, *extra, labels=LABELS, tag="out"):
    proc, man = tmp_path / f"proc_{tag}", tmp_path / f"{tag}_manifest.csv"
    rc = build_sequences.main(["--config", str(DATA_CFG), "--raw-root", str(raw), "--processed-root", str(proc),
                               "--manifest", str(man), "--label-config", str(labels), "--overwrite", *extra])
    return rc, proc, man


def test_end_to_end_manifest_tensors_and_invariants(raw, tmp_path):
    rc, proc, man = run_build(raw, tmp_path)
    assert rc == 0
    m = load_manifest(man)
    assert set(m["label"]) == {"benign", "malicious_associated"}
    assert set(m.loc[m["label"] == "benign", "binary_label"]) == {0}
    assert set(m.loc[m["label"] == "malicious_associated", "binary_label"]) == {1}
    assert m["example_id"].is_unique
    for col in ("source_dataset", "capture_id", "app_id", "malware_family", "original_label", "label",
                "label_source", "label_reliability", "flow_id", "n_packets", "warnings"):
        assert col in m.columns
    arr = load_arrays(m, proc)
    assert arr["meta"].shape[0] == len(m) and arr["meta"].shape[1] == 32 and arr["payload"].shape[2] == 128
    assert inspect_processed.check_invariants({k: v for k, v in arr.items() if k != "spec"}, m) == []
    feats = mlp_features(arr)
    assert feats.shape == (len(m), len(AGG_FEATURE_NAMES)) and np.isfinite(feats).all()
    assert (proc / "processing_log.jsonl").exists() or (man.parent / "processing_log.jsonl").exists()
    # label provenance: families only on malware-like rows, apps only on benign rows
    assert m.loc[m["label"] == "benign", "malware_family"].eq("").all()
    assert m.loc[m["label"] == "malicious_associated", "malware_family"].eq("family_x").all()


def test_deterministic_outputs(raw, tmp_path):
    _, p1, m1 = run_build(raw, tmp_path, tag="a")
    _, p2, m2 = run_build(raw, tmp_path, tag="b")
    d1, d2 = pd.read_csv(m1, keep_default_na=False), pd.read_csv(m2, keep_default_na=False)
    pd.testing.assert_frame_equal(d1.drop(columns=["shard_file"]), d2.drop(columns=["shard_file"]))
    a1, a2 = load_arrays(load_manifest(m1), p1), load_arrays(load_manifest(m2), p2)
    for k in ("meta", "mask", "payload", "payload_valid_len", "rel_time"):
        assert np.array_equal(a1[k], a2[k])


def test_no_rules_means_everything_unknown_and_nonzero_exit_free(raw, tmp_path):
    empty = tmp_path / "empty_labels.yaml"
    empty.write_text(
        "categories: [benign, malicious_associated, confirmed_attack, unknown]\n"
        "binary_target: {benign: 0, malicious_associated: 1, confirmed_attack: 1, unknown: -1}\n"
        "default_rule: {mapped_label: unknown}\nrules: []\n")
    rc, proc, man = run_build(raw, tmp_path, labels=empty, tag="unk")
    m = load_manifest(man)
    assert rc == 0 and set(m["label"]) == {"unknown"} and set(m["binary_label"]) == {-1}


def test_window_size_16_and_no_payload(raw, tmp_path):
    rc, proc, man = run_build(raw, tmp_path, "--window-size", "16", "--no-payload", tag="w16")
    assert rc == 0
    m = load_manifest(man)
    arr = load_arrays(m, proc)
    assert arr["meta"].shape[1] == 16 and "payload" not in arr and arr["spec"]["include_payload"] is False
    assert m["n_packets"].max() <= 16


def test_sliding_windows_share_flow_id(tmp_path):
    root = tmp_path / "raw"
    write_pcap(root / "benign/app_a/long.pcap", tcp_session(1.7e9, 40000, n_exchanges=40, gap=0.01))
    rc, proc, man = run_build(root, tmp_path, "--stride", "16", tag="slide")
    m = load_manifest(man)
    assert rc == 0 and m["flow_id"].nunique() == 1 and len(m) >= 3
    assert m["window_start_packet"].tolist() == list(range(0, 16 * len(m), 16))[:len(m)]


def test_dry_run_labels_processes_nothing(raw, tmp_path, capsys):
    rc, proc, man = run_build(raw, tmp_path, "--dry-run-labels", tag="dry")
    out = capsys.readouterr().out
    assert rc == 0 and "malicious_associated" in out and not man.exists()


def test_corrupt_capture_does_not_abort_batch_and_is_logged(raw, tmp_path):
    (raw / "benign/app_a/bad.pcap").write_bytes(b"not a capture" * 10)
    rc, proc, man = run_build(raw, tmp_path, tag="bad")
    assert rc == 1                       # non-zero: something failed
    m = load_manifest(man)               # ...but good captures were still processed
    assert len(m) > 0 and "bad.pcap" not in " ".join(m["capture_path"])
    log = [json.loads(l) for l in (man.parent / "processing_log.jsonl").read_text().splitlines()]
    assert any(r["status"] == "open_error" and "bad.pcap" in r["capture_path"] for r in log)


def test_missing_raw_root_returns_error(tmp_path):
    rc = build_sequences.main(["--config", str(DATA_CFG), "--raw-root", str(tmp_path / "nope")])
    assert rc == 2


def test_inspect_dataset_reports_problems(raw, tmp_path, capsys):
    (raw / "benign/app_a/bad.pcap").write_bytes(b"x" * 50)
    (raw / "benign/app_a/empty.pcap").write_bytes(b"")
    (raw / "meta.csv").write_text("flow_id,src_ip,dst_ip,src_port,label\n1,a,b,1,Benign\n2,a,b,2,Adware\n3,a,b,3,\n")
    (raw / "stuff.zip").write_bytes(b"PK")
    out = tmp_path / "audit"
    rc = inspect_dataset.main(["--root", str(raw), "--out", str(out)])
    rep = json.loads(out.with_suffix(".json").read_text())
    probs = " | ".join(p["problem"] for p in rep["problems"])
    assert rc == 0 and "zero_size_file" in probs and "unreadable" in probs and "archive_not_extracted" in probs
    csv = rep["csvs"][0]
    assert csv["missing_values"] == {"label": 1} and csv["candidate_label_columns"]["label"]
    assert csv["granularity_hint"].startswith("rows look flow-level")
    assert "benign" in rep["path_vocabulary"]["depth_0"]
    assert out.with_suffix(".md").read_text().startswith("# Dataset audit")


def test_out_of_order_capture_passes_invariants_and_is_recorded(tmp_path):
    import random
    pk = benign_shaped_capture(seed=3)
    rng = random.Random(0)
    for p in pk:                       # jitter timestamps by up to +/-30 ms without re-sorting the file
        p.time = float(p.time) + rng.uniform(-0.03, 0.03)
    root = tmp_path / "raw"
    write_pcap(root / "benign/app_a/jit.pcap", pk)
    rc, proc, man = run_build(root, tmp_path, tag="jit")
    assert rc == 0
    m = load_manifest(man)
    arr = load_arrays(m, proc)
    assert (m["n_ts_backward"] > 0).any()
    assert "ts_backward_in_window" in " ".join(m["warnings"])
    assert inspect_processed.check_invariants({k: v for k, v in arr.items() if k != "spec"}, m) == []
    assert arr["spec"]["feature_spec_version"] == "1.1" and arr["spec"]["time_model"] == "monotone_window_clock"
