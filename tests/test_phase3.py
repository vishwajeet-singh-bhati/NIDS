import json
from pathlib import Path

import pandas as pd
import pytest

from src.data import build_sequences, label_audit
from src.data.dataset_loader import load_manifest
from src.data.flow_tags import add_flow_tags, ip_scope, service_hint

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("ip,scope", [
    ("10.42.0.1", "private"), ("192.168.0.9", "private"), ("172.16.5.5", "private"), ("100.64.1.1", "private"),
    ("224.0.0.251", "multicast"), ("ff02:0:0:0:0:0:0:fb", "multicast"), ("fe80:0:0:0:1:2:3:4", "link_local"),
    ("169.254.1.1", "link_local"), ("127.0.0.1", "loopback"), ("8.8.8.8", "public"),
    ("142.250.80.46", "public"), ("2001:4860:4860:0:0:0:0:8888", "public"), ("not-an-ip", "unparsable"),
])
def test_ip_scope(ip, scope):
    assert ip_scope(ip) == scope


def test_service_hint_and_flow_tag():
    assert service_hint(17, 53) == "dns" and service_hint(6, 443) == "https" and service_hint(17, 443) == "quic"
    assert service_hint(58, 0) == "icmp" and service_hint(6, 12345) == "other"
    df = pd.DataFrame({"responder_ip": ["10.42.0.1", "8.8.8.8", "142.250.1.1", "224.0.0.251"],
                       "responder_port": [53, 53, 443, 5353], "ip_proto": [17, 17, 6, 17]})
    t = add_flow_tags(df)
    assert t["flow_tag"].tolist() == ["local_infra", "dns", "https", "local_infra"]
    assert t["responder_scope"].tolist() == ["private", "public", "public", "multicast"]
    assert "flow_tag" not in df.columns                      # input is not mutated


def test_label_audit_on_andmal_like_synthetic(andmal_like_manifest, tmp_path):
    man_path = andmal_like_manifest["manifest"]
    out = tmp_path / "audit"
    assert label_audit.main(["--manifest", str(man_path), "--out", str(out)]) == 0
    rep = json.loads(out.with_suffix(".json").read_text())
    assert set(rep["classes"]) == {"benign", "malicious_associated"}
    b, m = rep["classes"]["benign"], rep["classes"]["malicious_associated"]
    assert b["captures"] == 40 and m["captures"] == 32
    assert 0.0 < b["background_share_examples"] < 1.0 and 0.0 < m["background_share_examples"] < 1.0
    assert b["apps"] == 20                                    # two captures per package
    assert len(rep["by_family"]) == 8 and rep["confounds"]["capture_dates_disjoint"] is True
    assert any("TEMPORAL CONFOUND" in w for w in rep["warnings"])
    assert rep["responder_overlap"]["shared_ips"] == 0
    assert "Label audit" in out.with_suffix(".md").read_text()


def test_label_audit_rejects_old_manifest(tmp_path):
    p = tmp_path / "old.csv"
    pd.DataFrame({"example_id": ["a"], "label": ["benign"]}).to_csv(p, index=False)
    assert label_audit.main(["--manifest", str(p), "--out", str(tmp_path / "x")]) == 2
