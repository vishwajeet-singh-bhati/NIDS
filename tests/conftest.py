import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.parse_pcap import TCP_ACK, TCP_FIN, TCP_RST, TCP_SYN, ParsedPacket  # noqa: E402
from src.utils.config import (PROTO_DNS, PROTO_OTHER, PROTO_TCP, PROTO_UDP, FlowConfig,  # noqa: E402
                              ParseConfig, PipelineConfig)


def mk(ts, src, sport, dst, dport, proto=6, flags=0, seq=0, plen=0, payload=b"", ack=0, category=None):
    if category is None:
        category = PROTO_TCP if proto == 6 else PROTO_UDP if proto == 17 else PROTO_OTHER
    return ParsedPacket(ts=ts, ip_version=4, ip_proto=proto, src_ip=src, dst_ip=dst, sport=sport,
                        dport=dport, category=category, tcp_flags=flags, seq=seq, ack=ack,
                        payload_len=plen, payload=payload[:128], wirelen=60 + plen, caplen=60 + plen)


@pytest.fixture
def flow_cfg():
    return FlowConfig()


@pytest.fixture
def parse_cfg():
    return ParseConfig()


@pytest.fixture
def pipeline_cfg():
    cfg = PipelineConfig()
    cfg.validate()
    return cfg


@pytest.fixture(scope="session")
def andmal_like_manifest(tmp_path_factory):
    """Synthetic CIC-AndMal2017-LIKE dataset processed with the REAL configs/labels.yaml rules.
    40 benign captures (20 packages x 2), 8 families x 4 captures. Session-scoped (built once)."""
    from src.data import build_sequences
    from src.utils.synthetic import make_andmal_like
    root = Path(__file__).resolve().parents[1]
    base = tmp_path_factory.mktemp("andmal_like")
    raw, proc, man = base / "raw", base / "proc", base / "manifest.csv"
    make_andmal_like(raw, n_benign=40, n_families=8, per_family=4)
    rc = build_sequences.main(["--config", str(root / "configs/data.yaml"), "--label-config",
                               str(root / "configs/labels.yaml"), "--raw-root", str(raw),
                               "--processed-root", str(proc), "--manifest", str(man), "--overwrite"])
    assert rc == 0
    return {"raw": raw, "processed": proc, "manifest": man, "root": root}


@pytest.fixture(scope="session")
def consolidated_dataset(andmal_like_manifest, tmp_path_factory):
    """Splits + consolidates the synthetic AndMal-like manifest (group_random). Used by Phase 5/6 tests."""
    from src.data import consolidate, splits
    base = tmp_path_factory.mktemp("consolidated")
    sp_dir = base / "splits"
    rc = splits.main(["--manifest", str(andmal_like_manifest["manifest"]), "--name", "t", "--strategy",
                      "group_random", "--seed", "0", "--out-dir", str(sp_dir)])
    assert rc == 0
    out = base / "ds"
    rc = consolidate.main(["--manifest", str(andmal_like_manifest["manifest"]), "--processed-root",
                           str(andmal_like_manifest["processed"]), "--splits", str(sp_dir / "t.csv"),
                           "--out", str(out)])
    assert rc == 0
    return out
