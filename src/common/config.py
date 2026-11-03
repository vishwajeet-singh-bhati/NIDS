"""Typed configuration objects shared by every phase.

The YAML file is the single source of truth. These dataclasses validate it and are
serialised verbatim into every processed shard (`feature_spec_json`) so that training and
the Android runtime can reproduce the exact same feature layout.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

SUPPORTED_WINDOW_SIZES = (16, 32, 64)

# Fixed channel layout of the per-packet metadata tensor. Do not reorder without bumping
# feature_spec_version: the model and the Android encoder both depend on this order.
META_CHANNELS: List[str] = [
    "direction",      # +1 initiator->responder, -1 responder->initiator
    "payload_len",    # raw transport payload length in bytes (normalised later, train-set stats)
    "log_iat",        # log1p(inter-arrival ms) w.r.t. previous packet in the SAME window; 0 for first
    "proto_tcp",
    "proto_udp",      # UDP that is not DNS
    "proto_dns",
    "proto_other",    # ICMP, ICMPv6, ESP, ... (ports are 0)
    "flag_syn",
    "flag_fin",
    "flag_rst",
    "flag_psh",
    "flag_ack",
]
N_META = len(META_CHANNELS)

PROTO_TCP, PROTO_UDP, PROTO_DNS, PROTO_OTHER = 0, 1, 2, 3
PROTO_NAMES = {PROTO_TCP: "tcp", PROTO_UDP: "udp", PROTO_DNS: "dns", PROTO_OTHER: "other"}


@dataclass
class ParseConfig:
    dns_ports: List[int] = field(default_factory=lambda: [53, 5353])
    payload_bytes_kept: int = 128
    max_packets_per_capture: int = 0

    def validate(self) -> None:
        if self.payload_bytes_kept < 0 or self.payload_bytes_kept > 1500:
            raise ValueError("parse.payload_bytes_kept must be in [0, 1500]")
        if self.max_packets_per_capture < 0:
            raise ValueError("parse.max_packets_per_capture must be >= 0")


@dataclass
class FlowConfig:
    tcp_idle_timeout_s: float = 120.0
    udp_idle_timeout_s: float = 60.0
    other_idle_timeout_s: float = 60.0
    max_flow_duration_s: float = 600.0
    tcp_time_wait_s: float = 5.0
    max_packets_per_flow: int = 1500
    drop_tcp_retransmissions: bool = True
    expire_check_interval_s: float = 1.0

    def validate(self) -> None:
        for name in ("tcp_idle_timeout_s", "udp_idle_timeout_s", "other_idle_timeout_s",
                     "max_flow_duration_s", "expire_check_interval_s"):
            if getattr(self, name) <= 0:
                raise ValueError(f"flow.{name} must be > 0")
        if self.tcp_time_wait_s < 0:
            raise ValueError("flow.tcp_time_wait_s must be >= 0")
        if self.max_packets_per_flow < 2:
            raise ValueError("flow.max_packets_per_flow must be >= 2")


@dataclass
class SequenceConfig:
    window_size: int = 32
    stride: int = 32
    min_packets: int = 2
    min_tail_packets: int = 8
    max_windows_per_flow: int = 20
    include_payload: bool = True
    iat_log_unit_ms: bool = True
    max_examples_per_capture: int = 0   # 0 = keep all. >0: deterministic uniform sample of windows
    sampling_seed: int = 0

    def validate(self) -> None:
        if self.max_examples_per_capture < 0:
            raise ValueError("sequence.max_examples_per_capture must be >= 0")
        if self.window_size not in SUPPORTED_WINDOW_SIZES:
            raise ValueError(f"sequence.window_size must be one of {SUPPORTED_WINDOW_SIZES}")
        if not (1 <= self.stride <= self.window_size):
            raise ValueError("sequence.stride must satisfy 1 <= stride <= window_size")
        if not (1 <= self.min_packets <= self.window_size):
            raise ValueError("sequence.min_packets must satisfy 1 <= min_packets <= window_size")
        if self.min_tail_packets < 1:
            raise ValueError("sequence.min_tail_packets must be >= 1")
        if self.max_windows_per_flow < 1:
            raise ValueError("sequence.max_windows_per_flow must be >= 1")


@dataclass
class DatasetConfig:
    name: str = "unnamed"
    raw_root: str = "data/raw"
    processed_root: str = "data/processed"
    manifest_path: str = "data/manifests/manifest.csv"
    pcap_extensions: List[str] = field(default_factory=lambda: [".pcap", ".pcapng", ".cap"])
    label_config: str = "configs/labels.yaml"


@dataclass
class PipelineConfig:
    feature_spec_version: str = "1.1"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    parse: ParseConfig = field(default_factory=ParseConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    sequence: SequenceConfig = field(default_factory=SequenceConfig)

    def validate(self) -> None:
        self.parse.validate()
        self.flow.validate()
        self.sequence.validate()
        need = (self.sequence.max_windows_per_flow - 1) * self.sequence.stride + self.sequence.window_size
        if self.flow.max_packets_per_flow < need:
            raise ValueError(
                f"flow.max_packets_per_flow={self.flow.max_packets_per_flow} is too small for sequence settings "
                f"(window={self.sequence.window_size}, stride={self.sequence.stride}, "
                f"max_windows_per_flow={self.sequence.max_windows_per_flow}); need >= {need}")

    def feature_spec(self) -> Dict[str, Any]:
        """Everything that defines the layout/meaning of the saved tensors."""
        return {
            "feature_spec_version": self.feature_spec_version,
            "meta_channels": META_CHANNELS,
            "window_size": self.sequence.window_size,
            "stride": self.sequence.stride,
            "payload_bytes_kept": self.parse.payload_bytes_kept,
            "include_payload": self.sequence.include_payload,
            "iat_log_unit_ms": self.sequence.iat_log_unit_ms,
            "time_model": "monotone_window_clock",
            "dns_ports": list(self.parse.dns_ports),
            "proto_codes": {v: k for k, v in PROTO_NAMES.items()},
        }

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def _section(cls, raw: Optional[Dict[str, Any]], name: str):
    raw = dict(raw or {})
    known = set(cls.__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown keys in config section '{name}': {sorted(unknown)}")
    return cls(**raw)


def load_config(path: str | Path, overrides: Optional[Dict[str, Any]] = None) -> PipelineConfig:
    """Load and validate configs/data.yaml. `overrides` uses dotted keys, e.g.
    {"sequence.window_size": 16}. Unknown keys raise instead of being silently ignored."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    raw = copy.deepcopy(raw)
    for dotted, value in (overrides or {}).items():
        if value is None:
            continue
        parts = dotted.split(".")
        node = raw
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    top_known = {"feature_spec_version", "dataset", "parse", "flow", "sequence"}
    unknown = set(raw) - top_known
    if unknown:
        raise ValueError(f"Unknown top-level config keys: {sorted(unknown)}")
    cfg = PipelineConfig(
        feature_spec_version=str(raw.get("feature_spec_version", "1.1")),
        dataset=_section(DatasetConfig, raw.get("dataset"), "dataset"),
        parse=_section(ParseConfig, raw.get("parse"), "parse"),
        flow=_section(FlowConfig, raw.get("flow"), "flow"),
        sequence=_section(SequenceConfig, raw.get("sequence"), "sequence"),
    )
    cfg.validate()
    return cfg
