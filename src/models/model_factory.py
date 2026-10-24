"""Build models from config + the dataset's feature spec, summarise them, save/load checkpoints."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import yaml
from torch import nn

from src.data.feature_extraction import AGG_FEATURE_NAMES
from src.models.cnn_1d import MetaCNN1D, PayloadMetaCNN1D
from src.models.mlp_baseline import MLPBaseline
from src.models.hybrid_resnet import HybridResNet1D
from src.models.packet_bert import PacketBERT
from src.utils.config import N_META

MODEL_NAMES = ("mlp", "cnn_meta", "cnn_payload", "hybrid_resnet", "packet_bert", "packet_bert_pay")
LABEL_MAPPING = {"0": "benign", "1": "malicious_associated"}


def load_model_config(path: str | Path, name: str) -> Dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text()) or {}
    if name not in cfg:
        raise KeyError(f"model '{name}' not in {path}")
    return dict(cfg[name])


def build_model(name: str, cfg: Dict[str, Any]) -> nn.Module:
    """Input sizes are fixed by the feature spec (N_META channels, 22 aggregate features)."""
    cfg = dict(cfg)
    if name == "mlp":
        return MLPBaseline(len(AGG_FEATURE_NAMES), **cfg)
    if name == "cnn_meta":
        return MetaCNN1D(N_META, **cfg)
    if name == "cnn_payload":
        return PayloadMetaCNN1D(N_META, **cfg)
    if name == "hybrid_resnet":
        return HybridResNet1D(N_META, len(AGG_FEATURE_NAMES), **cfg)
    if name in ("packet_bert", "packet_bert_pay"):
        return PacketBERT(N_META, **cfg)
    raise ValueError(f"unknown model '{name}'; choose from {MODEL_NAMES}")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(train),
            "fp32_bytes": int(total * 4), "int8_bytes_estimate": int(total)}


def model_summary(model: nn.Module) -> str:
    lines = [f"{type(model).__name__}", f"inputs: {sorted(model.inputs)}"]
    for n, p in model.named_parameters():
        lines.append(f"  {n:45s} {str(tuple(p.shape)):18s} {p.numel():>8d}")
    c = count_parameters(model)
    lines.append(f"total parameters: {c['total']}  (~{c['fp32_bytes'] / 1024:.1f} KiB fp32, "
                 f"~{c['int8_bytes_estimate'] / 1024:.1f} KiB if int8 - estimate only, not measured)")
    return "\n".join(lines)


def save_checkpoint(path: str | Path, model: nn.Module, model_name: str, model_cfg: Dict[str, Any],
                    feature_spec: Dict[str, Any], normalization: Dict[str, Any], train_cfg: Dict[str, Any],
                    epoch: int, val_metrics: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_name": model_name, "model_cfg": model_cfg, "state_dict": model.state_dict(),
                "feature_spec": feature_spec, "normalization": normalization, "label_mapping": LABEL_MAPPING,
                "train_cfg": train_cfg, "epoch": epoch, "val_metrics": val_metrics, "extra": extra or {}}, path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> Tuple[nn.Module, Dict[str, Any]]:
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ck["model_name"], ck["model_cfg"])
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, ck
