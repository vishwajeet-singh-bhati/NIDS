"""Losses on logits. Class imbalance is handled HERE (pos_weight / focal), never by duplicating data."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_bce(logits: torch.Tensor, y: torch.Tensor, pos_weight: float = 1.0) -> torch.Tensor:
    pw = torch.tensor(pos_weight, dtype=logits.dtype, device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)


def focal_loss(logits: torch.Tensor, y: torch.Tensor, gamma: float = 2.0, pos_weight: float = 1.0) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    p_t = torch.exp(-bce)
    w = torch.where(y > 0.5, torch.full_like(y, pos_weight), torch.ones_like(y))
    return (w * (1 - p_t) ** gamma * bce).mean()


def make_loss(name: str, pos_weight: float, gamma: float = 2.0):
    if name == "bce":
        return lambda lo, y: weighted_bce(lo, y, pos_weight)
    if name == "focal":
        return lambda lo, y: focal_loss(lo, y, gamma, pos_weight)
    raise ValueError("loss must be 'bce' or 'focal'")
