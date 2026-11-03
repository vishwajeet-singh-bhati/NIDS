"""Model A: MLP on the 22 aggregate window features (the simple baseline to beat)."""
from __future__ import annotations

from typing import Dict, List, Sequence

import torch
from torch import nn


class MLPBaseline(nn.Module):
    inputs = {"agg"}

    def __init__(self, in_dim: int, hidden: Sequence[int] = (64, 32), dropout: float = 0.1):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.net(batch["agg"]).squeeze(-1)      # logits [B]
