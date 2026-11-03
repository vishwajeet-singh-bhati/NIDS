"""Models B and C: 1D CNNs over the packet sequence.

Padding handling (important for correctness and for early/prefix prediction):
  * the packet mask is appended as an extra input channel,
  * after every convolution block the output is multiplied by the mask so padded positions stay 0,
  * pooling is masked (max over real packets, mean over real packets).
So the logits for a window do not depend on whatever sits in padded positions, and a prefix of a window
(first k packets, rest masked) is a valid input.

Model C adds a per-packet payload branch: byte embedding -> two strided 1D convs along the bytes ->
global max over bytes -> small per-packet vector, concatenated with the metadata channels. It is an
experiment, not an assumption: the payload of encrypted traffic is ciphertext, and the first bytes of
a TLS record (headers, handshake fields) are what it can realistically see.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import torch
from torch import nn

NEG = -1e4


def _norm(kind: str, ch: int) -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm1d(ch)
    if kind == "none":
        return nn.Identity()
    raise ValueError("norm must be 'batch' or 'none'")


class TemporalCNN(nn.Module):
    """[B, C_in, W] + mask [B, W] -> [B, 2*C_last] (masked max || masked mean)."""

    def __init__(self, in_ch: int, channels: Sequence[int], kernel_size: int, norm: str):
        super().__init__()
        self.convs, self.norms = nn.ModuleList(), nn.ModuleList()
        c = in_ch
        for ch in channels:
            self.convs.append(nn.Conv1d(c, ch, kernel_size, padding=kernel_size // 2))
            self.norms.append(_norm(norm, ch))
            c = ch
        self.out_dim = 2 * c

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(1)                                  # [B,1,W]
        x = x * m                                              # padded input rows never reach a convolution
        for conv, nrm in zip(self.convs, self.norms):
            x = torch.relu(nrm(conv(x))) * m
        mx = x.masked_fill(m == 0, NEG).max(dim=2).values
        mx = torch.where(mask.sum(1, keepdim=True) > 0, mx, torch.zeros_like(mx))
        mean = x.sum(dim=2) / m.sum(dim=2).clamp(min=1.0)
        return torch.cat([mx, mean], dim=1)


class MetaCNN1D(nn.Module):
    inputs = {"meta"}

    def __init__(self, n_meta: int, channels: Sequence[int] = (32, 64), kernel_size: int = 3, hidden: int = 32,
                 dropout: float = 0.1, norm: str = "batch"):
        super().__init__()
        self.body = TemporalCNN(n_meta + 1, channels, kernel_size, norm)
        self.head = nn.Sequential(nn.Linear(self.body.out_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, 1))

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        mask = batch["mask"]
        x = torch.cat([batch["meta"], mask.unsqueeze(-1)], dim=-1).transpose(1, 2)   # [B, C+1, W]
        return self.head(self.body(x, mask)).squeeze(-1)


class PayloadBranch(nn.Module):
    """Tokens [B, W, Bn] (0 = padding, 1..256 = byte+1) -> per-packet vector [B, W, D]."""

    def __init__(self, embed_dim: int, channels: Sequence[int], kernel: int, stride: int, out_dim: int):
        super().__init__()
        self.embed = nn.Embedding(257, embed_dim, padding_idx=0)
        layers: List[nn.Module] = []
        c = embed_dim
        for ch in channels:
            layers += [nn.Conv1d(c, ch, kernel, stride=stride, padding=kernel // 2), nn.ReLU()]
            c = ch
        self.convs = nn.Sequential(*layers)
        self.proj = nn.Linear(c, out_dim)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, W, Bn = tokens.shape
        x = self.embed(tokens.reshape(B * W, Bn)).transpose(1, 2)        # [B*W, E, Bn]
        x = self.convs(x).max(dim=2).values                              # [B*W, C]
        x = torch.relu(self.proj(x)).reshape(B, W, -1)
        return x * mask.unsqueeze(-1)


class PayloadMetaCNN1D(nn.Module):
    inputs = {"meta", "payload"}

    def __init__(self, n_meta: int, byte_embed_dim: int = 8, byte_channels: Sequence[int] = (16, 32),
                 byte_kernel: int = 5, byte_stride: int = 2, packet_embed_dim: int = 16,
                 channels: Sequence[int] = (32, 64), kernel_size: int = 3, hidden: int = 32,
                 dropout: float = 0.1, norm: str = "batch"):
        super().__init__()
        self.payload = PayloadBranch(byte_embed_dim, byte_channels, byte_kernel, byte_stride, packet_embed_dim)
        self.body = TemporalCNN(n_meta + 1 + packet_embed_dim, channels, kernel_size, norm)
        self.head = nn.Sequential(nn.Linear(self.body.out_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, 1))

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        mask = batch["mask"]
        pay = self.payload(batch["payload"], mask)
        x = torch.cat([batch["meta"], mask.unsqueeze(-1), pay], dim=-1).transpose(1, 2)
        return self.head(self.body(x, mask)).squeeze(-1)
