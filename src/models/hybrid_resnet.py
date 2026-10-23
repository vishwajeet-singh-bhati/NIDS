"""Model D: Multi-Modal Dilated 1D-ResNet with INT8 Quantization-Aware Training (QAT).

Fuses three views of each flow window:
  1. Packet payload branch: byte embedding + strided 1D convs + BatchNorm + max||mean byte pooling.
  2. Packet sequence backbone: 4 Dilated Residual 1D-CNN blocks (d = 1, 2, 4, 8) + Squeeze-and-Excitation
     channel attention + masked (max || mean || attention) temporal pooling.
  3. Flow aggregate branch: 2-layer MLP projection over the 22 normalized aggregate window features.

QAT (Quantization-Aware Training):
  - QATConv1d and QATLinear apply symmetric per-out-channel INT8 weight fake-quantization ([-127, 127])
    and per-tensor INT8 activation fake-quantization via Straight-Through Estimator (STE) once
    `qat_active` is enabled (controlled by `qat_start_epoch` during training).
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from torch import nn

NEG = -1e4


def fake_quant_weight_per_channel(w: torch.Tensor, num_bits: int = 8) -> torch.Tensor:
    """Symmetric per-output-channel INT8 fake quantization with STE."""
    qmax = float((1 << (num_bits - 1)) - 1)  # 127.0
    flat = w.reshape(w.shape[0], -1)
    max_val = flat.abs().amax(dim=1).clamp(min=1e-8)
    scale = (max_val / qmax).reshape([w.shape[0]] + [1] * (w.ndim - 1))
    w_q = torch.clamp(torch.round(w / scale), -qmax, qmax) * scale
    return w + (w_q - w).detach()


def fake_quant_act_per_tensor(x: torch.Tensor, num_bits: int = 8) -> torch.Tensor:
    """Symmetric per-tensor INT8 activation fake quantization with STE."""
    qmax = float((1 << (num_bits - 1)) - 1)  # 127.0
    max_val = x.detach().abs().amax().clamp(min=1e-8)
    scale = max_val / qmax
    x_q = torch.clamp(torch.round(x / scale), -qmax, qmax) * scale
    return x + (x_q - x).detach()


class QATConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qat_active = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.qat_active:
            x = fake_quant_act_per_tensor(x)
            w = fake_quant_weight_per_channel(self.weight)
            return F.conv1d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return super().forward(x)


class QATLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qat_active = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.qat_active:
            x = fake_quant_act_per_tensor(x)
            w = fake_quant_weight_per_channel(self.weight)
            return F.linear(x, w, self.bias)
        return super().forward(x)


class PayloadResBranch(nn.Module):
    """Tokens [B, W, Bn] -> per-packet vector [B, W, out_dim]."""

    def __init__(self, embed_dim: int = 24, channels: Sequence[int] = (64, 128, 128),
                 kernel: int = 5, stride: int = 2, out_dim: int = 64):
        super().__init__()
        self.embed = nn.Embedding(257, embed_dim, padding_idx=0)
        layers: List[nn.Module] = []
        c = embed_dim
        for ch in channels:
            layers += [
                QATConv1d(c, ch, kernel, stride=stride, padding=kernel // 2, bias=False),
                nn.BatchNorm1d(ch),
                nn.ReLU(inplace=True),
            ]
            c = ch
        self.convs = nn.Sequential(*layers)
        self.proj = nn.Sequential(
            QATLinear(2 * c, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, W, Bn = tokens.shape
        x = self.embed(tokens.reshape(B * W, Bn)).transpose(1, 2)  # [B*W, E, Bn]
        x = self.convs(x)                                          # [B*W, C, Bn']
        mx = x.amax(dim=2)
        mn = x.mean(dim=2)
        out = self.proj(torch.cat([mx, mn], dim=1)).reshape(B, W, -1)
        return out * mask.unsqueeze(-1)


class DilatedResBlock1D(nn.Module):
    """Masked Residual 1D-CNN block with dilation + Squeeze-and-Excitation."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1,
                 dropout: float = 0.15):
        super().__init__()
        pad = (kernel_size // 2) * dilation
        self.conv1 = QATConv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = QATConv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.down = (
            nn.Sequential(QATConv1d(in_ch, out_ch, 1, bias=False), nn.BatchNorm1d(out_ch))
            if in_ch != out_ch else nn.Identity()
        )
        se_hidden = max(out_ch // 4, 16)
        self.se1 = QATLinear(out_ch, se_hidden)
        self.se2 = QATLinear(se_hidden, out_ch)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        # x: [B, C, W], m: [B, 1, W]
        res = self.down(x) * m
        h = F.relu(self.bn1(self.conv1(x * m))) * m
        h = self.drop(h)
        h = self.bn2(self.conv2(h)) * m
        # Masked Squeeze-and-Excitation channel gating
        denom = m.sum(dim=2).clamp(min=1.0)
        s = h.sum(dim=2) / denom
        w = torch.sigmoid(self.se2(F.relu(self.se1(s)))).unsqueeze(2)
        h = h * w
        return F.relu(h + res) * m


class HybridResNet1D(nn.Module):
    inputs = {"meta", "payload", "agg"}

    def __init__(
        self,
        n_meta: int,
        n_agg: int = 22,
        byte_embed_dim: int = 24,
        byte_channels: Sequence[int] = (64, 128, 128),
        byte_kernel: int = 5,
        byte_stride: int = 2,
        packet_embed_dim: int = 64,
        channels: Sequence[int] = (128, 192, 256, 256),
        dilations: Sequence[int] = (1, 2, 4, 8),
        kernel_size: int = 3,
        agg_hidden: int = 64,
        head_hidden: int = 128,
        dropout: float = 0.2,
        qat: bool = True,
        qat_start_epoch: int = 8,
    ):
        super().__init__()
        self.qat_enabled = bool(qat)
        self.qat_start_epoch = int(qat_start_epoch)
        self.register_buffer("qat_active_buf", torch.tensor(0, dtype=torch.uint8))

        self.payload = PayloadResBranch(byte_embed_dim, byte_channels, byte_kernel, byte_stride, packet_embed_dim)
        in_ch = n_meta + 1 + packet_embed_dim
        self.blocks = nn.ModuleList()
        c = in_ch
        for ch, dil in zip(channels, dilations):
            self.blocks.append(DilatedResBlock1D(c, ch, kernel_size=kernel_size, dilation=dil, dropout=dropout))
            c = ch
        self.attn = QATConv1d(c, 1, kernel_size=1)

        self.agg_net = nn.Sequential(
            QATLinear(n_agg, agg_hidden, bias=False),
            nn.BatchNorm1d(agg_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            QATLinear(agg_hidden, agg_hidden, bias=False),
            nn.BatchNorm1d(agg_hidden),
            nn.ReLU(inplace=True),
        )

        fusion_dim = (3 * c) + agg_hidden
        self.head = nn.Sequential(
            QATLinear(fusion_dim, head_hidden, bias=False),
            nn.BatchNorm1d(head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            QATLinear(head_hidden, head_hidden // 2, bias=False),
            nn.BatchNorm1d(head_hidden // 2),
            nn.ReLU(inplace=True),
            QATLinear(head_hidden // 2, 1),
        )

    def enable_qat(self, active: bool = True) -> None:
        self.qat_active_buf.fill_(1 if active else 0)
        for m in self.modules():
            if isinstance(m, (QATConv1d, QATLinear)):
                m.qat_active = bool(active)

    def set_qat_epoch(self, epoch: int) -> bool:
        active = self.qat_enabled and (epoch >= self.qat_start_epoch)
        self.enable_qat(active)
        return active

    def _sync_qat_from_buffer(self) -> None:
        active = bool(self.qat_active_buf.item() == 1)
        for m in self.modules():
            if isinstance(m, (QATConv1d, QATLinear)):
                m.qat_active = active

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        self._sync_qat_from_buffer()
        mask = batch["mask"]                                                        # [B, W]
        m = mask.unsqueeze(1)                                                       # [B, 1, W]
        pay = self.payload(batch["payload"], mask)                                  # [B, W, P]
        x = torch.cat([batch["meta"], mask.unsqueeze(-1), pay], dim=-1).transpose(1, 2)  # [B, C, W]
        x = x * m
        for blk in self.blocks:
            x = blk(x, m)

        has_pkt = mask.sum(1, keepdim=True) > 0
        # 1. Masked Max Pool
        mx = x.masked_fill(m == 0, NEG).max(dim=2).values
        mx = torch.where(has_pkt, mx, torch.zeros_like(mx))
        # 2. Masked Mean Pool
        mn = x.sum(dim=2) / m.sum(dim=2).clamp(min=1.0)
        # 3. Masked Attention Pool
        scores = self.attn(x).masked_fill(m == 0, NEG)
        weights = torch.softmax(scores, dim=2) * m
        att = (x * weights).sum(dim=2)

        seq_feat = torch.cat([mx, mn, att], dim=1)
        agg_feat = self.agg_net(batch["agg"])
        return self.head(torch.cat([seq_feat, agg_feat], dim=1)).squeeze(-1)
