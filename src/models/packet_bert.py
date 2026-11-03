"""Model E: PacketBERT - a compact BERT-style Transformer encoder over the packets of a flow window.

Why this and not `bert-base`:
  * a text-pretrained BERT (110M parameters, word-piece vocabulary) has nothing to transfer to packet
    metadata, and is far too heavy for an always-on on-device detector;
  * there is no public pretrained *network-traffic* checkpoint that matches this feature spec.
  So the "pretrained model" here is a small encoder (~0.3-1M parameters) that is pre-trained by us on
  UNLABELLED train-split windows (masked packet modelling, `src.training.pretrain`) and then
  fine-tuned for the classification head with quantisation-aware training (QAT).

Token = one packet. Packet embedding = Linear(12 normalised meta channels) [+ payload-byte branch]
+ learned position embedding. A learned [CLS] token is prepended. Padding (mask = 0) is never attended
to, so a window PREFIX (first k packets) is a valid input - that is what makes early/streaming
prediction work with the same weights.

INT8 friendliness: ReLU feed-forward (no GELU), pre-LayerNorm, every Linear/Conv is a QAT layer
(per-channel INT8 weights, per-tensor INT8 activations with straight-through gradients, reused from
`hybrid_resnet`). Softmax / LayerNorm / residual adds are NOT fake-quantised; a real INT8 export has to
handle them (see README, "What is not done yet").
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from src.models.hybrid_resnet import QATConv1d, QATLinear

NEG = -1e4


class PayloadBytes(nn.Module):
    """tokens [B,W,Bn] (0 = pad, 1..256 = byte+1) -> per-packet vector [B,W,D]."""

    def __init__(self, embed_dim: int, channels: Sequence[int], kernel: int, stride: int, out_dim: int):
        super().__init__()
        self.embed = nn.Embedding(257, embed_dim, padding_idx=0)
        layers, c = [], embed_dim
        for ch in channels:
            layers += [QATConv1d(c, ch, kernel, stride=stride, padding=kernel // 2), nn.ReLU()]
            c = ch
        self.convs = nn.Sequential(*layers)
        self.proj = QATLinear(c, out_dim)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, W, Bn = tokens.shape
        x = self.embed(tokens.reshape(B * W, Bn)).transpose(1, 2)
        x = self.convs(x).max(dim=2).values
        return torch.relu(self.proj(x)).reshape(B, W, -1) * mask.unsqueeze(-1)


class Block(nn.Module):
    def __init__(self, d: int, heads: int, ff: int, dropout: float):
        super().__init__()
        if d % heads:
            raise ValueError("d_model must be divisible by heads")
        self.h, self.dk = heads, d // heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = QATLinear(d, 3 * d)
        self.out = QATLinear(d, d)
        self.ff1, self.ff2 = QATLinear(d, ff), QATLinear(ff, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_ok: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).reshape(B, T, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk)               # [B,h,T,T]
        att = att.masked_fill(~key_ok[:, None, None, :], NEG).softmax(-1)
        y = (self.drop(att) @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.drop(self.out(y))
        return x + self.drop(self.ff2(self.drop(F.relu(self.ff1(self.ln2(x))))))


class PacketBERT(nn.Module):
    def __init__(self, n_meta: int, d_model: int = 96, heads: int = 4, layers: int = 3,
                 ff_mult: int = 2, dropout: float = 0.1, use_payload: bool = False, byte_embed_dim: int = 8,
                 byte_channels: Sequence[int] = (16, 32), byte_kernel: int = 5, byte_stride: int = 2,
                 head_hidden: int = 64, max_window: int = 64, qat: bool = True, qat_start_epoch: int = 6):
        super().__init__()
        self.inputs = {"meta", "payload"} if use_payload else {"meta"}
        self.use_payload, self.d = bool(use_payload), int(d_model)
        self.qat_enabled, self.qat_start_epoch = bool(qat), int(qat_start_epoch)
        self.register_buffer("qat_active_buf", torch.tensor(0, dtype=torch.uint8))
        self.n_meta = n_meta
        self.meta_in = QATLinear(n_meta, d_model)
        self.payload = PayloadBytes(byte_embed_dim, byte_channels, byte_kernel, byte_stride, d_model) if use_payload else None
        self.pos = nn.Parameter(torch.zeros(max_window + 1, d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.mask_tok = nn.Parameter(torch.zeros(1, 1, d_model))        # [MASK] embedding (pretraining only)
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.mask_tok, std=0.02)
        self.blocks = nn.ModuleList(Block(d_model, heads, ff_mult * d_model, dropout) for _ in range(layers))
        self.ln_f = nn.LayerNorm(d_model)
        # heads (NOT part of the transferable encoder)
        self.head = nn.Sequential(QATLinear(2 * d_model, head_hidden), nn.ReLU(), nn.Dropout(dropout),
                                  QATLinear(head_hidden, 1))
        self.mpm_head = nn.Linear(d_model, n_meta)

    # ---------------------------------------------------------------- QAT switches (same API as hybrid_resnet)
    def enable_qat(self, active: bool = True) -> None:
        self.qat_active_buf.fill_(1 if active else 0)
        for m in self.modules():
            if isinstance(m, (QATConv1d, QATLinear)):
                m.qat_active = bool(active)

    def set_qat_epoch(self, epoch: int) -> bool:
        active = self.qat_enabled and epoch >= self.qat_start_epoch
        self.enable_qat(active)
        return active

    def _sync_qat(self) -> None:
        a = bool(self.qat_active_buf.item() == 1)
        for m in self.modules():
            if isinstance(m, (QATConv1d, QATLinear)):
                m.qat_active = a

    # ---------------------------------------------------------------- encoder
    def encode(self, batch: Dict[str, torch.Tensor], masked_packets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """-> hidden [B, 1+W, D]; position 0 is [CLS]. `masked_packets` [B,W] bool (pretraining) replaces a
        packet's embedding with the learned [MASK] token."""
        self._sync_qat()
        mask = batch["mask"]
        B, W = mask.shape
        x = self.meta_in(batch["meta"])
        if self.payload is not None:
            x = x + self.payload(batch["payload"], mask)
        if masked_packets is not None:
            x = torch.where(masked_packets.unsqueeze(-1), self.mask_tok.expand(B, W, -1), x)
        x = torch.cat([self.cls.expand(B, 1, -1), x], dim=1) + self.pos[: W + 1]
        key_ok = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=mask.device), mask > 0], dim=1)
        for blk in self.blocks:
            x = blk(x, key_ok)
        return self.ln_f(x)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.encode(batch)
        mask = batch["mask"].unsqueeze(-1)
        mean = (h[:, 1:] * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return self.head(torch.cat([h[:, 0], mean], dim=1)).squeeze(-1)

    def mpm_predict(self, batch: Dict[str, torch.Tensor], masked_packets: torch.Tensor) -> torch.Tensor:
        """Masked-packet modelling: predict the (normalised) meta vector of every packet -> [B,W,n_meta]."""
        return self.mpm_head(self.encode(batch, masked_packets)[:, 1:])

    def encoder_state(self) -> Dict[str, torch.Tensor]:
        skip = ("head.", "mpm_head.", "qat_active_buf")
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items() if not k.startswith(skip)}
