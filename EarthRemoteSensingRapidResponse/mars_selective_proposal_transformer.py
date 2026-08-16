"""Selective spatial transformer for verifying released MARS plume proposals."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SelectiveProposalTransformer(nn.Module):
    """Verify a released connected-component proposal from label-free spatial maps."""

    def __init__(
        self,
        input_channels: int = 9,
        image_size: int = 64,
        patch_size: int = 4,
        dimension: int = 64,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("Image size must be divisible by patch size")
        self.input_channels = input_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.dimension = dimension
        self.grid_size = image_size // patch_size
        self.patch_embed = nn.Conv2d(
            input_channels,
            dimension,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.position = nn.Parameter(
            torch.zeros(1, self.grid_size * self.grid_size, dimension)
        )
        self.sensor_embedding = nn.Embedding(2, dimension)
        layer = nn.TransformerEncoderLayer(
            d_model=dimension,
            nhead=heads,
            dim_feedforward=dimension * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.attention = nn.Linear(dimension, 1)
        self.proposal_attention_scale = nn.Parameter(torch.tensor(1.0))
        self.head = nn.Sequential(
            nn.LayerNorm(dimension * 3),
            nn.Linear(dimension * 3, dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension, 1),
        )
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(
        self,
        values: torch.Tensor,
        sensors: torch.Tensor,
        proposal_probability: torch.Tensor,
        observable: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if values.ndim != 4 or values.shape[1:] != (
            self.input_channels,
            self.image_size,
            self.image_size,
        ):
            raise ValueError("Selective-proposal input differs from the frozen schema")
        if sensors.shape != (values.shape[0],):
            raise ValueError("Sensor indices must have shape B")
        if proposal_probability.shape != (
            values.shape[0],
            1,
            self.image_size,
            self.image_size,
        ):
            raise ValueError("Proposal probability must have shape Bx1xHxW")
        tokens = self.patch_embed(values).flatten(2).transpose(1, 2)
        tokens = tokens + self.position + self.sensor_embedding(sensors)[:, None, :]
        tokens = self.encoder(tokens)
        proposal = F.avg_pool2d(
            proposal_probability.clamp(1e-4, 1.0 - 1e-4),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        ).flatten(1)
        proposal_bias = torch.logit(proposal)
        attention = self.attention(tokens).squeeze(-1)
        attention = attention + self.proposal_attention_scale * proposal_bias
        visible_mask: torch.Tensor | None = None
        if observable is not None:
            if observable.shape != proposal_probability.shape:
                raise ValueError("Observable map must align with proposal probability")
            visible = F.avg_pool2d(
                observable,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            ).flatten(1)
            visible_mask = visible >= 0.1
            attention = attention.masked_fill(~visible_mask, -1e4)
        weights = torch.softmax(attention, dim=1)
        attended = (tokens * weights[:, :, None]).sum(dim=1)
        if visible_mask is None:
            mean = tokens.mean(dim=1)
            maximum = tokens.amax(dim=1)
        else:
            expanded = visible_mask[:, :, None]
            count = expanded.sum(dim=1).clamp_min(1)
            mean = (tokens * expanded).sum(dim=1) / count
            maximum = tokens.masked_fill(~expanded, -1e4).amax(dim=1)
            empty = ~visible_mask.any(dim=1)
            if bool(empty.any()):
                mean = torch.where(empty[:, None], tokens.mean(dim=1), mean)
                maximum = torch.where(empty[:, None], tokens.amax(dim=1), maximum)
        return self.head(torch.cat([attended, mean, maximum], dim=1)).squeeze(1)
