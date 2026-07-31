"""Exact DOFA-v2 ViT backbone used by the MARS sensor-aware experiments.

The architecture and dynamic wavelength layer are adapted from the official
DOFA repository at commits c850a16 and 0cfb7e1. The upstream code is MIT
licensed, Copyright (c) 2024 Zhitong Xiong. This local version removes unused
classes/imports, keeps state-dict key names unchanged, and exposes the four
official intermediate maps needed for frozen feature extraction.
"""

from __future__ import annotations

from functools import partial
from typing import Callable

import torch
from timm.models.vision_transformer import VisionTransformer
from torch import nn
from torch.nn import functional as F
from torch.nn import init


def get_1d_sincos_pos_embed_from_grid_torch(
    embed_dim: int, pos: torch.Tensor
) -> torch.Tensor:
    if embed_dim % 2 != 0:
        raise ValueError("Wavelength embedding dimension must be even")
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    values = torch.einsum("m,d->md", pos.reshape(-1), omega)
    return torch.cat((torch.sin(values), torch.cos(values)), dim=1)


class TransformerWeightGenerator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        embed_dim: int,
        num_heads: int = 4,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            activation="gelu",
            norm_first=False,
            batch_first=False,
            dropout=0.0,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.fc_weight = nn.Linear(input_dim, output_dim)
        self.fc_bias = nn.Linear(input_dim, embed_dim)
        self.wt_num = 128
        self.weight_tokens = nn.Parameter(torch.empty((self.wt_num, input_dim)))
        self.bias_token = nn.Parameter(torch.empty((1, input_dim)))
        nn.init.normal_(self.weight_tokens, std=0.02)
        nn.init.normal_(self.bias_token, std=0.02)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        positional_wave = values
        values = torch.cat((self.weight_tokens, positional_wave, self.bias_token), dim=0)
        transformed = self.transformer_encoder(values)
        weights = self.fc_weight(
            transformed[self.wt_num : -1] + positional_wave
        )
        bias = self.fc_bias(transformed[-1])
        return weights, bias


class FCResLayer(nn.Module):
    def __init__(self, linear_size: int = 128) -> None:
        super().__init__()
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.w1 = nn.Linear(linear_size, linear_size)
        self.w2 = nn.Linear(linear_size, linear_size)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.w1(values)
        residual = self.nonlin1(residual)
        residual = self.w2(residual)
        residual = self.nonlin2(residual)
        return values + residual


class Dynamic_MLP_OFA(nn.Module):  # noqa: N801 - upstream checkpoint name
    """Generate a wavelength-conditioned patch projection."""

    def __init__(
        self,
        wv_planes: int,
        inter_dim: int = 128,
        kernel_size: int = 3,
        embed_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.wv_planes = wv_planes
        self.embed_dim = embed_dim
        self._num_kernel = kernel_size * kernel_size * embed_dim
        self.inter_dim = inter_dim
        self.patch_size = (kernel_size, kernel_size)
        self.num_patches = -1
        self.weight_generator = TransformerWeightGenerator(
            wv_planes, self._num_kernel, embed_dim
        )
        self.scaler = 0.01
        self.fclayer = FCResLayer(wv_planes)
        self._init_weights()

    @staticmethod
    def weight_init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)

    def _init_weights(self) -> None:
        self.weight_generator.apply(self.weight_init)
        self.fclayer.apply(self.weight_init)

    def forward(
        self, image_features: torch.Tensor, wavelengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inplanes = wavelengths.size(0)
        waves = get_1d_sincos_pos_embed_from_grid_torch(
            self.wv_planes, wavelengths * 1000
        )
        waves = self.fclayer(waves)
        weight, bias = self.weight_generator(waves)
        dynamic_weight = weight.view(
            inplanes, self.kernel_size, self.kernel_size, self.embed_dim
        ).permute(3, 0, 1, 2)
        scaled_bias = bias.view((self.embed_dim,)) * self.scaler
        projected = F.conv2d(
            image_features,
            dynamic_weight * self.scaler,
            bias=scaled_bias,
            stride=self.kernel_size,
            padding=1,
            dilation=1,
        )
        return projected.flatten(2).transpose(1, 2), waves


class DOFAViT(nn.Module):
    """DOFA-v2 dynamic-wavelength patch embedding plus timm ViT."""

    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int = 14,
        out_indices: tuple[int, ...] = (4, 6, 10, 11),
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        wv_planes: int = 128,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.wv_planes = wv_planes
        self.out_indices = out_indices
        self.patch_embed = Dynamic_MLP_OFA(
            wv_planes=wv_planes,
            inter_dim=128,
            kernel_size=patch_size,
            embed_dim=embed_dim,
        )
        scalar_size = img_size[0] if isinstance(img_size, tuple) else img_size
        self.img_size = scalar_size
        self.num_patches = (scalar_size // patch_size) ** 2
        self.patch_embed.num_patches = self.num_patches
        self.model = VisionTransformer(
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )
        del self.model.patch_embed.proj
        self.dynamic_img_size = True
        self.waves: torch.Tensor | None = None
        # Retained for strict compatibility with the official v2 checkpoint.
        self.norm = norm_layer(embed_dim)

    def forward_features(
        self, values: torch.Tensor, wave_list: list[float] | tuple[float, ...] | None = None
    ) -> list[torch.Tensor]:
        if wave_list is None:
            wave_list = [0.665, 0.56, 0.49]
        if values.shape[1] != len(wave_list):
            raise ValueError("Input channel count and wavelength count differ")
        wavelengths = torch.tensor(wave_list, device=values.device).float()
        self.waves = wavelengths
        values, _ = self.patch_embed(values, wavelengths)
        batch, patches, channels = values.shape
        height = int(patches**0.5)
        if height * height != patches:
            raise ValueError("DOFA-v2 patch grid is not square")
        values = values.view(batch, height, height, channels)
        values = self.model._pos_embed(values)
        values = self.model.patch_drop(values)
        values = self.model.norm_pre(values)
        outputs: list[torch.Tensor] = []
        for index, block in enumerate(self.model.blocks):
            values = block(values)
            if index in self.out_indices:
                patches_only = values[:, 1:]
                output = patches_only.reshape(
                    batch, height, height, channels
                ).permute(0, 3, 1, 2)
                outputs.append(output.contiguous())
        # Match the official architecture even though its normalized final token
        # is not part of the returned multi-depth feature maps.
        self.model.norm(values)
        if len(outputs) != len(self.out_indices):
            raise RuntimeError("DOFA-v2 intermediate feature count differs")
        return outputs

    def forward(
        self, values: torch.Tensor, wave_list: list[float] | tuple[float, ...] | None = None
    ) -> list[torch.Tensor]:
        return self.forward_features(values, wave_list)


def vit_base_patch14(**kwargs: object) -> DOFAViT:
    return DOFAViT(
        out_indices=(4, 6, 10, 11),
        patch_size=14,
        embed_dim=768,
        depth=12,
        num_heads=12,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
