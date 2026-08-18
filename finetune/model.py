# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block


class PolIIPeakRegression(nn.Module):
    """Transformer encoder for Pol II S5P peak regression."""

    def __init__(
        self,
        feature_count,
        hidden_dim=256,
        embed_dim=512,
        depth=24,
        num_heads=8,
        mlp_ratio=4,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.feature_encoder = nn.Embedding(feature_count, embed_dim)
        self.value_encoder = ValueEncoder(embed_dim)
        self.bin_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.regression_decoder = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward_features(self, values):
        feature_ids = torch.arange(values.shape[1], device=values.device)
        feature_embs = self.feature_encoder(feature_ids).unsqueeze(0)
        value_embs = self.value_encoder(values)
        x = feature_embs + value_embs

        bin_tokens = self.bin_token.expand(values.shape[0], -1, -1)
        x = torch.cat((bin_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward(self, values):
        x = self.forward_features(values)
        return self.regression_decoder(x[:, 0, :]).squeeze(-1)


class ValueEncoder(nn.Module):
    def __init__(self, d_model, dropout=0.5):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)
        return self.dropout(x)


def polii_peak_base(**kwargs):
    return PolIIPeakRegression(
        hidden_dim=256,
        embed_dim=512,
        depth=8,
        num_heads=8,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def polii_peak_large(**kwargs):
    return PolIIPeakRegression(
        hidden_dim=256,
        embed_dim=512,
        depth=12,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def polii_peak_huge(**kwargs):
    return PolIIPeakRegression(
        hidden_dim=256,
        embed_dim=768,
        depth=16,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
