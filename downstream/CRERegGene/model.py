from __future__ import annotations

import copy

import torch
from torch import nn


def encoder_only(model: nn.Module) -> nn.Module:
    model = copy.deepcopy(model)
    for name in (
        "decoder_embed", "mask_token", "decoder_blocks",
        "decoder_norm", "decoder_value",
    ):
        setattr(model, name, None)
    return model


class CRERegGeneModel(nn.Module):
    def __init__(self, pretrained_encoder: nn.Module, depth: int = 2, heads: int = 8):
        super().__init__()
        self.region_encoder = encoder_only(pretrained_encoder)
        dim = pretrained_encoder.bin_token.shape[-1]
        self.role_embedding = nn.Embedding(3, dim)
        self.gene_token = nn.Parameter(torch.zeros(1, 1, dim))
        layer = nn.TransformerEncoderLayer(
            dim, heads, dim * 4, dropout=0.1, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.aggregator = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(dim)
        self.regressor = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(dim, 4), nn.Sigmoid(),
        )
        nn.init.normal_(self.gene_token, std=0.02)

    def forward(self, features, role, valid):
        batch, count, width = features.shape
        encoded = self.region_encoder.encode(features.reshape(-1, width))
        encoded = encoded.reshape(batch, count, -1) + self.role_embedding(role)
        token = self.gene_token.expand(batch, -1, -1)
        sequence = torch.cat([token, encoded], dim=1)
        padding = torch.cat([
            torch.zeros(batch, 1, dtype=torch.bool, device=valid.device), ~valid
        ], dim=1)
        gene = self.aggregator(sequence, src_key_padding_mask=padding)[:, 0]
        return self.regressor(self.norm(gene))


def regression_loss(prediction, target, nonzero_weight: float = 2.0):
    error = nn.functional.smooth_l1_loss(prediction, target, reduction="none")
    weights = torch.where(target > 0, nonzero_weight, 1.0)
    return (error * weights).sum() / weights.sum()
