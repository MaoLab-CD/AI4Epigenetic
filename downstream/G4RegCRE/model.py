from __future__ import annotations

import copy

import torch
import torch.nn as nn


def encoder_only(model: nn.Module) -> nn.Module:
    """Drop MAE reconstruction modules that are unused during fine-tuning."""
    model = copy.deepcopy(model)
    for name in (
        "decoder_embed", "mask_token", "decoder_feature_encoder",
        "decoder_input_norm", "decoder_modality_encoder",
        "decoder_partner_modality_encoder", "decoder_data_form_encoder",
        "decoder_blocks", "decoder_norm", "decoder_value",
    ):
        if hasattr(model, name):
            setattr(model, name, None)
    return model


class DistanceEncoder(nn.Module):
    """Encode relation class, signed linear distance, cis/trans and Hi-C strength."""

    def __init__(self, dim: int, max_distance: float = 1e8):
        super().__init__()
        self.max_distance = max_distance
        self.net = nn.Sequential(
            nn.Linear(7, dim), nn.GELU(), nn.LayerNorm(dim), nn.Linear(dim, dim)
        )

    def forward(self, relation, distance, cis, hic):
        signed_log_distance = torch.sign(distance) * torch.log1p(distance.abs())
        signed_log_distance = signed_log_distance / torch.log(
            torch.tensor(self.max_distance + 1, device=distance.device)
        )
        hic = torch.log1p(torch.clamp(hic, min=0))
        features = torch.cat([
            relation, signed_log_distance.unsqueeze(-1),
            distance.abs().log1p().div(torch.log(torch.tensor(
                self.max_distance + 1, device=distance.device
            ))).unsqueeze(-1), cis.unsqueeze(-1), hic.unsqueeze(-1),
        ], dim=-1)
        return self.net(features)


class RelationAwareFusion(nn.Module):
    """Fuse one CRE with its G4 set using relation-specific cross-attention."""

    def __init__(self, dim: int, heads: int = 8, depth: int = 2, dropout: float = 0.1):
        super().__init__()
        self.cross = nn.ModuleList([
            nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
            for _ in range(3)
        ])
        layer = nn.TransformerEncoderLayer(
            dim, heads, dim * 4, dropout, batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.context = nn.TransformerEncoder(layer, depth)
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(dim)

    def forward(self, cre, pairs, relation):
        query = cre.unsqueeze(1)
        contexts = []
        for relation_id, attention in enumerate(self.cross):
            absent = relation[..., relation_id] == 0
            all_absent = absent.all(dim=1)
            safe_mask = absent.clone()
            safe_mask[all_absent, 0] = False
            value, _ = attention(query, pairs, pairs, key_padding_mask=safe_mask)
            value[all_absent] = 0
            contexts.append(value)
        tokens = torch.cat([query] + contexts, dim=1)
        fused = self.context(tokens)[:, 0]
        gate = self.gate(torch.cat([cre, fused], dim=-1))
        return self.norm(cre + gate * fused)


class G4ToCREModel(nn.Module):
    def __init__(
        self, pretrained_encoder: nn.Module, target_dim: int,
        perturbation_dim: int = 4, fusion_depth: int = 2,
    ):
        super().__init__()
        self.g4_encoder = encoder_only(pretrained_encoder)
        self.cre_encoder = copy.deepcopy(self.g4_encoder)
        dim = pretrained_encoder.bin_token.shape[-1]
        self.distance_encoder = DistanceEncoder(dim)
        self.perturbation_encoder = nn.Sequential(
            nn.Linear(perturbation_dim, dim), nn.GELU(),
            nn.LayerNorm(dim), nn.Linear(dim, dim),
        )
        self.pair_norm = nn.LayerNorm(dim)
        self.fusion_encoder = RelationAwareFusion(dim, depth=fusion_depth)
        self.decoder = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(dim, target_dim), nn.Sigmoid(),
        )

    def forward(
        self, g4, cre, relation, distance, cis, hic, perturbation,
        g4_observed, cre_observed,
    ):
        batch, count, features = g4.shape
        g4_repr = self.g4_encoder.encode(
            g4.reshape(batch * count, features), g4_observed
        ).reshape(batch, count, -1)
        cre_repr = self.cre_encoder.encode(cre, cre_observed)
        spatial = self.distance_encoder(relation, distance, cis, hic)
        perturb = self.perturbation_encoder(perturbation)
        pairs = self.pair_norm(g4_repr + spatial + perturb)
        fused = self.fusion_encoder(cre_repr, pairs, relation)
        return self.decoder(fused)


def reconstruction_loss(prediction, target, nonzero_weight: float = 2.0):
    """Smooth-L1 is less dominated by rare extreme peaks than plain MSE."""
    error = nn.functional.smooth_l1_loss(prediction, target, reduction="none")
    weights = torch.where(target > 0, nonzero_weight, 1.0)
    return (error * weights).sum() / weights.sum()
