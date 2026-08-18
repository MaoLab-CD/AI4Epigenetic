# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Adapted from scMETH/model/models_scmeth.py and the MAE implementation.

from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block


class MaskedAutoencoderBin(nn.Module):
    """Masked Autoencoder with a Vision Transformer backbone."""

    def __init__(
        self,
        feature_count,
        hidden_dim=128,
        embed_dim=512,
        depth=24,
        num_heads=8,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
    ):
        super().__init__()

        # --------------------------------------------------------------------------
        # Bin MAE encoder specifics
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
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # Bin MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_feature_encoder = nn.Embedding(
            feature_count + 1, decoder_embed_dim
        )

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_value = nn.Sequential(
            nn.Linear(decoder_embed_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # --------------------------------------------------------------------------

        self.feature_count = feature_count
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.bin_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.

        x: [N, L, D], sequence
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
        )

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, values, mask_ratio):
        feature_ids = torch.arange(self.feature_count, device=values.device)
        feature_embs = self.feature_encoder(feature_ids).unsqueeze(0)
        value_embs = self.value_encoder(values)
        x = feature_embs + value_embs

        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        bin_tokens = self.bin_token.expand(values.shape[0], -1, -1)
        x = torch.cat((bin_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)

        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(
            x_,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]),
        )
        x = torch.cat([x[:, :1, :], x_], dim=1)

        position_ids = torch.arange(
            self.feature_count + 1, device=x.device
        )
        x = x + self.decoder_feature_encoder(position_ids).unsqueeze(0)

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        x = x[:, 1:, :]
        x = self.decoder_value(x).squeeze(-1)

        return x

    def forward_loss(self, values, pred, mask):
        target = values.float()

        if self.norm_pix_loss:
            mean = target.mean(dim=1, keepdim=True)
            var = target.var(dim=1, keepdim=True)
            target = (target - mean) / torch.sqrt(var + 1.0e-6)

        loss = (pred.float() - target).pow(2)
        loss = (loss * mask).sum() / mask.sum()

        return loss

    def forward(self, values, mask_ratio=0.4):
        latent, mask, ids_restore = self.forward_encoder(values, mask_ratio)

        with torch.cuda.amp.autocast(enabled=False):
            pred = self.forward_decoder(latent, ids_restore)
            loss_mask = self.forward_loss(values, pred, mask)

        return latent, loss_mask, pred, mask


class ValueEncoder(nn.Module):
    """Encode real-valued multimodal features with a shared projection."""

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


def mae_bin_base(**kwargs):
    model = MaskedAutoencoderBin(
        hidden_dim=128,
        embed_dim=512,
        depth=8,
        num_heads=8,
        decoder_embed_dim=256,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_bin_large(**kwargs):
    model = MaskedAutoencoderBin(
        hidden_dim=128,
        embed_dim=512,
        depth=12,
        num_heads=16,
        decoder_embed_dim=256,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_bin_huge(**kwargs):
    model = MaskedAutoencoderBin(
        hidden_dim=128,
        embed_dim=768,
        depth=16,
        num_heads=16,
        decoder_embed_dim=256,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model
