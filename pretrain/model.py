from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block


DATA_FORMS = ("peak", "self_ratio", "bin_ratio", "overlap")


def build_feature_metadata(feature_names):
    """Parse exact feature, denominator/numerator modality and value form.

    For ``A_overlapping_B``, B is the denominator modality and A is the
    covering/numerator modality. Unary features only have a denominator/owner.
    """
    parsed, modalities = [], []
    for name in feature_names:
        if "_overlapping_" in name:
            partner, primary = name.split("_overlapping_", 1)
            form = "overlap"
        else:
            form = next(
                (x for x in DATA_FORMS[:3] if name.endswith(f"_{x}")), None
            )
            if form is None:
                raise ValueError(f"Cannot identify data form: {name}")
            primary, partner = name[: -len(form) - 1], None
        parsed.append((primary, partner, form))
        for modality in (primary, partner):
            if modality is not None and modality not in modalities:
                modalities.append(modality)

    modality_ids = {name: i for i, name in enumerate(modalities)}
    form_ids = {name: i for i, name in enumerate(DATA_FORMS)}
    none_modality = len(modalities)
    return {
        "modalities": modalities,
        "parsed": parsed,
        "primary_ids": [modality_ids[x[0]] for x in parsed],
        "partner_ids": [
            none_modality if x[1] is None else modality_ids[x[1]] for x in parsed
        ],
        "form_ids": [form_ids[x[2]] for x in parsed],
    }


class MaskedAutoencoderBin(nn.Module):
    """Masked autoencoder for one multimodal genomic bin."""

    def __init__(
        self,
        feature_count=None,
        feature_names=None,
        hidden_dim=128,
        embed_dim=512,
        depth=24,
        num_heads=8,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        embedding_dropout=0.1,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
    ):
        super().__init__()
        if feature_names is not None:
            feature_names = list(feature_names)
            if feature_count is not None and feature_count != len(feature_names):
                raise ValueError("feature_count and feature_names disagree")
            feature_count = len(feature_names)
        if feature_count is None:
            raise ValueError("feature_count or feature_names is required")

        self.feature_count = feature_count
        self.feature_names = feature_names
        self.typed_features = feature_names is not None
        self.norm_pix_loss = norm_pix_loss

        self.feature_encoder = nn.Embedding(feature_count, embed_dim)
        self.value_encoder = ValueEncoder(embed_dim, dropout=embedding_dropout)
        self.input_norm = norm_layer(embed_dim)
        self.input_dropout = nn.Dropout(embedding_dropout)
        self.bin_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if self.typed_features:
            metadata = build_feature_metadata(feature_names)
            self.modalities = metadata["modalities"]
            self.modality_count = len(self.modalities)
            self.none_modality = self.modality_count
            self.none_form = len(DATA_FORMS)
            for name, values in (
                ("primary_modality_ids", metadata["primary_ids"]),
                ("partner_modality_ids", metadata["partner_ids"]),
                ("data_form_ids", metadata["form_ids"]),
            ):
                self.register_buffer(name, torch.tensor(values, dtype=torch.long))
            self.modality_encoder = nn.Embedding(
                self.modality_count + 1, embed_dim, padding_idx=self.none_modality
            )
            self.partner_modality_encoder = nn.Embedding(
                self.modality_count + 1, embed_dim, padding_idx=self.none_modality
            )
            self.data_form_encoder = nn.Embedding(
                len(DATA_FORMS) + 1, embed_dim, padding_idx=self.none_form
            )

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None,
                  norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_feature_encoder = nn.Embedding(feature_count + 1, decoder_embed_dim)
        self.decoder_input_norm = norm_layer(decoder_embed_dim)
        if self.typed_features:
            self.decoder_modality_encoder = nn.Embedding(
                self.modality_count + 1, decoder_embed_dim,
                padding_idx=self.none_modality
            )
            self.decoder_partner_modality_encoder = nn.Embedding(
                self.modality_count + 1, decoder_embed_dim,
                padding_idx=self.none_modality
            )
            self.decoder_data_form_encoder = nn.Embedding(
                len(DATA_FORMS) + 1, decoder_embed_dim, padding_idx=self.none_form
            )
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True,
                  qk_scale=None, norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_value = nn.Sequential(
            nn.Linear(decoder_embed_dim, hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.bin_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)
        embeddings = [self.feature_encoder, self.decoder_feature_encoder]
        if self.typed_features:
            embeddings.extend((
                self.modality_encoder, self.partner_modality_encoder,
                self.data_form_encoder, self.decoder_modality_encoder,
                self.decoder_partner_modality_encoder,
                self.decoder_data_form_encoder,
            ))
        for embedding in embeddings:
            torch.nn.init.normal_(embedding.weight, std=0.02)
            if embedding.padding_idx is not None:
                with torch.no_grad():
                    embedding.weight[embedding.padding_idx].zero_()

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def feature_type_embeddings(self):
        return (
            self.modality_encoder(self.primary_modality_ids)
            + self.partner_modality_encoder(self.partner_modality_ids)
            + self.data_form_encoder(self.data_form_ids)
        )

    def embed_features(self, values):
        """Combine value, exact feature, modality-role and data-form semantics."""
        ids = torch.arange(self.feature_count, device=values.device)
        x = self.feature_encoder(ids).unsqueeze(0) + self.value_encoder(values)
        if self.typed_features:
            x = x + self.feature_type_embeddings().unsqueeze(0)
        return self.input_dropout(self.input_norm(x))

    def encode(self, values, observed_features=None):
        """Return one bin representation without stochastic masking.

        observed_features is a shared boolean [F] schema mask. It lets a
        downstream branch omit biologically unavailable feature groups instead
        of representing missing columns as measured zeros.
        """
        x = self.embed_features(values)
        if observed_features is not None:
            observed_features = torch.as_tensor(
                observed_features, dtype=torch.bool, device=values.device
            )
            if observed_features.ndim != 1 or observed_features.numel() != self.feature_count:
                raise ValueError("observed_features must be a boolean vector of length F")
            if not torch.any(observed_features):
                raise ValueError("at least one feature must be observed")
            x = x[:, observed_features]
        x = torch.cat((self.bin_token.expand(values.shape[0], -1, -1), x), dim=1)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)[:, 0]

    @staticmethod
    def apply_mask(x, ids_shuffle, len_keep):
        batch, length, dim = x.shape
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, dim))
        mask = torch.ones(batch, length, device=x.device)
        mask[:, :len_keep] = 0
        return x, torch.gather(mask, 1, ids_restore), ids_restore

    def random_masking(self, x, mask_ratio):
        batch, length, _ = x.shape
        ids_shuffle = torch.rand(batch, length, device=x.device).argsort(dim=1)
        return self.apply_mask(x, ids_shuffle, int(length * (1 - mask_ratio)))

    def modality_mask_plan(self, mask_ratio):
        """Choose the modality count whose union is closest to mask_ratio."""
        if not self.typed_features:
            raise ValueError("modality masking requires feature_names")
        primary = self.primary_modality_ids.detach().cpu()
        partner = self.partner_modality_ids.detach().cpu()
        target = self.feature_count * mask_ratio
        plans = []
        for count in range(1, self.modality_count):
            selected = torch.zeros(self.modality_count, dtype=torch.bool)
            selected[:count] = True
            related = selected[primary]
            related |= (partner < self.modality_count) & selected[
                partner.clamp(max=self.modality_count - 1)
            ]
            masked = int(related.sum())
            plans.append((abs(masked - target), count, masked))
        _, count, masked = min(plans)
        return count, masked, masked / self.feature_count

    def modality_masking(self, x, mask_ratio):
        batch = x.shape[0]
        selected_count, masked_count, _ = self.modality_mask_plan(mask_ratio)
        choices = torch.rand(batch, self.modality_count, device=x.device).argsort(
            dim=1
        )[:, :selected_count]
        selected = torch.zeros(
            batch, self.modality_count, dtype=torch.bool, device=x.device
        )
        selected.scatter_(1, choices, True)
        primary = self.primary_modality_ids.unsqueeze(0).expand(batch, -1)
        partner = self.partner_modality_ids.unsqueeze(0).expand(batch, -1)
        mask = torch.gather(selected, 1, primary)
        mask |= (partner < self.modality_count) & torch.gather(
            selected, 1, partner.clamp(max=self.modality_count - 1)
        )
        if not torch.all(mask.sum(1) == masked_count):
            raise RuntimeError(
                "Modality groups have unequal sizes; verify all overlap columns"
            )
        noise = torch.rand(batch, self.feature_count, device=x.device)
        ids_shuffle = (mask.float() * 2.0 + noise).argsort(dim=1)
        return self.apply_mask(x, ids_shuffle, self.feature_count - masked_count)

    def forward_encoder(self, values, mask_ratio, mask_strategy):
        x = self.embed_features(values)
        if mask_strategy == "random":
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        elif mask_strategy == "modality":
            x, mask, ids_restore = self.modality_masking(x, mask_ratio)
        else:
            raise ValueError(f"Unknown mask strategy: {mask_strategy}")
        x = torch.cat((self.bin_token.expand(values.shape[0], -1, -1), x), dim=1)
        for block in self.blocks:
            x = block(x)
        return self.norm(x), mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)

        ids = torch.arange(self.feature_count + 1, device=x.device)
        x = x + self.decoder_feature_encoder(ids).unsqueeze(0)
        if self.typed_features:
            none_m = torch.tensor([self.none_modality], device=x.device)
            none_f = torch.tensor([self.none_form], device=x.device)
            primary = torch.cat([none_m, self.primary_modality_ids])
            partner = torch.cat([none_m, self.partner_modality_ids])
            forms = torch.cat([none_f, self.data_form_ids])
            x = x + (
                self.decoder_modality_encoder(primary)
                + self.decoder_partner_modality_encoder(partner)
                + self.decoder_data_form_encoder(forms)
            ).unsqueeze(0)
        x = self.decoder_input_norm(x)
        for block in self.decoder_blocks:
            x = block(x)
        return self.decoder_value(self.decoder_norm(x)[:, 1:, :]).squeeze(-1)

    def forward_loss(self, values, pred, mask):
        target = values.float()
        if self.norm_pix_loss:
            mean = target.mean(1, keepdim=True)
            var = target.var(1, keepdim=True)
            target = (target - mean) / torch.sqrt(var + 1.0e-6)
        return (((pred.float() - target) ** 2) * mask).sum() / mask.sum()

    def forward(self, values, mask_ratio=0.4, mask_strategy="random"):
        latent, mask, ids_restore = self.forward_encoder(
            values, mask_ratio, mask_strategy
        )
        with torch.cuda.amp.autocast(enabled=False):
            pred = self.forward_decoder(latent, ids_restore)
            loss = self.forward_loss(values, pred, mask)
        return latent, loss, pred, mask


class ValueEncoder(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = torch.relu(self.linear1(x.unsqueeze(-1)))
        return self.dropout(self.norm(self.linear2(x)))


def mae_bin_base(**kwargs):
    return MaskedAutoencoderBin(
        hidden_dim=128, embed_dim=512, depth=8, num_heads=8,
        decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )


def mae_bin_large(**kwargs):
    return MaskedAutoencoderBin(
        hidden_dim=128, embed_dim=512, depth=12, num_heads=16,
        decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )


def mae_bin_huge(**kwargs):
    return MaskedAutoencoderBin(
        hidden_dim=128, embed_dim=768, depth=16, num_heads=16,
        decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
