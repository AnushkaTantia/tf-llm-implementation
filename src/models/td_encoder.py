"""
Time-Domain (TD) Encoder for TF-LLM — channel-independent version.

Each variable (channel) is processed independently through SHARED weights
(same pattern as PatchTST, cited by the paper as its patch-size reference).
Variables are never mixed together in the patch projection. Per Section 3.3:
"We ensure these tokens retain the variable dimension d_v from the input
sequence." Learnable positional embeddings added per Section 3.3.

Two outputs:
  - patch_embeddings: (batch, num_patches, num_variables, d_emb) — T_Seq
  - z: (batch, d_emb) — pooled over patches AND variables, used for the
    TD contrastive loss (Section 3.1's C_T_n is a single per-sample embedding)
"""

import random

import torch
import torch.nn as nn


class TDEncoder(nn.Module):
    def __init__(self, patch_len=16, num_features=7, d_emb=64, n_heads=4, n_layers=2, dropout=0.2):
        super().__init__()
        self.patch_len = patch_len
        self.num_features = num_features
        self.d_emb = d_emb

        # shared linear projection: one patch (patch_len values) -> d_emb,
        # applied identically to every variable (channel-independent)
        self.patch_proj = nn.Linear(patch_len, d_emb)

        self.max_patches = 256
        self.pos_embedding = nn.Parameter(torch.randn(1, self.max_patches, 1, d_emb) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_emb, nhead=n_heads, dim_feedforward=d_emb * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_patched):
        """
        Args:
            x_patched: (batch, num_patches, patch_len, num_variables)
        Returns:
            z: (batch, d_emb)
            patch_embeddings: (batch, num_patches, num_variables, d_emb)
        """
        batch, num_patches, patch_len, num_variables = x_patched.shape

        # move variables next to batch so the SAME projection/transformer
        # weights are applied independently per variable
        x = x_patched.permute(0, 3, 1, 2)  # (batch, num_variables, num_patches, patch_len)
        x = x.reshape(batch * num_variables, num_patches, patch_len)

        emb = self.patch_proj(x)
        emb = self.dropout(emb)
        emb = self.transformer(emb)

        emb = emb.reshape(batch, num_variables, num_patches, self.d_emb)
        emb = emb.permute(0, 2, 1, 3)  # (batch, num_patches, num_variables, d_emb)
        emb = emb + self.pos_embedding[:, :num_patches, :, :]

        patch_embeddings = emb
        z = patch_embeddings.mean(dim=(1, 2))  # pooled for contrastive loss

        return z, patch_embeddings


def jitter(x, sigma=0.03):
    return x + torch.randn_like(x) * sigma


def scaling(x, sigma=0.1):
    factor = 1.0 + torch.randn(x.shape[0], 1, 1, x.shape[-1], device=x.device) * sigma
    return x * factor


def time_masking(x, mask_ratio=0.1):
    x = x.clone()
    batch, num_patches = x.shape[0], x.shape[1]
    mask_len = max(1, int(num_patches * mask_ratio))
    for b in range(batch):
        start = random.randint(0, max(0, num_patches - mask_len))
        x[b, start:start + mask_len] = 0.0
    return x


def apply_td_augmentation(x_patched):
    aug_fn = random.choice([jitter, scaling, time_masking])
    return aug_fn(x_patched)
