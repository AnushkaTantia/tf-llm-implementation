"""
Time-Domain (TD) Encoder for TF-LLM.

Implements the TD branch described in Section 3.1 of the paper:
    - Linear projection of patches into embedding space (d_emb=64)
    - Time-domain augmentation library (jitter, scaling, time masking)
    - Encoder E_T producing embeddings z^T = E_T(x^T)

The augmentation choice is not fully specified in the paper (it references
"a time-based augmentation mechanism library" without listing exact
transforms), so we use standard, well-established time series augmentations
common in contrastive time series literature (e.g., TF-C).
"""

import random

import torch
import torch.nn as nn


class TDEncoder(nn.Module):
    """
    Projects patched time series into a shared embedding space, then
    encodes with a small transformer to capture intra-patch and
    inter-patch dependencies.

    Input:  x_patched of shape (batch, num_patches, patch_len, num_features)
    Output: z of shape (batch, d_emb) — one embedding per sample,
            obtained by mean-pooling over patches.
    """

    def __init__(self, patch_len=16, num_features=7, d_emb=64, n_heads=4, n_layers=2, dropout=0.2):
        super().__init__()
        self.patch_len = patch_len
        self.num_features = num_features
        self.d_emb = d_emb

        # flatten each patch (patch_len * num_features) and linearly project to d_emb
        self.patch_proj = nn.Linear(patch_len * num_features, d_emb)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_emb,
            nhead=n_heads,
            dim_feedforward=d_emb * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_patched):
        """
        Args:
            x_patched: (batch, num_patches, patch_len, num_features)
        Returns:
            z: (batch, d_emb)
            patch_embeddings: (batch, num_patches, d_emb) — useful later for
                               feeding into the LLM as sequence tokens
        """
        batch, num_patches, patch_len, num_features = x_patched.shape

        flat = x_patched.reshape(batch, num_patches, patch_len * num_features)
        patch_embeddings = self.patch_proj(flat)          # (batch, num_patches, d_emb)
        patch_embeddings = self.dropout(patch_embeddings)

        patch_embeddings = self.transformer(patch_embeddings)  # (batch, num_patches, d_emb)

        z = patch_embeddings.mean(dim=1)                   # (batch, d_emb) — pooled representation

        return z, patch_embeddings


# ---------------------------------------------------------------------------
# Time-domain augmentations
# ---------------------------------------------------------------------------

def jitter(x, sigma=0.03):
    """Add small Gaussian noise. x: (..., seq_len, num_features) or patched shape."""
    return x + torch.randn_like(x) * sigma


def scaling(x, sigma=0.1):
    """Multiply by a random scalar close to 1.0, per-sample."""
    # scale factor shape broadcasts across time/patch dims, one factor per sample+feature
    factor = 1.0 + torch.randn(x.shape[0], 1, 1, x.shape[-1], device=x.device) * sigma
    if x.dim() == 3:
        factor = factor.squeeze(1)
    return x * factor


def time_masking(x, mask_ratio=0.1):
    """
    Randomly zero out a contiguous segment along the time dimension.
    Works on patched input: (batch, num_patches, patch_len, num_features).
    """
    x = x.clone()
    batch, num_patches = x.shape[0], x.shape[1]
    mask_len = max(1, int(num_patches * mask_ratio))
    for b in range(batch):
        start = random.randint(0, max(0, num_patches - mask_len))
        x[b, start:start + mask_len] = 0.0
    return x


def apply_td_augmentation(x_patched):
    """
    Randomly select one augmentation to apply. x_patched shape:
    (batch, num_patches, patch_len, num_features).
    """
    aug_fn = random.choice([jitter, scaling, time_masking])
    return aug_fn(x_patched)


if __name__ == "__main__":
    # quick smoke test
    batch_size = 4
    num_patches = 32
    patch_len = 16
    num_features = 7
    d_emb = 64

    x_patched = torch.randn(batch_size, num_patches, patch_len, num_features)
    x_patched_aug = apply_td_augmentation(x_patched)

    encoder = TDEncoder(patch_len=patch_len, num_features=num_features, d_emb=d_emb)

    z, patch_emb = encoder(x_patched)
    z_aug, _ = encoder(x_patched_aug)

    print(f"x_patched:     {tuple(x_patched.shape)}")
    print(f"z:             {tuple(z.shape)}")
    print(f"patch_emb:     {tuple(patch_emb.shape)}")
    print(f"z_aug:         {tuple(z_aug.shape)}")

    from contrastive import nt_xent_loss  # only works if run from src/models with losses on path
    loss = nt_xent_loss(z, z_aug)
    print(f"NT-Xent loss:  {loss.item():.4f}")
