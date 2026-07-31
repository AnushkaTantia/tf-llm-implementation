"""
Time-Frequency Balance (TFB) Module for TF-LLM.

Implements Section 3.1's TFB module:
    - Projection functions P_T, P_F mapping TD and FD embeddings into a
      shared time-frequency space
    - A balance loss pulling matched time/frequency embeddings (same
      underlying sample) together, while keeping mismatched
      (cross-augmentation) pairs apart
    - Combined loss: L_TFB = lambda * (L_T + L_F) + (1 - lambda) * L_B

IMPLEMENTATION NOTE (deviation from paper, documented for transparency):
The paper's Eq. 3 for the balance loss, as extracted from the PDF, reads as
summing (epsilon + |D_TF - D_pair|) across negative pairs. As written this
does not produce a gradient that consistently favors matched pairs over
mismatched ones (no margin/hinge structure), which is likely a PDF
extraction artifact rather than the authors' actual intended formula. We
implement the *described behavior* instead: a standard margin-based triplet
loss that pulls the matched (z_T, z_F) pair together and pushes mismatched
augmented combinations apart by at least a margin. This is the standard,
well-established way to achieve the stated goal ("ensuring the
time-frequency distance is smaller than that of negative samples").
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """Small MLP projecting an encoder embedding into the shared TF space."""

    def __init__(self, d_emb=64, proj_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_emb, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, z):
        return self.net(z)


class TFBModule(nn.Module):
    """
    Wraps the two projection heads (P_T, P_F) and exposes the balance loss.
    """

    def __init__(self, d_emb=64, proj_dim=128, dropout=0.1):
        super().__init__()
        self.proj_T = ProjectionHead(d_emb, proj_dim, dropout)
        self.proj_F = ProjectionHead(d_emb, proj_dim, dropout)

    def forward(self, z_T, z_F):
        """
        Args:
            z_T: (batch, d_emb) — TD encoder output (original, not augmented)
            z_F: (batch, d_emb) — FD encoder output (original, not augmented)
        Returns:
            p_T, p_F: (batch, proj_dim) each — projected into shared space
        """
        p_T = self.proj_T(z_T)
        p_F = self.proj_F(z_F)
        return p_T, p_F


def _distance(a, b):
    """1 - cosine similarity, so identical vectors have distance 0."""
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return 1.0 - (a * b).sum(dim=-1)


def balance_loss(p_T, p_T_aug, p_F, p_F_aug, margin=0.2):
    """
    Margin-based triplet loss over time-frequency pairs.

    Positive pair:   (p_T, p_F)              — same underlying sample, matched domains
    Negative pairs:  (p_T, p_F_aug)           — matched sample, mismatched augmentation
                      (p_T_aug, p_F)           — matched sample, mismatched augmentation
                      (p_T_aug, p_F_aug)       — both augmented, weakest match

    For each negative pair, we penalize cases where the positive distance
    is not at least `margin` smaller than the negative distance:
        loss = relu(D_positive - D_negative + margin)

    Returns:
        scalar loss, averaged over batch and over the three negative pairs
    """
    d_pos = _distance(p_T, p_F)                  # (batch,)
    d_neg1 = _distance(p_T, p_F_aug)
    d_neg2 = _distance(p_T_aug, p_F)
    d_neg3 = _distance(p_T_aug, p_F_aug)

    loss1 = F.relu(d_pos - d_neg1 + margin)
    loss2 = F.relu(d_pos - d_neg2 + margin)
    loss3 = F.relu(d_pos - d_neg3 + margin)

    loss = (loss1 + loss2 + loss3) / 3.0
    return loss.mean()


def tfb_total_loss(l_t, l_f, l_b, lam=0.5):
    """
    Combined TFB loss per Eq. 4: L_TFB = lambda*(L_T + L_F) + (1-lambda)*L_B

    Args:
        l_t: scalar, TD contrastive loss (NT-Xent)
        l_f: scalar, FD contrastive loss (NT-Xent)
        l_b: scalar, balance loss
        lam: weighting hyperparameter (paper doesn't specify exact value used
             in main results; 0.5 is a neutral default, treated as tunable)
    """
    return lam * (l_t + l_f) + (1 - lam) * l_b


if __name__ == "__main__":
    # quick smoke test — end-to-end TD + FD + TFB
    import sys
    sys.path.append("../losses")
    sys.path.append(".")

    from td_encoder import TDEncoder, apply_td_augmentation
    from fd_encoder import FDEncoder, apply_fd_augmentation
    from contrastive import nt_xent_loss

    batch_size = 4
    seq_len = 512
    patch_len = 16
    num_features = 7
    d_emb = 64

    x = torch.randn(batch_size, seq_len, num_features)
    x_patched = x.reshape(batch_size, seq_len // patch_len, patch_len, num_features)

    x_aug = apply_fd_augmentation(x)  # reuse for TD input too, just for the smoke test
    x_patched_aug = apply_td_augmentation(x_patched)

    td_encoder = TDEncoder(patch_len=patch_len, num_features=num_features, d_emb=d_emb)
    fd_encoder = FDEncoder(seq_len=seq_len, num_features=num_features, d_emb=d_emb)
    tfb = TFBModule(d_emb=d_emb, proj_dim=128)

    z_T, _ = td_encoder(x_patched)
    z_T_aug, _ = td_encoder(x_patched_aug)
    z_F, _ = fd_encoder(x)
    z_F_aug, _ = fd_encoder(x_aug)

    l_t = nt_xent_loss(z_T, z_T_aug)
    l_f = nt_xent_loss(z_F, z_F_aug)

    p_T, p_F = tfb(z_T, z_F)
    p_T_aug, p_F_aug = tfb(z_T_aug, z_F_aug)

    l_b = balance_loss(p_T, p_T_aug, p_F, p_F_aug)

    l_total = tfb_total_loss(l_t, l_f, l_b, lam=0.5)

    print(f"z_T:     {tuple(z_T.shape)}")
    print(f"z_F:     {tuple(z_F.shape)}")
    print(f"p_T:     {tuple(p_T.shape)}")
    print(f"p_F:     {tuple(p_F.shape)}")
    print(f"L_T:     {l_t.item():.4f}")
    print(f"L_F:     {l_f.item():.4f}")
    print(f"L_B:     {l_b.item():.4f}")
    print(f"L_TFB:   {l_total.item():.4f}")
