"""
Frequency-Domain (FD) Encoder for TF-LLM.

Each variable's spectrum is projected and encoded with
SHARED weights, independently of other variables.
"""

import random

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """
    Reversible Instance Normalization applied to spectral magnitude,
    per-variable (affine params indexed by variable), matching the
    paper's usage (Kim et al., 2022, applied to spectral features here).
    """

    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        # x: (batch, freq_bins, num_variables)
        mean = x.mean(dim=1, keepdim=True)
        std = (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt()
        x_norm = (x - mean) / std
        if self.affine:
            x_norm = x_norm * self.affine_weight + self.affine_bias
        return x_norm


class FDEncoder(nn.Module):
    """
    Channel-independent frequency-domain encoder.

    Input:  x of shape (batch, seq_len, num_variables)
    Output: z (batch, d_emb), freq_embeddings (batch, freq_bins, num_variables, d_emb)
    """

    def __init__(self, seq_len=512, num_features=7, d_emb=64, n_heads=4, n_layers=2, dropout=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.d_emb = d_emb
        self.freq_bins = seq_len // 2 + 1

        self.revin = RevIN(num_features)

        # shared linear projection: one scalar magnitude value -> d_emb,
        # applied identically to every variable (channel-independent)
        self.freq_proj = nn.Linear(1, d_emb)

        self.pos_embedding = nn.Parameter(torch.randn(1, self.freq_bins, 1, d_emb) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_emb,
            nhead=n_heads,
            dim_feedforward=d_emb * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, num_variables) — raw (unpatched) input sequence
        Returns:
            z: (batch, d_emb) — pooled over freq bins and variables
            freq_embeddings: (batch, freq_bins, num_variables, d_emb)
        """
        batch, seq_len, num_variables = x.shape

        spectrum = torch.fft.rfft(x, dim=1)     # (batch, freq_bins, num_variables), complex
        magnitude = spectrum.abs()                # (batch, freq_bins, num_variables)

        magnitude = self.revin(magnitude)

        # move variables next to batch for channel-independent processing
        mag = magnitude.permute(0, 2, 1)           # (batch, num_variables, freq_bins)
        mag = mag.reshape(batch * num_variables, self.freq_bins, 1)

        emb = self.freq_proj(mag)                  # (batch*num_variables, freq_bins, d_emb)
        emb = self.dropout(emb)

        emb = self.transformer(emb)                # (batch*num_variables, freq_bins, d_emb)

        emb = emb.reshape(batch, num_variables, self.freq_bins, self.d_emb)
        emb = emb.permute(0, 2, 1, 3)               # (batch, freq_bins, num_variables, d_emb)
        emb = emb + self.pos_embedding

        freq_embeddings = emb

        z = freq_embeddings.mean(dim=(1, 2))        # (batch, d_emb)

        return z, freq_embeddings

    def pooled_per_variable(self, x):
        """
        Returns a per-variable pooled embedding (batch, num_variables, d_emb),
        pooling only over frequency bins (not variables). Used by the
        backbone's fusion step, which needs to broadcast FD context into
        each variable's TD patches without collapsing the variable axis.
        """
        _, freq_embeddings = self.forward(x)
        return freq_embeddings.mean(dim=1)  # (batch, num_variables, d_emb)


# ---------------------------------------------------------------------------
# Frequency-domain augmentations
# ---------------------------------------------------------------------------

def phase_perturbation(x, max_shift=0.1):
    spectrum = torch.fft.rfft(x, dim=1)
    magnitude = spectrum.abs()
    phase = spectrum.angle()

    noise = (torch.rand_like(phase) - 0.5) * 2 * max_shift * torch.pi
    phase_perturbed = phase + noise

    spectrum_perturbed = torch.polar(magnitude, phase_perturbed)
    x_aug = torch.fft.irfft(spectrum_perturbed, n=x.shape[1], dim=1)
    return x_aug


def amplitude_modification(x, num_bands=5, boost_factor=1.5):
    spectrum = torch.fft.rfft(x, dim=1)
    magnitude = spectrum.abs()
    phase = spectrum.angle()

    batch, freq_bins, num_features = magnitude.shape
    magnitude = magnitude.clone()

    for b in range(batch):
        zero_idx = torch.randperm(freq_bins)[:num_bands]
        magnitude[b, zero_idx] = 0.0

        boost_idx = torch.randperm(freq_bins)[:num_bands]
        magnitude[b, boost_idx] = magnitude[b, boost_idx] * boost_factor

    spectrum_aug = torch.polar(magnitude, phase)
    x_aug = torch.fft.irfft(spectrum_aug, n=x.shape[1], dim=1)
    return x_aug


def apply_fd_augmentation(x):
    aug_fn = random.choice([phase_perturbation, amplitude_modification])
    return aug_fn(x)


if __name__ == "__main__":
    batch_size = 4
    seq_len = 512
    num_features = 7
    d_emb = 64

    x = torch.randn(batch_size, seq_len, num_features)
    x_aug = apply_fd_augmentation(x)

    encoder = FDEncoder(seq_len=seq_len, num_features=num_features, d_emb=d_emb)

    z, freq_emb = encoder(x)
    z_aug, _ = encoder(x_aug)

    print(f"x:           {tuple(x.shape)}")
    print(f"z:           {tuple(z.shape)}")
    print(f"freq_emb:    {tuple(freq_emb.shape)}  (batch, freq_bins, num_variables, d_emb)")
    print(f"z_aug:       {tuple(z_aug.shape)}")

    from contrastive import nt_xent_loss
    loss = nt_xent_loss(z, z_aug)
    print(f"NT-Xent loss: {loss.item():.4f}")
