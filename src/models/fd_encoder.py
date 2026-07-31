"""
Frequency-Domain (FD) Encoder for TF-LLM.

Implements the FD branch described in Section 3.1 of the paper:
    - Real FFT (torch.fft.rfft) to obtain spectral representation
    - RevIN normalization applied to spectral magnitude (Kim et al., 2022)
    - Linear projection into embedding space (d_emb=64)
    - Frequency-domain augmentation: phase perturbation + amplitude
      zeroing/boosting on randomly selected frequency bands

Operates on the *raw* (unpatched) input sequence, since FFT needs the full
time series to extract meaningful frequency components.
"""

import random

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """
    Reversible Instance Normalization, applied here to spectral magnitude
    rather than raw time series (per TF-LLM paper's usage).

    Normalizes per-sample (per instance) using its own mean/std, with
    learnable affine parameters. Standardizes spectra with different
    global characteristics into a comparable distribution.
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
        # x: (batch, freq_bins, num_features)
        mean = x.mean(dim=1, keepdim=True)
        std = (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt()

        x_norm = (x - mean) / std
        if self.affine:
            x_norm = x_norm * self.affine_weight + self.affine_bias
        return x_norm


class FDEncoder(nn.Module):
    """
    Converts a raw time series into a frequency-domain embedding.

    Input:  x of shape (batch, seq_len, num_features)
    Output: z of shape (batch, d_emb)
    """

    def __init__(self, seq_len=512, num_features=7, d_emb=64, n_heads=4, n_layers=2, dropout=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.d_emb = d_emb

        # rfft output length for a real signal of length seq_len
        self.freq_bins = seq_len // 2 + 1

        self.revin = RevIN(num_features)

        # project each frequency bin's per-feature magnitude to d_emb,
        # then use a small transformer over frequency bins (analogous to
        # the TD encoder's transformer over patches)
        self.freq_proj = nn.Linear(num_features, d_emb)

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
            x: (batch, seq_len, num_features) — raw (unpatched) input sequence
        Returns:
            z: (batch, d_emb)
            freq_embeddings: (batch, freq_bins, d_emb)
        """
        # real FFT along the time dimension
        spectrum = torch.fft.rfft(x, dim=1)          # complex, (batch, freq_bins, num_features)
        magnitude = spectrum.abs()                     # (batch, freq_bins, num_features)

        magnitude = self.revin(magnitude)

        freq_embeddings = self.freq_proj(magnitude)    # (batch, freq_bins, d_emb)
        freq_embeddings = self.dropout(freq_embeddings)

        freq_embeddings = self.transformer(freq_embeddings)  # (batch, freq_bins, d_emb)

        z = freq_embeddings.mean(dim=1)                 # (batch, d_emb)

        return z, freq_embeddings


# ---------------------------------------------------------------------------
# Frequency-domain augmentations
# ---------------------------------------------------------------------------

def phase_perturbation(x, max_shift=0.1):
    """
    Perturb the phase of the FFT spectrum by a small random amount,
    then convert back to time domain. x: (batch, seq_len, num_features).
    """
    spectrum = torch.fft.rfft(x, dim=1)
    magnitude = spectrum.abs()
    phase = spectrum.angle()

    noise = (torch.rand_like(phase) - 0.5) * 2 * max_shift * torch.pi
    phase_perturbed = phase + noise

    spectrum_perturbed = torch.polar(magnitude, phase_perturbed)
    x_aug = torch.fft.irfft(spectrum_perturbed, n=x.shape[1], dim=1)
    return x_aug


def amplitude_modification(x, num_bands=5, boost_factor=1.5):
    """
    Randomly zero out some frequency bands and boost others,
    then convert back to time domain. x: (batch, seq_len, num_features).
    """
    spectrum = torch.fft.rfft(x, dim=1)
    magnitude = spectrum.abs()
    phase = spectrum.angle()

    batch, freq_bins, num_features = magnitude.shape
    magnitude = magnitude.clone()

    for b in range(batch):
        # zero out num_bands random bins
        zero_idx = torch.randperm(freq_bins)[:num_bands]
        magnitude[b, zero_idx] = 0.0

        # boost num_bands random low-amplitude bins
        boost_idx = torch.randperm(freq_bins)[:num_bands]
        magnitude[b, boost_idx] = magnitude[b, boost_idx] * boost_factor

    spectrum_aug = torch.polar(magnitude, phase)
    x_aug = torch.fft.irfft(spectrum_aug, n=x.shape[1], dim=1)
    return x_aug


def apply_fd_augmentation(x):
    """Randomly select one frequency-domain augmentation."""
    aug_fn = random.choice([phase_perturbation, amplitude_modification])
    return aug_fn(x)


if __name__ == "__main__":
    # quick smoke test
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
    print(f"x_aug:       {tuple(x_aug.shape)}")
    print(f"z:           {tuple(z.shape)}")
    print(f"freq_emb:    {tuple(freq_emb.shape)}")
    print(f"z_aug:       {tuple(z_aug.shape)}")

    from contrastive import nt_xent_loss  # only works if run from src/models with losses on path
    loss = nt_xent_loss(z, z_aug)
    print(f"NT-Xent loss: {loss.item():.4f}")
