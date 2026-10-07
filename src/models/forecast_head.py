"""
Forecasting task head for TF-LLM — channel-independent version.

T_Mask is duplicated across BOTH the forecast-patch dimension AND the
variable dimension (num_mask_patches * num_variables mask tokens total).
The Mask Block decodes each mask token's hidden state into one patch's
worth of a SINGLE variable (channel-independent decoding), matching the
channel-independent encoders.

Eq. 7: Z_pred = concat(T_Pro, T_Seq, dup(T_Mask, l_f))
Mask Block structure per Fig. 5: MLP -> Linear -> Fold.

Input/output RevIN (Fig. 2: "RevIN & Patching" on the input path, "RevIN"
after the output projection; Kim et al., 2022): each input window is
normalized per variable by its own mean/std before encoding, and the
forecast is de-normalized with the same statistics. This is separate from
the RevIN on spectral magnitude inside fd_encoder.py. use_revin=False
restores the previous behaviour (needed to load checkpoints trained
before this change).
"""

import torch
import torch.nn as nn


class InstanceRevIN(nn.Module):
    """Reversible instance normalization over the time axis, per variable."""

    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def stats(self, x_raw):
        # x_raw: (batch, seq_len, num_variables)
        mean = x_raw.mean(dim=1, keepdim=True).detach()
        std = torch.sqrt(x_raw.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        return mean, std  # each (batch, 1, num_variables)

    def norm(self, x, mean, std):
        # works for (B, L, C) and, via broadcasting, (B, num_patches, patch_len, C)
        if x.dim() == 4:
            mean, std = mean.unsqueeze(1), std.unsqueeze(1)
        x = (x - mean) / std
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def denorm(self, y, mean, std):
        # y: (batch, pred_len, num_variables)
        if self.affine:
            y = (y - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
        return y * std + mean


class MaskBlock(nn.Module):
    def __init__(self, hidden_size, patch_len, dropout=0.1):
        super().__init__()
        self.patch_len = patch_len

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # shared across variables: decodes to ONE patch of a single variable
        self.linear = nn.Linear(hidden_size, patch_len)

    def forward(self, mask_hidden_states):
        batch, num_mask_patches, num_variables, hidden_size = mask_hidden_states.shape

        x = self.mlp(mask_hidden_states)
        x = self.linear(x)  # (B, num_mask_patches, v, patch_len)

        # Fold: reorder so patch positions + their timesteps sit together,
        # then flatten into the full forecast horizon per variable
        x = x.permute(0, 1, 3, 2)  # (B, num_mask_patches, patch_len, v)
        forecast = x.reshape(batch, num_mask_patches * self.patch_len, num_variables)

        return forecast


class TFLLMForecaster(nn.Module):
    def __init__(self, td_encoder, fd_encoder, backbone, pred_len=96, patch_len=16, num_features=7,
                 use_revin=True):
        super().__init__()
        self.td_encoder = td_encoder
        self.fd_encoder = fd_encoder
        self.backbone = backbone

        self.pred_len = pred_len
        self.patch_len = patch_len
        self.num_features = num_features

        assert pred_len % patch_len == 0
        self.num_mask_patches = pred_len // patch_len

        # T_Mask: a single learnable mask token, duplicated across patches and variables
        self.mask_token = nn.Parameter(torch.randn(backbone.hidden_size) * 0.02)

        self.mask_block = MaskBlock(hidden_size=backbone.hidden_size, patch_len=patch_len)

        self.use_revin = use_revin
        if use_revin:
            self.revin = InstanceRevIN(num_features)

    def forward(self, x_patched, x_raw):
        """
        Args:
            x_patched: (batch, num_patches, patch_len, num_variables)
            x_raw:     (batch, seq_len, num_variables)
        Returns:
            forecast: (batch, pred_len, num_variables)  -- in the dataset's (z-scored) space
            aux: dict with intermediate embeddings for the TFB losses, and the
                 inputs actually fed to the encoders (normalized if use_revin)
        """
        batch_size = x_patched.shape[0]
        num_variables = x_patched.shape[-1]

        if self.use_revin:
            mean, std = self.revin.stats(x_raw)
            x_raw = self.revin.norm(x_raw, mean, std)
            x_patched = self.revin.norm(x_patched, mean, std)

        z_T, td_patch_embeddings = self.td_encoder(x_patched)
        z_F, _ = self.fd_encoder(x_raw)
        fd_pooled_per_var = self.fd_encoder.pooled_per_variable(x_raw)

        seq_tokens = self.backbone.build_sequence_tokens(td_patch_embeddings, fd_pooled_per_var)
        prompt_tokens = self.backbone.get_prompt_tokens(batch_size)

        num_mask_tokens = self.num_mask_patches * num_variables
        mask_tokens = self.mask_token.unsqueeze(0).unsqueeze(0).expand(
            batch_size, num_mask_tokens, -1
        )

        full_input = torch.cat([prompt_tokens, seq_tokens, mask_tokens], dim=1)

        hidden_states = self.backbone.forward_backbone(full_input)

        mask_hidden_states = hidden_states[:, -num_mask_tokens:, :]
        mask_hidden_states = mask_hidden_states.reshape(
            batch_size, self.num_mask_patches, num_variables, self.backbone.hidden_size
        )

        forecast = self.mask_block(mask_hidden_states)

        if self.use_revin:
            forecast = self.revin.denorm(forecast, mean, std)

        aux = {"z_T": z_T, "z_F": z_F, "td_patch_embeddings": td_patch_embeddings,
               "x_patched_in": x_patched, "x_raw_in": x_raw}

        return forecast, aux
