"""
Two published forecasters used to test RAFT as a plug-in objective (Stage P):

  PatchTST  (Nie et al., ICLR 2023)      channel-independent patch transformer, RevIN, flatten head
  iTransformer (Liu et al., ICLR 2024)   one token per variable, attention ACROSS variables (channel mixing)

Re-implemented after the official code (PatchTST_supervised: BatchNorm in the encoder, learnable positional
encoding, end padding + stride; iTransformer: per-variate normalisation, linear series embedding, LayerNorm
encoder, linear projector). Time-feature covariate tokens of iTransformer are not used (no timestamps are
fed to any model in this study).

Both return (forecast, aux) with the same interface as the forecaster F, so the same training loop, metrics and
RAFT losses apply. aux["h"] is exactly the representation the forecasting head reads, one row per instance
(= one variable of one window): PatchTST (B*V, P*d), iTransformer (B*V, d). aux["mean"], aux["std"]: (B, 1, V)
normalisation statistics of the input window, used to express the futures in the model's space.
"""
import math
import torch
import torch.nn as nn


def _stats(x, eps=1e-5):
    mean = x.mean(1, keepdim=True).detach()
    std = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + eps).detach()
    return mean, std


class _BN(nn.Module):
    """BatchNorm over the model dimension for (N, T, d) tensors (PatchTST's default 'BatchNorm')."""

    def __init__(self, d):
        super().__init__()
        self.bn = nn.BatchNorm1d(d)

    def forward(self, x):
        return self.bn(x.transpose(1, 2)).transpose(1, 2)


class _TSTLayer(nn.Module):
    def __init__(self, d, heads, d_ff, dropout, attn_dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, dropout=attn_dropout, batch_first=True)
        self.drop1, self.norm1 = nn.Dropout(dropout), _BN(d)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d))
        self.drop2, self.norm2 = nn.Dropout(dropout), _BN(d)

    def forward(self, x):
        x = self.norm1(x + self.drop1(self.attn(x, x, x, need_weights=False)[0]))
        return self.norm2(x + self.drop2(self.ff(x)))


class PatchTST(nn.Module):
    def __init__(self, seq_len, pred_len, n_vars, patch_len=16, stride=8, d_model=16, n_heads=4, e_layers=3,
                 d_ff=128, dropout=0.3, fc_dropout=0.3, head_dropout=0.0, revin_affine=True):
        super().__init__()
        self.seq_len, self.pred_len, self.V = seq_len, pred_len, n_vars
        self.patch_len, self.stride = patch_len, stride
        self.n_patches = (seq_len - patch_len) // stride + 1 + 1          # +1: end padding (padding_patch='end')
        self.affine = revin_affine
        if revin_affine:
            self.rw = nn.Parameter(torch.ones(n_vars)); self.rb = nn.Parameter(torch.zeros(n_vars))
        self.W_P = nn.Linear(patch_len, d_model)
        self.W_pos = nn.Parameter(torch.empty(self.n_patches, d_model).uniform_(-0.02, 0.02))
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([_TSTLayer(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)])
        self.head_drop = nn.Dropout(head_dropout)
        self.head = nn.Linear(self.n_patches * d_model, pred_len)
        self.d_model = d_model

    def forward(self, x_patched_unused, x):
        B, L, V = x.shape
        mean, std = _stats(x)
        z = (x - mean) / std
        if self.affine:
            z = z * self.rw + self.rb
        z = z.permute(0, 2, 1)                                                 # (B, V, L)
        z = torch.cat([z, z[:, :, -1:].repeat(1, 1, self.stride)], dim=-1)    # end padding
        p = z.unfold(-1, self.patch_len, self.stride)                          # (B, V, P, patch_len)
        u = self.W_P(p).reshape(B * V, self.n_patches, -1) + self.W_pos
        u = self.drop(u)
        for lyr in self.layers:
            u = lyr(u)
        h = u.reshape(B * V, -1)                                               # what the head reads
        out = self.head(self.head_drop(h)).reshape(B, V, self.pred_len).permute(0, 2, 1)
        if self.affine:
            out = (out - self.rb) / (self.rw + 1e-10)
        out = out * std + mean
        return out, {"h": h, "mean": mean, "std": std}


class _EncLayer(nn.Module):
    def __init__(self, d, heads, d_ff, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.n1(x + self.drop(self.attn(x, x, x, need_weights=False)[0]))
        return self.n2(x + self.drop(self.ff(x)))


class ITransformer(nn.Module):
    def __init__(self, seq_len, pred_len, n_vars, d_model=128, n_heads=8, e_layers=2, d_ff=128, dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.embed = nn.Linear(seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([_EncLayer(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, pred_len)

    def forward(self, x_patched_unused, x):
        B, L, V = x.shape
        mean, std = _stats(x)
        z = ((x - mean) / std).permute(0, 2, 1)                                # (B, V, L): one token per variable
        t = self.drop(self.embed(z))
        for lyr in self.layers:
            t = lyr(t)
        t = self.norm(t)                                                       # (B, V, d)
        h = t.reshape(B * V, -1)                                               # what the projector reads
        out = self.proj(t).permute(0, 2, 1) * std + mean
        return out, {"h": h, "mean": mean, "std": std}


# Official per-dataset settings (PatchTST_supervised scripts; iTransformer scripts). Look-back: each model's own.
def config(backbone, dataset):
    if backbone == "patchtst":
        if dataset == "ILI":
            return dict(seq_len=104, kw=dict(patch_len=24, stride=2, d_model=16, n_heads=4, e_layers=3, d_ff=128,
                                             dropout=0.3, fc_dropout=0.3), lr=2.5e-3, batch=16)
        if dataset in ("ETTh1", "ETTh2", "Exchange"):
            return dict(seq_len=336, kw=dict(d_model=16, n_heads=4, e_layers=3, d_ff=128, dropout=0.3, fc_dropout=0.3),
                        lr=1e-4, batch=128)
        return dict(seq_len=336, kw=dict(d_model=128, n_heads=16, e_layers=3, d_ff=256, dropout=0.2, fc_dropout=0.2),
                    lr=1e-4, batch=128)                                       # ETTm1, ETTm2, Weather
    if backbone == "itransformer":
        L = 36 if dataset == "ILI" else 96
        if dataset == "ETTh1":
            kw = dict(d_model=256, d_ff=256, e_layers=2)
        elif dataset == "Weather":
            kw = dict(d_model=512, d_ff=512, e_layers=3)
        else:
            kw = dict(d_model=128, d_ff=128, e_layers=2)
        return dict(seq_len=L, kw=kw, lr=1e-4, batch=32)
    raise ValueError(backbone)


def build(backbone, dataset, pred_len, n_vars):
    c = config(backbone, dataset)
    if backbone == "patchtst":
        kw = dict(c["kw"]); kw.pop("fc_dropout", None)
        return PatchTST(c["seq_len"], pred_len, n_vars, **kw), c
    return ITransformer(c["seq_len"], pred_len, n_vars, **c["kw"]), c
