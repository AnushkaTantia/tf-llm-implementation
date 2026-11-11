"""
RAFT: Relational Alignment of Frequency and Time with conflict-free gradients.

Proposed replacement for the time-frequency balance (TFB) objective of Zhang et al. (Neural Networks 199,
108687, 2026). Each component answers a failure diagnosed in Stage R1:

  (1) L_RT  target-aware relational alignment.  R1: the instance-discrimination term L_T made the
      time-domain representations of different windows nearly orthogonal (cosine distance 0.126 -> 0.971)
      and caused 94% of the harm. L_RT instead asks the representation the forecasting head reads to
      reproduce the similarity structure of the windows' FUTURES: windows with similar futures are pulled
      together in proportion to that similarity; nothing is pushed to orthogonality.
  (2) Informative frequency branch.  R1: the FD embedding was nearly constant across windows (cosine
      distance 1.5e-5). RAFT encodes log(1 + |rFFT|) normalised with batch statistics (BatchNorm over
      variables), instead of per-window RevIN on the magnitude, and computes the FD branch once per step.
  (3) L_TF  relational time-frequency consistency.  Replaces the triplet balance loss: the TD and FD
      representations must induce the same within-batch similarity structure (they are not forced to be
      the same vectors).
  (4) Conflict-free auxiliary gradient.  g = g_task + alpha * P(g_aux), with
          P(g_aux) = g_aux - min(0, <g_aux, g_task>) / ||g_task||^2 * g_task.
      Proposition: <g, g_task> >= ||g_task||^2, i.e. the update direction handed to the optimiser is a
      descent direction for L_task to first order, for any alpha >= 0 (proof in the paper; checked in
      tests/test_raft.py).

Everything here is new code; the baseline forecaster F (src/models/*, src/train.py) is unchanged, so the
no-auxiliary and TFB baselines run exactly the code of Stages S2-R1.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from fd_encoder import FDEncoder
from forecast_head import TFLLMForecaster


# ---------------------------------------------------------------------------------------------- FD branch
class RAFTFDEncoder(FDEncoder):
    """FD encoder with log-magnitude features normalised by batch statistics (BatchNorm over variables)."""

    def __init__(self, seq_len=512, num_features=7, d_emb=64, **kw):
        super().__init__(seq_len=seq_len, num_features=num_features, d_emb=d_emb, **kw)
        self.spec_norm = nn.BatchNorm1d(num_features)       # channels = variables; stats over (batch, bins)

    def forward(self, x):
        batch, seq_len, V = x.shape
        assert seq_len == self.seq_len, f"seq_len {seq_len} != {self.seq_len}"
        mag = torch.log1p(torch.fft.rfft(x, dim=1).abs())     # (B, bins, V)
        mag = self.spec_norm(mag.permute(0, 2, 1))             # (B, V, bins)
        mag = mag.reshape(batch * V, self.freq_bins, 1)
        emb = self.freq_proj(mag) + self.pos_embedding[0, :, 0, :].unsqueeze(0)
        emb = self.transformer(self.dropout(emb))
        emb = emb.reshape(batch, V, self.freq_bins, self.d_emb).permute(0, 2, 1, 3)   # (B, bins, V, d)
        return emb.mean(dim=(1, 2)), emb

    def pooled_per_variable(self, x):
        return self.forward(x)[1].mean(dim=1)


class RAFTForecaster(TFLLMForecaster):
    """
    Same forecaster as F (flatten head, no LLM), except that the FD branch is evaluated ONCE per forward
    pass (F evaluates it twice) and the representations used by RAFT are returned in `aux`:
      h_td : (B*V, P*d) per-variable time-domain patch representation, exactly what the head reads (before fusion)
      h_fd : (B*V, d)   per-variable pooled frequency-domain representation
      mean, std : RevIN statistics of the input window (B, 1, V), to express targets in the model's space
    """

    def forward(self, x_patched, x_raw):
        B, V = x_patched.shape[0], x_patched.shape[-1]
        assert self.use_revin and self.head_type == "flatten" and not self.backbone.use_llm
        mean, std = self.revin.stats(x_raw)
        x_raw = self.revin.norm(x_raw, mean, std)
        x_patched = self.revin.norm(x_patched, mean, std)
        z_T, td = self.td_encoder(x_patched)                     # td: (B, P, V, d)
        z_F, fe = self.fd_encoder(x_raw)                         # fe: (B, bins, V, d)
        fd_var = fe.mean(dim=1)                                  # (B, V, d)
        seq = self.backbone.build_sequence_tokens(td, fd_var)    # (B, P*V, H)
        Hh, P = self.backbone.hidden_size, self.num_patches
        hidden = seq.reshape(B, P, V, Hh).permute(0, 2, 1, 3).reshape(B, V, P * Hh)
        out = self.flat_head(self.head_drop(hidden))             # (B, V, pred_len)
        forecast = self.revin.denorm(out.permute(0, 2, 1), mean, std)
        h_td = td.permute(0, 2, 1, 3).reshape(B * V, -1)        # (B*V, P*d)
        h_fd = fd_var.reshape(B * V, -1)                          # (B*V, d)
        return forecast, {"z_T": z_T, "z_F": z_F, "h_td": h_td, "h_fd": h_fd, "mean": mean, "std": std,
                          "td_patch_embeddings": td, "x_patched_in": x_patched, "x_raw_in": x_raw}


# ---------------------------------------------------------------------------------------------- losses
def _offdiag_logsoftmax(logits):
    n = logits.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=logits.device)
    return torch.log_softmax(logits.masked_fill(eye, -1e9), dim=1), eye


def target_similarity(y_rel):
    """
    Row-stochastic target similarity P[i, j] proportional to exp(-d_ij / t), d_ij = mean squared distance
    between the (normalised) futures of instances i and j; t = median off-diagonal distance (scale-free).
    Diagonal excluded. No gradient (targets are data).
    """
    with torch.no_grad():
        d = torch.cdist(y_rel, y_rel).pow(2) / y_rel.shape[1]
        n = d.shape[0]
        off = d[~torch.eye(n, dtype=torch.bool, device=d.device)]
        t = off.median().clamp_min(1e-8)
        logp, _ = _offdiag_logsoftmax(-d / t)
        return logp.exp()


def relational_kl(P, h, tau=0.1):
    """KL(P || Q_h), Q_h = row softmax of cosine similarities of h / tau (diagonal excluded); mean over rows."""
    hn = Fn.normalize(h, dim=1)
    logq, eye = _offdiag_logsoftmax(hn @ hn.t() / tau)
    P = P.masked_fill(eye, 0.0)
    return (P * (torch.log(P.clamp_min(1e-12)) - logq)).sum(1).mean()


def relational_target_loss(h_td, y, mean, std, tau=0.1, max_n=512):
    """L_RT. y: (B, H, V) targets in data space; mean/std: (B, 1, V) RevIN stats of the input windows."""
    y_rel = ((y - mean) / std).permute(0, 2, 1).reshape(-1, y.shape[1])     # (B*V, H), model's space
    idx = _subsample(h_td.shape[0], max_n, h_td.device)
    return relational_kl(target_similarity(y_rel[idx]), h_td[idx], tau)


def relational_consistency_loss(h_td, h_fd, tau=0.1, max_n=512):
    """L_TF: symmetric KL between the within-batch similarity distributions of the TD and FD views."""
    idx = _subsample(h_td.shape[0], max_n, h_td.device)
    a, b = Fn.normalize(h_td[idx], dim=1), Fn.normalize(h_fd[idx], dim=1)
    la, _ = _offdiag_logsoftmax(a @ a.t() / tau)
    lb, eye = _offdiag_logsoftmax(b @ b.t() / tau)
    pa, pb = la.exp().masked_fill(eye, 0), lb.exp().masked_fill(eye, 0)
    kl_ab = (pa * (la.masked_fill(eye, 0) - lb.masked_fill(eye, 0))).sum(1).mean()
    kl_ba = (pb * (lb.masked_fill(eye, 0) - la.masked_fill(eye, 0))).sum(1).mean()
    return 0.5 * (kl_ab + kl_ba)


def _subsample(n, max_n, device):
    if n <= max_n:
        return torch.arange(n, device=device)
    return torch.randperm(n, device=device)[:max_n]


# ---------------------------------------------------------------------------------------------- gradients
def conflict_free_combine(g_task, g_aux, alpha):
    """
    g = g_task + alpha * P(g_aux), P removes the component of g_aux that opposes g_task.
    g_task, g_aux: lists of tensors (None allowed in g_aux). Returns (list g, info dict).
    """
    ga = [torch.zeros_like(t) if a is None else a for t, a in zip(g_task, g_aux)]
    dot = sum((t * a).sum() for t, a in zip(g_task, ga))
    nt2 = sum((t * t).sum() for t in g_task)
    conflict = bool(dot < 0)
    coef = (dot / nt2.clamp_min(1e-20)) if conflict else torch.zeros((), device=dot.device)
    g = [t + alpha * (a - coef * t) for t, a in zip(g_task, ga)]
    na2 = sum((a * a).sum() for a in ga)
    cos = float(dot / (nt2.sqrt() * na2.sqrt()).clamp_min(1e-20))
    return g, {"conflict": conflict, "cos": cos}


# ---------------------------------------------------------------------------------------------- step
RAFT_VARIANTS = {
    # name: (use L_RT, use L_TF, project)
    "raft": (True, True, True),
    "raft_rt": (True, False, True),       # target-relational only (+ projection)
    "raft_tf": (False, True, True),       # TD-FD relational only (+ projection)
    "raft_noproj": (True, True, False),   # both losses, plain sum (no projection)
    "fdfix": (False, False, False),       # RAFT architecture (FD fix) with no auxiliary loss
}


def make_compute_losses(variant, alpha, stats):
    """
    Returns a drop-in replacement for train.compute_losses. train() calls it, then
    optimizer.zero_grad(); total_loss.backward(); optimizer.step().
    For projected variants, total_loss is a surrogate whose gradient equals the conflict-free direction g
    exactly:  surrogate = sum_p <g_p, theta_p>  (value shifted to equal L_task + alpha * L_aux).
    `stats` collects per-step diagnostics (conflict rate, cosine).
    """
    use_rt, use_tf, project = RAFT_VARIANTS[variant]

    def compute_losses(forecaster, tfb, x_patched, x_raw, y, lam=0.5, loss_mode=None, alpha_unused=None,
                       loss_combiner=None, epoch=None, total_epochs=None, warmup_epochs=None, **kw):
        forecast, aux = forecaster(x_patched, x_raw)
        task = Fn.mse_loss(forecast, y)
        zero = task.new_zeros(())
        l_rt = relational_target_loss(aux["h_td"], y, aux["mean"], aux["std"]) if use_rt else zero
        l_tf = relational_consistency_loss(aux["h_td"], aux["h_fd"]) if use_tf else zero
        l_aux = l_rt + l_tf
        value = task + alpha * l_aux
        if (use_rt or use_tf) and alpha > 0 and project:
            params = [p for p in forecaster.parameters() if p.requires_grad]
            g_t = torch.autograd.grad(task, params, retain_graph=True, allow_unused=True)
            g_a = torch.autograd.grad(l_aux, params, allow_unused=True)
            g_t = [torch.zeros_like(p) if g is None else g for g, p in zip(g_t, params)]
            g, info = conflict_free_combine(list(g_t), list(g_a), alpha)
            stats["steps"] = stats.get("steps", 0) + 1
            stats["conflicts"] = stats.get("conflicts", 0) + int(info["conflict"])
            stats["cos_sum"] = stats.get("cos_sum", 0.0) + info["cos"]
            surrogate = sum((gi.detach() * p).sum() for gi, p in zip(g, params))
            total = value.detach() + surrogate - surrogate.detach()
        else:
            total = value
        return {"task_loss": task, "l_t": l_rt.detach(), "l_f": l_tf.detach(), "l_b": zero,
                "l_tfb": l_aux.detach(), "total_loss": total, "effective_alpha": alpha, "forecast": forecast}

    return compute_losses


# ---------------------------------------------------------------------------------------------- builder
def build_raft_model(d_emb=64, patch_len=16, num_features=7, pred_len=96, num_prompt_tokens=10, use_revin=True,
                     seq_len=512, *, head_type, use_llm, llm_policy):
    """Same signature and components as train.build_model, with RAFTFDEncoder and RAFTForecaster."""
    from td_encoder import TDEncoder
    from tf_llm import TFLLMBackbone
    from tfb_module import TFBModule
    assert head_type == "flatten" and not use_llm and use_revin, "RAFT is defined on the frozen forecaster F"
    td = TDEncoder(patch_len=patch_len, num_features=num_features, d_emb=d_emb)
    fd = RAFTFDEncoder(seq_len=seq_len, num_features=num_features, d_emb=d_emb)
    backbone = TFLLMBackbone(d_emb=d_emb, num_prompt_tokens=num_prompt_tokens, use_llm=use_llm, llm_policy=llm_policy)
    tfb = TFBModule(d_emb=d_emb, proj_dim=128)          # unused by RAFT; kept so train() runs unchanged
    f = RAFTForecaster(td_encoder=td, fd_encoder=fd, backbone=backbone, pred_len=pred_len, patch_len=patch_len,
                       num_features=num_features, use_revin=use_revin, head_type=head_type, seq_len=seq_len)
    return f, tfb
