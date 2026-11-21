"""
Stage J (journal version): additional objectives, gradient combiners and the GPT-2 variant of RAFT.

Nothing in src/ is modified: the forecaster F, the TFB baseline and RAFT (src/raft.py) are imported unchanged, so
every Stage X result stays reproducible from tag x-declared. This module only ADDS:

  Auxiliary objectives on the time-domain representation h (all with RAFT's informative FD branch):
    "raft"   : L_RT + L_TF                     (RAFT; same functions as src/raft.py, temperature tau)
    "rnc"    : L_RnC                           Rank-N-Contrast (Zha et al., NeurIPS 2023) on h, ranked by the
                                               distance between the windows' normalised futures (replaces L_RT)
    "rnctf"  : L_RnC + L_TF                    RnC in place of L_RT, with RAFT's consistency loss

  Gradient combiners for g_task and g_aux (two tasks):
    "cf"     : conflict-free projection of RAFT  g = g_t + a * P(g_a)  (src/raft.conflict_free_combine)
    "pc"     : PCGrad (Yu et al., NeurIPS 2020), symmetric projection of the two gradients
    "gs"     : gradient-similarity gating (Du et al., 2018): g = g_t + a * g_a if cos(g_t, g_a) > 0 else g_t
    "sum"    : plain weighted sum g = g_t + a * g_a

  RAFTForecasterLLM: RAFT on the GPT-2 forecaster E (flatten head, LLM policy norms_pos). The representations
  used by the losses are the same as for F: h = time-domain encoder output, u = pooled FD output.
"""
import torch
import torch.nn.functional as Fn

import raft as R
from forecast_head import TFLLMForecaster


# ---------------------------------------------------------------------------------------------- RnC
def rnc_loss(h, y_rel, tau=0.1):
    """
    Rank-N-Contrast. For anchor i and each j != i:
        l_ij = -log( exp(s_ij / tau) / sum_{k != i, d_ik >= d_ij} exp(s_ik / tau) ),
    s = cosine similarity of h (same similarity as RAFT's Q), d = mean squared distance between the futures.
    Returns the mean of l_ij over all ordered pairs. Ties in d: the denominator includes every k with d_ik >= d_ij.
    """
    n = h.shape[0]
    hn = Fn.normalize(h, dim=1)
    logits = hn @ hn.t() / tau                                            # (n, n)
    with torch.no_grad():
        d = torch.cdist(y_rel, y_rel).pow(2) / y_rel.shape[1]
    eye = torch.eye(n, dtype=torch.bool, device=h.device)
    d = d.masked_fill(eye, -1.0)                                          # the anchor itself is never in a denominator
    # sort each row by distance, descending; the denominator for j is a log-cum-sum-exp up to j's last tie
    order = torch.argsort(d, dim=1, descending=True)
    ds = torch.gather(d, 1, order)
    ls = torch.gather(logits.masked_fill(eye, -1e9), 1, order)
    lcse = torch.logcumsumexp(ls, dim=1)                                  # (n, n): over k with rank <= position
    # include all ties: for descending ds, the last position with ds >= ds[p] is (count of k with ds[k] >= ds[p]) - 1
    neg = (-ds).contiguous()                                              # ascending per row
    last = torch.searchsorted(neg, neg, right=True) - 1
    denom = torch.gather(lcse, 1, last)
    valid = ds >= 0                                                       # exclude the anchor (d = -1)
    return (-(ls - denom))[valid].mean()


def rnc_target_loss(h_td, y, mean, std, tau=0.1, max_n=512):
    y_rel = ((y - mean) / std).permute(0, 2, 1).reshape(-1, y.shape[1])
    idx = R._subsample(h_td.shape[0], max_n, h_td.device)
    return rnc_loss(h_td[idx], y_rel[idx], tau)


# ---------------------------------------------------------------------------------------------- combiners
def _dot(a, b):
    return sum((x * y).sum() for x, y in zip(a, b))


def pcgrad_combine(g_task, g_aux, alpha):
    """PCGrad for two tasks: each gradient is projected onto the normal plane of the other if they conflict."""
    ga = [torch.zeros_like(t) if a is None else a for t, a in zip(g_task, g_aux)]
    dot, nt2, na2 = _dot(g_task, ga), _dot(g_task, g_task), _dot(ga, ga)
    conflict = bool(dot < 0)
    if conflict:
        gt_p = [t - (dot / na2.clamp_min(1e-20)) * a for t, a in zip(g_task, ga)]
        ga_p = [a - (dot / nt2.clamp_min(1e-20)) * t for t, a in zip(g_task, ga)]
    else:
        gt_p, ga_p = list(g_task), ga
    g = [t + alpha * a for t, a in zip(gt_p, ga_p)]
    return g, {"conflict": conflict, "cos": float(dot / (nt2.sqrt() * na2.sqrt()).clamp_min(1e-20))}


def gsgate_combine(g_task, g_aux, alpha):
    """Gradient-similarity gating: use the auxiliary gradient only when its cosine with g_task is positive."""
    ga = [torch.zeros_like(t) if a is None else a for t, a in zip(g_task, g_aux)]
    dot, nt2, na2 = _dot(g_task, ga), _dot(g_task, g_task), _dot(ga, ga)
    conflict = bool(dot <= 0)
    g = list(g_task) if conflict else [t + alpha * a for t, a in zip(g_task, ga)]
    return g, {"conflict": conflict, "cos": float(dot / (nt2.sqrt() * na2.sqrt()).clamp_min(1e-20))}


def sum_combine(g_task, g_aux, alpha):
    ga = [torch.zeros_like(t) if a is None else a for t, a in zip(g_task, g_aux)]
    dot, nt2, na2 = _dot(g_task, ga), _dot(g_task, g_task), _dot(ga, ga)
    g = [t + alpha * a for t, a in zip(g_task, ga)]
    return g, {"conflict": bool(dot < 0), "cos": float(dot / (nt2.sqrt() * na2.sqrt()).clamp_min(1e-20))}


COMBINERS = {"cf": R.conflict_free_combine, "pc": pcgrad_combine, "gs": gsgate_combine, "sum": sum_combine}
AUX = {"raft": (True, True), "rnc": (True, False), "rnctf": (True, True)}     # (target term, L_TF)


def make_compute_losses_j(aux, combiner, alpha, stats, tau=0.1):
    """
    Drop-in replacement for train.compute_losses (same contract as src/raft.make_compute_losses).
    aux in AUX, combiner in COMBINERS. The target term is L_RT for aux="raft", L_RnC for "rnc"/"rnctf".
    The returned total_loss is a surrogate whose gradient equals the combined direction g exactly.
    """
    assert aux in AUX and combiner in COMBINERS
    use_target, use_tf = AUX[aux]
    comb = COMBINERS[combiner]

    def compute_losses(forecaster, tfb, x_patched, x_raw, y, **kw):
        forecast, a = forecaster(x_patched, x_raw)
        task = Fn.mse_loss(forecast, y)
        zero = task.new_zeros(())
        if aux == "raft":
            l_t = R.relational_target_loss(a["h_td"], y, a["mean"], a["std"], tau=tau)
        else:
            l_t = rnc_target_loss(a["h_td"], y, a["mean"], a["std"], tau=tau)
        l_tf = R.relational_consistency_loss(a["h_td"], a["h_fd"], tau=tau) if use_tf else zero
        l_aux = l_t + l_tf
        value = task + alpha * l_aux
        params = [p for p in forecaster.parameters() if p.requires_grad]
        g_t = torch.autograd.grad(task, params, retain_graph=True, allow_unused=True)
        g_a = torch.autograd.grad(l_aux, params, allow_unused=True)
        g_t = [torch.zeros_like(p) if g is None else g for g, p in zip(g_t, params)]
        g, info = comb(list(g_t), list(g_a), alpha)
        stats["steps"] = stats.get("steps", 0) + 1
        stats["conflicts"] = stats.get("conflicts", 0) + int(info["conflict"])
        stats["cos_sum"] = stats.get("cos_sum", 0.0) + info["cos"]
        surrogate = sum((gi.detach() * p).sum() for gi, p in zip(g, params))
        total = value.detach() + surrogate - surrogate.detach()
        return {"task_loss": task, "l_t": l_t.detach(), "l_f": l_tf.detach(), "l_b": zero,
                "l_tfb": l_aux.detach(), "total_loss": total, "effective_alpha": alpha, "forecast": forecast}

    return compute_losses


# ---------------------------------------------------------------------------------------------- GPT-2 variant
class RAFTForecasterLLM(TFLLMForecaster):
    """RAFTForecaster with the GPT-2 backbone (flatten head). Same aux outputs as src/raft.RAFTForecaster."""

    def forward(self, x_patched, x_raw):
        B, V = x_patched.shape[0], x_patched.shape[-1]
        assert self.use_revin and self.head_type == "flatten" and self.backbone.use_llm
        mean, std = self.revin.stats(x_raw)
        x_raw = self.revin.norm(x_raw, mean, std)
        x_patched = self.revin.norm(x_patched, mean, std)
        z_T, td = self.td_encoder(x_patched)                     # (B, P, V, d)
        z_F, fe = self.fd_encoder(x_raw)                         # (B, bins, V, d)
        fd_var = fe.mean(dim=1)                                  # (B, V, d)
        seq = self.backbone.build_sequence_tokens(td, fd_var)    # (B, P*V, H)
        prompt = self.backbone.get_prompt_tokens(B)
        n_prompt = prompt.shape[1]
        hidden = self.backbone.forward_backbone(torch.cat([prompt, seq], dim=1))[:, n_prompt:, :]
        Hh, P = self.backbone.hidden_size, self.num_patches
        hidden = hidden.reshape(B, P, V, Hh).permute(0, 2, 1, 3).reshape(B, V, P * Hh)
        out = self.flat_head(self.head_drop(hidden))
        forecast = self.revin.denorm(out.permute(0, 2, 1), mean, std)
        return forecast, {"z_T": z_T, "z_F": z_F, "h_td": td.permute(0, 2, 1, 3).reshape(B * V, -1),
                          "h_fd": fd_var.reshape(B * V, -1), "mean": mean, "std": std,
                          "td_patch_embeddings": td, "x_patched_in": x_patched, "x_raw_in": x_raw}


def build_raft_model_any(d_emb=64, patch_len=16, num_features=7, pred_len=96, num_prompt_tokens=10, use_revin=True,
                         seq_len=512, *, head_type, use_llm, llm_policy):
    """src/raft.build_raft_model for F; the same components with the GPT-2 backbone for E."""
    if not use_llm:
        return R.build_raft_model(d_emb=d_emb, patch_len=patch_len, num_features=num_features, pred_len=pred_len,
                                  num_prompt_tokens=num_prompt_tokens, use_revin=use_revin, seq_len=seq_len,
                                  head_type=head_type, use_llm=use_llm, llm_policy=llm_policy)
    from td_encoder import TDEncoder
    from tf_llm import TFLLMBackbone
    from tfb_module import TFBModule
    assert head_type == "flatten" and use_revin
    td = TDEncoder(patch_len=patch_len, num_features=num_features, d_emb=d_emb)
    fd = R.RAFTFDEncoder(seq_len=seq_len, num_features=num_features, d_emb=d_emb)
    backbone = TFLLMBackbone(d_emb=d_emb, num_prompt_tokens=num_prompt_tokens, use_llm=True, llm_policy=llm_policy)
    tfb = TFBModule(d_emb=d_emb, proj_dim=128)
    f = RAFTForecasterLLM(td_encoder=td, fd_encoder=fd, backbone=backbone, pred_len=pred_len, patch_len=patch_len,
                          num_features=num_features, use_revin=use_revin, head_type=head_type, seq_len=seq_len)
    return f, tfb
