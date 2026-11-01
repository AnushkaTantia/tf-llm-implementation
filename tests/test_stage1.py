"""
Stage 1 verification of the corrected forecaster (checklist S1).

    python tests/test_stage1.py <path-to-src> [--gpt2 real|random] [--device cpu|cuda] [--csv ETTh1.csv]

--gpt2 real   : pretrained GPT-2 from HuggingFace (use on Colab/Kaggle).
--gpt2 random : GPT-2 with the identical architecture and parameter names (12 layers, 768 hidden,
                124M parameters) but random weights, for machines without internet access.
Tests that only need the module structure run on the full-size GPT-2; the end-to-end training tests
use 16-window subsets of each split so they finish in minutes on a CPU.

Groups
  P   positional embeddings (TD and FD): shape, learnability, sharing across variables, applied once,
      before self-attention, order sensitivity
  H   flatten head: the exact tensor shapes, sequence-token order, head input = sequence tokens only
  R   RevIN and variable ordering: round trip, per-variable shift/scale equivariance of the forecast
  G   GPT-2 policy and trainable parameters (E, F, original B), optimizer = exactly the trainable set
  B   forward/backward + optimizer step for E and F: every trainable tensor gets a gradient,
      frozen tensors get none and do not change
  T   TFB paths: alpha = 0 is exactly task-only; alpha > 0 adds alpha * L_TFB and reaches the encoders
  S   schedulers: NaN / +inf / -inf rejected for HMM, Kalman, OA-HMM; state, histories and alpha
      unchanged; counter; normal operation resumes (equivalent to never having seen them)
  C   configuration: required options, invalid combinations rejected
  I   integration: train() end to end for E and F; run_config.json, epoch_log.json, checkpoint config;
      checkpoint reloads strictly and reproduces the stored validation MSE
  D   scheduler dispatch in train(): alpha used in each epoch = the scheduler's alpha decided after the
      previous epoch; OA-HMM receives (val_mse, epoch-mean train task loss); warmup and fixed alpha
"""
import argparse, copy, glob, json, math, os, random, sys, tempfile, time, traceback

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("--gpt2", default="random", choices=["real", "random"])
ap.add_argument("--device", default="cpu")
ap.add_argument("--csv", default=None)
ap.add_argument("--skip_integration", action="store_true")
ap.add_argument("--only", default=None, help="comma-separated check ids, e.g. I1,D1")
args = ap.parse_args()
sys.path.insert(0, args.src)

import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model

if args.gpt2 == "random":
    _FULL = GPT2Config()   # = gpt2 (124M): n_layer 12, n_embd 768, n_head 12, n_positions 1024
    GPT2Model.from_pretrained = classmethod(lambda cls, name, **k: GPT2Model(_FULL))

import train as T
import hmm_scheduler as HS
import kalman_scheduler as KS
import oahmm_scheduler as OS

DEV = torch.device(args.device)
B, L, V, P, PL, HID, HOR = 2, 512, 7, 32, 16, 768, 96
RESULTS = []
LOG = []


def log(msg=""):
    print(msg, flush=True); LOG.append(msg)


def check(name, fn):
    if args.only and name.split()[0] not in args.only.split(","):
        return
    t0 = time.time()
    try:
        detail = fn()
        RESULTS.append((name, True, detail)); log(f"[PASS] {name}" + (f" -- {detail}" if detail else ""))
    except Exception as e:
        RESULTS.append((name, False, repr(e)))
        log(f"[FAIL] {name} -- {e!r}"); log(traceback.format_exc())
    finally:
        LOG.append(f"       ({time.time() - t0:.1f}s)")


def batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, L, V, generator=g) * 2 + 3
    return x.reshape(B, P, PL, V).to(DEV), x.to(DEV)


CONFIGS = {   # name -> build_model options
    "E": dict(head_type="flatten", use_llm=True, llm_policy="norms_pos"),
    "F": dict(head_type="flatten", use_llm=False, llm_policy="none"),
    "B_original": dict(head_type="mask", use_llm=True, llm_policy="frozen"),
}
_cache = {}


def model(name):
    if name not in _cache:
        torch.manual_seed(0)
        f, tfb = T.build_model(pred_len=HOR, **CONFIGS[name])
        _cache[name] = (f.to(DEV), tfb.to(DEV))
    return _cache[name]


# ----------------------------------------------------------------------------- P
def p1():
    f, _ = model("F")
    out = []
    for enc, n_pos, nm in [(f.td_encoder, 256, "TD"), (f.fd_encoder, 257, "FD")]:
        pe = enc.pos_embedding
        assert tuple(pe.shape) == (1, n_pos, 1, 64), (nm, pe.shape)
        assert pe.requires_grad, f"{nm} positional embedding not learnable"
        out.append(f"{nm} pos_embedding {tuple(pe.shape)} learnable")
    return "; ".join(out)


def p2():
    """With the transformer replaced by identity and dropout off, the encoder output must be exactly
    patch_proj(x) + pos[p] for every variable: added once, shared across variables."""
    f, _ = model("F")
    td = copy.deepcopy(f.td_encoder).eval(); td.transformer = nn.Identity()
    xp, _ = batch(1)
    _, emb = td(xp)                                    # (B, P, V, d)
    exp = td.patch_proj(xp.permute(0, 1, 3, 2)) + td.pos_embedding[0, :P, 0, :][None, :, None, :]
    err = (emb - exp).abs().max().item()
    assert err < 1e-5, err
    fd = copy.deepcopy(f.fd_encoder).eval(); fd.transformer = nn.Identity()
    _, xr = batch(1)
    _, femb = fd(xr)                                   # (B, F, V, d)
    mag = fd.revin(torch.fft.rfft(xr, dim=1).abs())
    fexp = fd.freq_proj(mag.unsqueeze(-1)) + fd.pos_embedding[0, :, 0, :][None, :, None, :]
    ferr = (femb - fexp).abs().max().item()
    assert ferr < 1e-4, ferr
    return f"TD max error {err:.1e}, FD max error {ferr:.1e} (added once, same vector for every variable)"


def p3():
    """Positional embedding sits before self-attention: permuting the patch order changes the
    (order-invariant) sum of the encoder outputs. Without a positional input it would not."""
    f, _ = model("F")
    td = copy.deepcopy(f.td_encoder).eval()
    xp, _ = batch(2)
    perm = torch.randperm(P, generator=torch.Generator().manual_seed(0))
    s1 = td(xp)[1].sum(1); s2 = td(xp[:, perm])[1].sum(1)
    d = (s1 - s2).abs().max().item()
    assert d > 1e-3, d
    with torch.no_grad():
        td.pos_embedding.zero_()
    s1 = td(xp)[1].sum(1); s2 = td(xp[:, perm])[1].sum(1)
    d0 = (s1 - s2).abs().max().item()
    assert d0 < 1e-4, d0
    return f"order-sensitive (difference {d:.3f}); with positions zeroed the encoder is order-blind ({d0:.1e})"


# ----------------------------------------------------------------------------- H
def h1():
    for nm in ("E", "F"):
        f, _ = model(nm); f.eval()
        xp, xr = batch(3)
        with torch.no_grad():
            y, _ = f(xp, xr)
        got = f.shape_log
        exp = {"before_flatten": (B, P, V, HID), "per_variable": (B, V, P, HID),
               "flattened": (B, V, P * HID), "linear_output": (B, V, HOR), "final_forecast": (B, HOR, V)}
        assert {k: got[k] for k in exp} == exp, (nm, got)
        assert f.flat_head.in_features == P * HID == 24576 and f.flat_head.out_features == HOR
        assert tuple(y.shape) == (B, HOR, V)
    return ("E and F: (2,32,7,768) -> (2,7,32,768) -> (2,7,24576) -> (2,7,96) -> (2,96,7); "
            "flat_head = Linear(24576, 96)")


def h2():
    """Sequence-token order is patch-major: token p*V + v is the projection of (patch p, variable v)."""
    f, _ = model("F")
    bb = copy.deepcopy(f.backbone).eval()
    g = torch.Generator().manual_seed(4)
    td = torch.randn(B, P, V, 64, generator=g).to(DEV); fd = torch.randn(B, V, 64, generator=g).to(DEV)
    seq = bb.build_sequence_tokens(td, fd)
    a = torch.sigmoid(bb.fusion_alpha)
    worst = 0.0
    for p_, v_ in [(0, 0), (0, 6), (5, 3), (31, 6)]:
        one = bb.input_proj(bb.fusion_norm(a * td[:, p_, v_] + (1 - a) * fd[:, v_]))
        worst = max(worst, (seq[:, p_ * V + v_] - one).abs().max().item())
    assert worst < 1e-5, worst
    assert seq.shape == (B, P * V, HID)
    return f"token p*7+v = (patch p, variable v); {P * V} tokens, no extra tokens; max error {worst:.1e}"


def h3():
    """E: the flatten head's input is exactly the GPT-2 output at the sequence positions (prompt
    positions dropped), regrouped per variable."""
    f, _ = model("E"); f.eval()
    xp, xr = batch(5)
    cap = {}
    h_in = f.flat_head.register_forward_hook(lambda m, i, o: cap.__setitem__("head_in", i[0]))
    h_bb = f.backbone.gpt2.register_forward_hook(lambda m, i, o: cap.__setitem__("gpt2_out", o.last_hidden_state))
    with torch.no_grad():
        f(xp, xr)
    h_in.remove(); h_bb.remove()
    n_prompt = f.backbone.num_prompt_tokens
    g = cap["gpt2_out"]
    assert g.shape == (B, n_prompt + P * V, HID), g.shape
    exp = g[:, n_prompt:].reshape(B, P, V, HID).permute(0, 2, 1, 3).reshape(B, V, P * HID)
    err = (cap["head_in"] - exp).abs().max().item()
    assert err == 0.0, err
    return f"GPT-2 sees {n_prompt} prompt + {P * V} sequence tokens; head input = sequence positions only (exact)"


# ----------------------------------------------------------------------------- R
def r1():
    f, _ = model("F")
    _, xr = batch(6)
    m, s = f.revin.stats(xr)
    back = f.revin.denorm(f.revin.norm(xr, m, s), m, s)
    err = (back - xr).abs().max().item()
    assert err < 1e-4, err
    assert m.shape == (B, 1, V) and s.shape == (B, 1, V)
    return f"stats per (window, variable) over time; norm->denorm round trip error {err:.1e}"


def r2():
    """Adding b to variable 3 (or scaling it by a) must shift (scale) ONLY forecast column 3, by
    exactly b (a). Proves input normalization, output de-normalization and the variable order."""
    out = []
    for nm in ("E", "F", "B_original"):
        f, _ = model(nm); f.eval()
        xp, xr = batch(7)
        with torch.no_grad():
            y0, _ = f(xp, xr)
            xr2 = xr.clone(); xr2[:, :, 3] += 5.0
            y1, _ = f(xr2.reshape(B, P, PL, V), xr2)
            xr3 = xr.clone(); xr3[:, :, 3] *= 3.0
            y2, _ = f(xr3.reshape(B, P, PL, V), xr3)
        others = [i for i in range(V) if i != 3]
        e_shift = (y1[..., 3] - (y0[..., 3] + 5)).abs().max().item()
        e_other = max((y1[..., others] - y0[..., others]).abs().max().item(),
                      (y2[..., others] - y0[..., others]).abs().max().item())
        y0n = y0[..., 3] - xr[:, :, 3].mean(1, keepdim=True)
        y2n = y2[..., 3] - xr3[:, :, 3].mean(1, keepdim=True)
        e_scale = (y2n - 3 * y0n).abs().max().item() / (1 + y0n.abs().max().item())
        assert e_shift < 1e-3 and e_other < 1e-4 and e_scale < 1e-3, (nm, e_shift, e_other, e_scale)
        out.append(f"{nm}: shift err {e_shift:.1e}, scale err {e_scale:.1e}, other columns {e_other:.1e}")
    return "; ".join(out)


# ----------------------------------------------------------------------------- G
EXPECTED_GPT2_TRAINABLE = ({f"h.{i}.ln_{j}.{w}" for i in range(12) for j in (1, 2) for w in ("weight", "bias")}
                           | {"ln_f.weight", "ln_f.bias", "wpe.weight"})


def g1():
    f, tfb = model("E")
    names = {n for n, p in f.backbone.gpt2.named_parameters() if p.requires_grad}
    assert names == EXPECTED_GPT2_TRAINABLE, (names ^ EXPECTED_GPT2_TRAINABLE)
    n_tr = sum(p.numel() for n, p in f.backbone.gpt2.named_parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in f.backbone.gpt2.parameters())
    frozen_kinds = {n.split(".")[2] if n.startswith("h.") else n.split(".")[0]
                    for n, p in f.backbone.gpt2.named_parameters() if not p.requires_grad}
    assert f.backbone.prompt_tokens.requires_grad
    assert not f.mask_token.requires_grad and not any(p.requires_grad for p in f.mask_block.parameters())
    return (f"E: GPT-2 trainable = exactly {len(names)} tensors (ln_1, ln_2 of 12 blocks, ln_f, wpe) = {n_tr:,} of "
            f"{n_all:,}; frozen GPT-2 groups {sorted(frozen_kinds)}; prompt tokens trainable; mask head frozen")


def g2():
    f, _ = model("F")
    assert f.backbone.gpt2 is None
    assert not any("gpt2" in n for n, _ in f.named_parameters())
    assert not f.backbone.prompt_tokens.requires_grad
    assert not f.mask_token.requires_grad and not any(p.requires_grad for p in f.mask_block.parameters())
    fb, _ = model("B_original")
    assert not any(p.requires_grad for p in fb.backbone.gpt2.parameters())
    assert fb.mask_token.requires_grad and all(p.requires_grad for p in fb.mask_block.parameters())
    assert not hasattr(fb, "flat_head")
    return "F: no GPT-2 parameters exist, prompt tokens and mask head frozen; original B: GPT-2 fully frozen, mask head trainable"


def g3():
    out = []
    for nm in ("E", "F", "B_original"):
        f, tfb = model(nm)
        params = T.get_trainable_parameters(f, tfb)
        ids = {id(p) for p in params}
        req = {id(p) for p in list(f.parameters()) + list(tfb.parameters()) if p.requires_grad}
        assert ids == req and len(ids) == len(params), nm
        opt = torch.optim.Adam(params, lr=1e-4)
        opt_ids = {id(p) for gr in opt.param_groups for p in gr["params"]}
        assert opt_ids == req
        rep = T.trainable_parameter_report(f, tfb)
        log(f"       {nm}: trainable tensors ({len(rep['trainable'])}), {rep['n_trainable']:,} parameters:")
        for n, k in rep["trainable"]:
            log(f"           {n:66s} {k:>10,}")
        out.append(f"{nm}: optimizer = {len(params)} tensors / {rep['n_trainable']:,} params, frozen {rep['n_frozen']:,}")
    return "; ".join(out)


# ----------------------------------------------------------------------------- B
def backward_step(nm):
    f, tfb = model(nm)
    f = copy.deepcopy(f).train(); tfb = copy.deepcopy(tfb).train()
    xp, xr = batch(8)
    y = torch.randn(B, HOR, V, generator=torch.Generator().manual_seed(9)).to(DEV)
    params = T.get_trainable_parameters(f, tfb)
    opt = torch.optim.Adam(params, lr=1e-3)
    before = {n: p.detach().clone() for n, p in list(f.named_parameters()) + [("tfb." + n, p) for n, p in tfb.named_parameters()]}
    random.seed(0); torch.manual_seed(0)
    losses = T.compute_losses(f, tfb, xp, xr, y, loss_mode="fixed_weighted", alpha=0.3)
    opt.zero_grad(); losses["total_loss"].backward()
    named = list(f.named_parameters()) + [("tfb." + n, p) for n, p in tfb.named_parameters()]
    no_grad = [n for n, p in named if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
    frozen_grad = [n for n, p in named if not p.requires_grad and p.grad is not None]
    assert not no_grad, f"trainable but no gradient: {no_grad}"
    assert not frozen_grad, f"frozen but has gradient: {frozen_grad}"
    opt.step()
    changed_frozen = [n for n, p in named if not p.requires_grad and not torch.equal(p.detach(), before[n])]
    unchanged_tr = [n for n, p in named if p.requires_grad and torch.equal(p.detach(), before[n])]
    assert not changed_frozen and not unchanged_tr, (changed_frozen, unchanged_tr)
    assert torch.isfinite(losses["total_loss"])
    n_tr = sum(1 for _, p in named if p.requires_grad)
    return f"{nm}: loss {losses['total_loss'].item():.3f}; all {n_tr} trainable tensors have gradients and moved; frozen tensors untouched"


# ----------------------------------------------------------------------------- T
def t1():
    f, tfb = model("F")
    f = copy.deepcopy(f).train(); tfb = copy.deepcopy(tfb).train()
    xp, xr = batch(10)
    y = torch.randn(B, HOR, V, generator=torch.Generator().manual_seed(11)).to(DEV)
    res = {}
    for a in (0.0, 0.3):
        for p in list(f.parameters()) + list(tfb.parameters()):
            p.grad = None
        random.seed(1); torch.manual_seed(1)
        l = T.compute_losses(f, tfb, xp, xr, y, loss_mode="fixed_weighted", alpha=a)
        l["total_loss"].backward()
        res[a] = dict(total=l["total_loss"].item(), task=l["task_loss"].item(), tfb=l["l_tfb"].item(),
                      tfb_grad=sum(float(p.grad.abs().sum()) for p in tfb.parameters() if p.grad is not None),
                      td_grad=[p.grad.clone() for p in f.td_encoder.parameters()])
    r0, r3 = res[0.0], res[0.3]
    assert r0["total"] == r0["task"], "alpha=0 must be exactly task-only"
    assert r0["tfb_grad"] == 0.0
    assert abs(r3["total"] - (r3["task"] + 0.3 * r3["tfb"])) < 1e-5
    assert r3["tfb_grad"] > 0
    diff = max((a - b).abs().max().item() for a, b in zip(r0["td_grad"], r3["td_grad"]))
    assert diff > 1e-6, "TFB gradient does not reach the TD encoder"
    return (f"alpha=0: total == task ({r0['total']:.4f}), TFB grad 0; alpha=0.3: total = task + 0.3*L_TFB, "
            f"TFB module grad {r3['tfb_grad']:.3f}, TD-encoder grad changes by up to {diff:.2e}")


# ----------------------------------------------------------------------------- S
def snap(s):
    d = {}
    for k, v in vars(s).items():
        if k == "skipped_observations":
            continue
        d[k] = copy.deepcopy(v.tolist() if isinstance(v, np.ndarray) else v)
    return d


def s1():
    out = []
    rng = np.random.default_rng(0)
    vals = list(1.0 + 0.05 * rng.standard_normal(9)); trn = list(0.6 - 0.02 * np.arange(9))
    mk = {"HMM": (HS.HMMScheduler, lambda s, v, t: s.observe(v)),
          "Kalman": (KS.KalmanScheduler, lambda s, v, t: s.observe(v)),
          "OA-HMM": (OS.OAHMMScheduler, lambda s, v, t: s.observe(v, t))}
    for nm, (cls, obs) in mk.items():
        s, ref = cls(), cls()
        assert s.skipped_observations == 0
        for v, t in zip(vals[:6], trn[:6]):
            obs(s, v, t); obs(ref, v, t)
        before, a_before = snap(s), s.get_alpha()
        st_before = s.get_state_name() if hasattr(s, "get_state_name") else None
        bad = [(float("nan"), 0.5), (float("inf"), 0.5), (float("-inf"), 0.5)]
        if nm == "OA-HMM":
            bad += [(1.0, float("nan")), (1.0, float("inf")), (1.0, float("-inf"))]
        for v, t in bad:
            obs(s, v, t)
            assert snap(s) == before, f"{nm}: state changed by {v},{t}"
            assert s.get_alpha() == a_before
            if st_before is not None:
                assert s.get_state_name() == st_before
        assert s.skipped_observations == len(bad)
        for v, t in zip(vals[6:], trn[6:]):
            obs(s, v, t); obs(ref, v, t)
        assert snap(s) == snap(ref) and s.get_alpha() == ref.get_alpha(), f"{nm}: did not resume normally"
        assert math.isfinite(s.get_alpha())
        out.append(f"{nm}: {len(bad)} rejected, unchanged, resumes = identical to a scheduler that never saw them")
    return "; ".join(out)


# ----------------------------------------------------------------------------- C
def c1():
    import inspect
    for fn in (T.build_model, T.train):
        sig = inspect.signature(fn)
        for k in ("head_type", "use_llm", "llm_policy"):
            assert sig.parameters[k].default is inspect.Parameter.empty, f"{fn.__name__}.{k} has a default"
    assert inspect.signature(T.train).parameters["seed"].default is inspect.Parameter.empty
    bad = [dict(head_type="mask", use_llm=False, llm_policy="none"),
           dict(head_type="flatten", use_llm=False, llm_policy="norms_pos"),
           dict(head_type="flatten", use_llm=True, llm_policy="all"),
           dict(head_type="linear", use_llm=True, llm_policy="frozen")]
    for b in bad:
        try:
            T.build_model(pred_len=HOR, **b)
        except AssertionError:
            continue
        raise AssertionError(f"accepted invalid configuration {b}")
    try:
        T.build_model(pred_len=HOR)
    except TypeError:
        pass
    else:
        raise AssertionError("build_model without architecture options was accepted")
    return "head_type, use_llm, llm_policy, seed have no defaults; 4 invalid combinations and a missing configuration are rejected"


# ----------------------------------------------------------------------------- I / D
def find_csv():
    if args.csv:
        return args.csv
    for pat in ["/kaggle/input/**/ETTh1.csv", os.path.join(args.src, "..", "**", "ETTh1.csv"), "**/ETTh1.csv"]:
        hits = glob.glob(pat, recursive=True)
        if hits:
            return hits[0]
    raise FileNotFoundError("ETTh1.csv not found; pass --csv")


N_SUB = 16
_orig_len = None


def small_data(on=True):
    global _orig_len
    from loaders import ETTh1Dataset
    if on and _orig_len is None:
        _orig_len = ETTh1Dataset.__len__
        ETTh1Dataset.__len__ = lambda self: min(_orig_len(self), N_SUB)
    elif not on and _orig_len is not None:
        ETTh1Dataset.__len__ = _orig_len; _orig_len = None


def run_train(tmp, nm, **kw):
    small_data(True)
    d = os.path.join(tmp, nm)
    base = dict(csv_path=find_csv(), checkpoint_dir=d, pred_len=HOR, batch_size=4, learning_rate=1e-3,
                epochs=3, seed=42, run_sanity_check=False, device=str(DEV), log_every=10 ** 9)
    base.update(kw)
    _, _, r = T.train(**base)
    return d, r


def i1(tmp):
    out = []
    for nm in ("E", "F"):
        d, r = run_train(tmp, nm, loss_mode="fixed_weighted", alpha=0.3, **CONFIGS[nm])
        cfg = json.load(open(os.path.join(d, "run_config.json")))
        for k in ("head_type", "use_llm", "llm_policy", "seed", "loss_mode", "alpha", "epochs", "patience",
                  "min_epochs", "pred_len", "seq_len", "batch_size", "learning_rate", "split", "checkpoint_rule"):
            assert k in cfg, k
        assert cfg["head_type"] == CONFIGS[nm]["head_type"] and cfg["use_llm"] == CONFIGS[nm]["use_llm"]
        hist = json.load(open(os.path.join(d, "epoch_log.json")))
        assert len(hist) == r["epochs_run"] == 3
        assert all(math.isfinite(h["val_mse"]) and math.isfinite(h["train_task"]) for h in hist)
        best = min(range(len(hist)), key=lambda i: hist[i]["val_mse"]) + 1
        assert r["best_epoch"] == best
        ck = torch.load(os.path.join(d, "best_model.pt"), map_location="cpu")
        assert ck["config"]["head_type"] == CONFIGS[nm]["head_type"] and ck["epoch"] == best
        import metrics
        f, _ = T.build_model(pred_len=HOR, **CONFIGS[nm]); f.load_state_dict(ck["forecaster"]); f.to(DEV)
        from loaders import ETTh1Dataset
        from torch.utils.data import DataLoader
        vds = ETTh1Dataset(cfg["csv_path"], flag="val", seq_len=512, pred_len=HOR, patch_len=16)
        v = metrics.evaluate_full(f, DataLoader(vds, batch_size=8), DEV)["mse"]
        assert abs(v - ck["val_mse"]) < 1e-4, (v, ck["val_mse"])
        m = metrics.eval_checkpoint(os.path.join(d, "best_model.pt"), cfg["csv_path"], device=str(DEV))
        assert abs(m["mse"] - r["test_mse"]) < 1e-4 and m["head_type"] == CONFIGS[nm]["head_type"]
        out.append(f"{nm}: 3 epochs, best epoch {best}, checkpoint reload reproduces val MSE "
                   f"{ck['val_mse']:.4f} (diff {abs(v - ck['val_mse']):.1e}) and test MSE; RMSE {m['rmse']:.3f}, SMAPE {m['smape']:.1f}%")
    return "; ".join(out)


def d1(tmp):
    calls = []
    orig = T.OAHMMScheduler.observe

    def spy(self, val_mse, train_task):
        calls.append((val_mse, train_task)); return orig(self, val_mse, train_task)
    T.OAHMMScheduler.observe = spy
    out = []
    try:
        for mode, kw in [("fixed_weighted", dict(alpha=0.0)), ("fixed_weighted", dict(alpha=0.3)),
                         ("time_varying", dict(alpha=0.3, warmup_epochs=2)), ("hmm_varying", {}),
                         ("kalman_varying", {}), ("oahmm_varying", dict(alpha_over=0.5)), ("variable_weighted", {})]:
            tag = f"{mode}_{kw.get('alpha', '')}"
            d, r = run_train(tmp, "D_" + tag, loss_mode=mode, epochs=4, **CONFIGS["F"], **kw)
            h = r["history"]
            a = [x["alpha_used"] for x in h]
            if mode == "fixed_weighted":
                assert all(x == kw["alpha"] for x in a), a
            elif mode == "time_varying":
                assert np.allclose(a, [0.3 * min(e / 2, 1) for e in range(1, 5)]), a
            elif mode == "variable_weighted":
                assert all(x is None for x in a)
            else:
                first = 0.30 if mode == "kalman_varying" else 0.05
                assert a[0] == first, (mode, a)
                for e in range(1, len(h)):
                    assert a[e] == h[e - 1]["alpha_next"], (mode, e, a, [x["alpha_next"] for x in h])
                    if mode != "kalman_varying":
                        assert h[e]["state_used"] == h[e - 1]["state_after_observation"]
                assert all(x["skipped_observations"] == 0 for x in h)
            if mode == "oahmm_varying":
                assert len(calls) == len(h)
                for (v, t), x in zip(calls, h):
                    assert v == x["val_mse"] and t == x["train_task"], (v, t, x)
            out.append(f"{tag}: alpha used {['-' if x is None else round(x, 3) for x in a]}")
    finally:
        T.OAHMMScheduler.observe = orig
    return "; ".join(out) + " -- each scheduled epoch used the alpha decided after the previous epoch; OA-HMM observed (val MSE, epoch-mean train task loss)"


def d2(tmp):
    """A scripted scheduler whose alpha changes after every observation (0.11, 0.12, ...): the alpha used
    in epoch e must be the value set by the e-1 observations, for both the HMM and the Kalman code paths."""
    class Scripted:
        def __init__(self, *a, **k):
            self.n = 0; self.skipped_observations = 0; self.seen = []
        def observe(self, val_mse, train_task=None):
            self.seen.append(val_mse); self.n += 1
        def get_alpha(self):
            return round(0.11 + 0.01 * self.n, 2)
        def get_state_name(self):
            return f"s{self.n}"
        def get_normalized_innovation(self):
            return float(self.n)
    out = []
    saved = (T.HMMScheduler, T.KalmanScheduler, T.OAHMMScheduler)
    T.HMMScheduler = T.KalmanScheduler = T.OAHMMScheduler = Scripted
    try:
        for mode in ("hmm_varying", "oahmm_varying", "kalman_varying"):
            _, r = run_train(tmp, "D2_" + mode, loss_mode=mode, epochs=4, **CONFIGS["F"])
            a = [x["alpha_used"] for x in r["history"]]
            assert a == [0.11, 0.12, 0.13, 0.14], (mode, a)
            if mode != "kalman_varying":
                assert [x["state_used"] for x in r["history"]] == ["s0", "s1", "s2", "s3"]
            out.append(f"{mode}: {a}")
    finally:
        T.HMMScheduler, T.KalmanScheduler, T.OAHMMScheduler = saved
    return "; ".join(out) + " (epoch e uses the alpha set after epoch e-1)"


# ----------------------------------------------------------------------------- run
if __name__ == "__main__":
    import transformers
    log(f"Stage 1 tests  src={os.path.abspath(args.src)}  gpt2={args.gpt2}  device={DEV}  "
        f"torch {torch.__version__}  transformers {transformers.__version__}")
    check("P1 positional embeddings: shape and learnable (TD, FD)", p1)
    check("P2 positional embedding added once, shared across variables (TD, FD)", p2)
    check("P3 positional embedding before self-attention: encoder is order-sensitive", p3)
    check("H1 flatten-head tensor shapes (B=2, V=7, P=32, H=768, horizon 96)", h1)
    check("H2 sequence-token order is (patch, variable); no extra tokens", h2)
    check("H3 flatten head receives only the sequence-token hidden states (E)", h3)
    check("R1 RevIN statistics and round trip", r1)
    check("R2 forecast is shift/scale-equivariant per variable; variable order preserved (E, F, B)", r2)
    check("G1 GPT-2 policy norms_pos: exactly layer norms + position embeddings trainable (E)", g1)
    check("G2 no-LLM path has no GPT-2 parameters; original model B unchanged (F, B)", g2)
    check("G3 optimizer receives exactly the trainable parameters; trainable names listed", g3)
    check("B1 forward/backward + optimizer step (E)", lambda: backward_step("E"))
    check("B2 forward/backward + optimizer step (F)", lambda: backward_step("F"))
    check("T1 TFB paths: alpha = 0 is task-only; alpha > 0 adds alpha*L_TFB and reaches the encoders", t1)
    check("S1 schedulers reject NaN/+inf/-inf without changing state; resume normally", s1)
    check("C1 configuration is explicit; invalid combinations rejected", c1)
    if not args.skip_integration:
        import gc
        _cache.clear(); gc.collect()          # free the unit-test models before training runs
        with tempfile.TemporaryDirectory() as tmp:
            check("I1 train() end to end (E, F): logs, config, checkpoint reload, metrics", lambda: i1(tmp))
            check("D1 scheduler dispatch and alpha logging in train() (all modes)", lambda: d1(tmp))
            check("D2 alpha timing with a scheduler whose alpha changes every epoch", lambda: d2(tmp))
        small_data(False)
    n_pass = sum(ok for _, ok, _ in RESULTS)
    log(f"\n{n_pass}/{len(RESULTS)} Stage 1 checks passed")
    with open(f"stage1_test_log_{args.gpt2}_{DEV.type}.txt", "w") as fh:
        fh.write("\n".join(LOG) + "\n")
    sys.exit(0 if n_pass == len(RESULTS) else 1)
