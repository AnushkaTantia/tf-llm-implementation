"""
Stage J checks (CPU).   python journal/test_j.py <path-to-src> <path-to-journal>
"""
import importlib.util, itertools, os, sys
SRC, JDIR = sys.argv[1], sys.argv[2]
for d in ("", "data", "models", "losses"):
    sys.path.insert(0, os.path.join(SRC, d))
sys.path.insert(0, JDIR)
import torch
from transformers import GPT2Config, GPT2Model
GPT2Model.from_pretrained = classmethod(lambda cls, name, **k: GPT2Model(GPT2Config(n_layer=1, n_head=2)))
import raft as R
import raft_j as RJ

FAILS = []


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}", flush=True)
    if not cond:
        FAILS.append(name)


torch.manual_seed(0)

# 1. RnC equals its definition (brute force), with ties in the target distances
n, dh, H = 9, 5, 4
h = torch.randn(n, dh, dtype=torch.float64)
y = torch.randn(n, H, dtype=torch.float64)
y[3] = y[1]; y[7] = y[1]                         # ties
tau = 0.1
hn = torch.nn.functional.normalize(h, dim=1); s = hn @ hn.t() / tau
d = torch.cdist(y, y).pow(2) / H
tot, cnt = 0.0, 0
for i in range(n):
    for j in range(n):
        if j == i:
            continue
        den = torch.logsumexp(torch.stack([s[i, k] for k in range(n) if k != i and d[i, k] >= d[i, j]]), 0)
        tot += -(s[i, j] - den); cnt += 1
check("RnC loss equals its definition (incl. ties)", abs(float(RJ.rnc_loss(h, y, tau)) - float(tot / cnt)) < 1e-9)

# 2. combiners
gt = [torch.randn(6, dtype=torch.float64), torch.randn(3, dtype=torch.float64)]
ga = [-1.5 * gt[0] + 0.3 * torch.randn(6, dtype=torch.float64), -gt[1]]          # conflicting
dot = lambda a, b: sum((x * y).sum() for x, y in zip(a, b))
g, info = RJ.pcgrad_combine(gt, ga, 1.0)
gt_p = [t - dot(gt, ga) / dot(ga, ga) * a for t, a in zip(gt, ga)]
ga_p = [a - dot(gt, ga) / dot(gt, gt) * t for t, a in zip(gt, ga)]
check("PCGrad: conflict detected and g = proj(g_t) + proj(g_a)",
      info["conflict"] and all(torch.allclose(x, p + q) for x, p, q in zip(g, gt_p, ga_p)))
check("PCGrad: projected gradients are orthogonal to the other task",
      abs(float(dot(gt_p, ga))) < 1e-9 and abs(float(dot(ga_p, gt))) < 1e-9)
g, info = RJ.gsgate_combine(gt, ga, 1.0)
check("gradient-similarity gating drops a conflicting g_aux", info["conflict"] and all(torch.equal(x, t) for x, t in zip(g, gt)))
ga2 = [0.5 * t for t in gt]
for name, fn in [("PCGrad", RJ.pcgrad_combine), ("gating", RJ.gsgate_combine), ("sum", RJ.sum_combine), ("cf", R.conflict_free_combine)]:
    g, info = fn(gt, ga2, 0.7)
    check(f"{name}: no conflict -> g = g_t + a*g_a", (not info["conflict"]) and all(torch.allclose(x, t + 0.7 * a) for x, t, a in zip(g, gt, ga2)))
g, _ = R.conflict_free_combine(gt, ga, 3.0)
check("cf: <g, g_t> >= ||g_t||^2 under conflict", float(dot(g, gt)) >= float(dot(gt, gt)) - 1e-9)

# 3. runner name lists match the module
spec = importlib.util.spec_from_file_location("run_j", os.path.join(JDIR, "run_j.py"))
rj = importlib.util.module_from_spec(spec); spec.loader.exec_module(rj)
check("run_j AUX/COMB names equal raft_j", set(rj.AUX_NAMES) == set(RJ.AUX) and set(rj.COMB_NAMES) == set(RJ.COMBINERS))
pa = rj.arm_parse
check("arm parsing", pa("none")["kind"] == "none" and pa("tfb_a0.01")["alpha"] == 0.01
      and pa("raft_a3") == dict(llm=False, kind="raft", variant="raft", aux=None, comb=None, alpha=3.0, tau=0.1)
      and pa("raft_cf_a1_tau0.05")["kind"] == "j" and pa("raft_cf_a1_tau0.05")["tau"] == 0.05
      and pa("rnctf_cf_a0.3")["aux"] == "rnctf" and pa("e_raft_a1")["llm"] and pa("e_raft_a1")["kind"] == "raft"
      and pa("raft_pc_a1")["comb"] == "pc")

# 4. the Stage J path with (raft, cf, tau 0.1) gives exactly the gradient of src/raft.py
def grads(make):
    torch.manual_seed(1)
    f, tfb = R.build_raft_model(num_features=3, pred_len=16, seq_len=64, head_type="flatten", use_llm=False, llm_policy="none")
    f.train()
    torch.manual_seed(2)
    x = torch.randn(4, 64, 3); xp = x.reshape(4, 4, 16, 3); yy = torch.randn(4, 16, 3)
    torch.manual_seed(3)
    out = make({})(f, tfb, xp, x, yy)
    out["total_loss"].backward()
    return [(p.grad.clone() if p.grad is not None else torch.zeros_like(p)) for p in f.parameters() if p.requires_grad], \
        float(out["total_loss"].detach())
g1, v1 = grads(lambda st: R.make_compute_losses("raft", 0.7, st))
g2, v2 = grads(lambda st: RJ.make_compute_losses_j("raft", "cf", 0.7, st, tau=0.1))
check("Stage J (raft, cf) reproduces src/raft.py exactly", len(g1) == len(g2) and all(torch.allclose(a, b, atol=1e-7) for a, b in zip(g1, g2)) and abs(v1 - v2) < 1e-6)
g3, _ = grads(lambda st: RJ.make_compute_losses_j("raft", "sum", 0.7, st))
g4, _ = grads(lambda st: R.make_compute_losses("raft_noproj", 0.7, st))
check("Stage J (raft, sum) equals the raft_noproj ablation", all(torch.allclose(a, b, atol=1e-6) for a, b in zip(g3, g4)))
for aux in ("rnc", "rnctf"):
    for comb in ("cf", "pc", "gs"):
        g5, v5 = grads(lambda st: RJ.make_compute_losses_j(aux, comb, 0.3, st))
        check(f"{aux}/{comb}: finite loss and gradients", v5 == v5 and all(torch.isfinite(t).all() for t in g5))

# 5. RAFT on the GPT-2 forecaster E (tiny random GPT-2 for the test)
torch.manual_seed(0)
f, tfb = RJ.build_raft_model_any(num_features=3, pred_len=16, seq_len=64, head_type="flatten", use_llm=True, llm_policy="norms_pos")
x = torch.randn(2, 64, 3); xp = x.reshape(2, 4, 16, 3); yy = torch.randn(2, 16, 3)
out = RJ.make_compute_losses_j("raft", "cf", 1.0, {})(f, tfb, xp, x, yy)
out["total_loss"].backward()
fc, aux = f(xp, x)
ln = [p for n_, p in f.backbone.gpt2.named_parameters() if p.requires_grad]
check("E + RAFT: forecast shape, aux shapes", fc.shape == (2, 16, 3) and aux["h_td"].shape == (6, 4 * 64) and aux["h_fd"].shape == (6, 64))
check("E + RAFT: GPT-2 trainable (norms/pos) parameters receive gradients", len(ln) > 0 and all(p.grad is not None for p in ln))
check("E + RAFT: other GPT-2 parameters stay frozen",
      all(not p.requires_grad for n_, p in f.backbone.gpt2.named_parameters() if ("ln" not in n_ and "wpe" not in n_)))

print(f"\n{len(FAILS)} failed" if FAILS else "\nall Stage J checks passed")
sys.exit(1 if FAILS else 0)
