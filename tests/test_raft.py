"""
Stage X (RAFT) checks, CPU.   python tests/test_raft.py <src_dir> --csv <ETTh1.csv>
Each check prints PASS/FAIL; exit code 1 if any check fails.
"""
import argparse, os, sys, tempfile, json
import numpy as np
import pandas as pd
import torch

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("--csv", required=True)
a = ap.parse_args()
for d in ("", "data", "models", "losses"):
    sys.path.insert(0, os.path.join(a.src, d))
import train as T
import raft as R
from datasets import BenchmarkDataset, borders
from loaders import ETTh1Dataset

torch.set_num_threads(2)
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""), flush=True)


def small_model(V=7, L=512, H=96, raft=True, seed=0):
    torch.manual_seed(seed)
    build = R.build_raft_model if raft else T.build_model
    return build(d_emb=64, patch_len=16, num_features=V, pred_len=H, use_revin=True, seq_len=L,
                 head_type="flatten", use_llm=False, llm_policy="none")


# X1 proposition on random vectors
rng = torch.Generator().manual_seed(0)
worst = float("inf"); exact = True
for k in range(2000):
    gt = [torch.randn(5, generator=rng), torch.randn(3, generator=rng)]
    ga = [torch.randn(5, generator=rng) * 3, torch.randn(3, generator=rng) * 3]
    al = float(torch.rand(1, generator=rng)) * 5
    g, info = R.conflict_free_combine(gt, ga, al)
    lhs = sum((x * y).sum() for x, y in zip(g, gt)); nt2 = sum((x * x).sum() for x in gt)
    worst = min(worst, float(lhs - nt2) / float(nt2))
    if not info["conflict"]:
        exact &= all(torch.allclose(x, t + al * u) for x, t, u in zip(g, gt, ga))
check("X1 conflict-free proposition: <g, g_task> >= ||g_task||^2 (2000 random cases); no conflict -> plain sum",
      worst > -1e-5 and exact, f"min relative slack {worst:.2e}")

# X2 relational target similarity is scale-free and row-stochastic
y = torch.randn(40, 24)
P1, P2 = R.target_similarity(y), R.target_similarity(10 * y)
check("X2 target similarity: rows sum to 1, zero diagonal, invariant to target scale",
      torch.allclose(P1.sum(1), torch.ones(40), atol=1e-5) and float(P1.diag().abs().max()) == 0 and torch.allclose(P1, P2, atol=1e-6))

# X3 optimising L_RT gives representations whose similarity follows the futures (two clusters of futures)
torch.manual_seed(1)
base = torch.randn(2, 24)
yc = torch.cat([base[0] + 0.1 * torch.randn(20, 24), base[1] + 0.1 * torch.randn(20, 24)])
P = R.target_similarity(yc)
h = torch.randn(40, 32, requires_grad=True)
opt = torch.optim.Adam([h], 0.05)
l0 = float(R.relational_kl(P, h))
for _ in range(300):
    opt.zero_grad(); l = R.relational_kl(P, h); l.backward(); opt.step()
hn = torch.nn.functional.normalize(h.detach(), dim=1); C = hn @ hn.t()
within = float((C[:20, :20].sum() - 20 + C[20:, 20:].sum() - 20) / (2 * 380)); between = float(C[:20, 20:].mean())
offmean = float((C.sum() - 40) / (40 * 39))
check("X3 L_RT converges; windows with similar futures are more similar; representations stay far from orthogonal",
      float(l) < 0.1 * l0 and within > between + 0.05 and offmean > 0.3, f"KL {l0:.3f}->{float(l):.3f}; cos within {within:.2f}, between {between:.2f}")

# X4 RAFT forecaster: shapes, single FD evaluation, RevIN equivariance
f, _ = small_model()
f.eval()
calls = {"n": 0}
orig = f.fd_encoder.forward
def counted(x):
    calls["n"] += 1
    return orig(x)
f.fd_encoder.forward = counted
x = torch.randn(3, 512, 7)
xp = x.reshape(3, 32, 16, 7)
with torch.no_grad():
    yh, aux = f(xp, x)
    n_first = calls["n"]
    y2, _ = f((2 * x + 5).reshape(3, 32, 16, 7), 2 * x + 5)
f.fd_encoder.forward = orig
check("X4 RAFT forecaster: forecast (B,H,V); h_td (B*V, P*d); h_fd (B*V, d); FD evaluated once; shift/scale equivariant",
      yh.shape == (3, 96, 7) and aux["h_td"].shape == (21, 32 * 64) and aux["h_fd"].shape == (21, 64)
      and n_first == 1 and torch.allclose(y2, 2 * yh + 5, atol=1e-3), f"FD calls per forward {n_first}")

# X5 FD branch is informative: different windows give different FD embeddings (R1 failure: 1.5e-5)
xs = torch.from_numpy(np.stack([ETTh1Dataset(a.csv, flag="val")[i]["x"].numpy() for i in range(0, 1600, 100)]))
f.train()
with torch.no_grad():
    for _ in range(5):
        f(xs.reshape(16, 32, 16, 7), xs)               # populate BatchNorm running statistics
f.eval()
with torch.no_grad():
    _, aux = f(xs.reshape(16, 32, 16, 7), xs)
    z = torch.nn.functional.normalize(aux["z_F"], dim=1); D_raft = float((1 - z @ z.t()).sum() / (16 * 15))
    fo, _ = small_model(raft=False); fo.eval()
    _, ao = fo(xs.reshape(16, 32, 16, 7), xs)
    zo = torch.nn.functional.normalize(ao["z_F"], dim=1); D_old = float((1 - zo @ zo.t()).sum() / (16 * 15))
check("X5 FD embedding dispersion at initialisation: RAFT FD > original FD", D_raft > D_old,
      f"RAFT {D_raft:.2e} vs original {D_old:.2e}")

# X6 surrogate gradient equals the conflict-free direction; proposition holds on real model gradients
f, tfb = small_model()
f.train()
ds = ETTh1Dataset(a.csv, flag="train")
b = [ds[i] for i in range(4)]
xb = torch.stack([s["x_patched"] for s in b]); xr = torch.stack([s["x"] for s in b]); yb = torch.stack([s["y"] for s in b])
stats = {}
cl = R.make_compute_losses("raft", 0.3, stats)
torch.manual_seed(5)
out = cl(f, tfb, xb, xr, yb, alpha=0.3)
params = [p for p in f.parameters() if p.requires_grad]
for p in params:
    p.grad = None
out["total_loss"].backward()
g_sur = [p.grad.clone() for p in params]
torch.manual_seed(5)
fc, aux = f(xb, xr)
task = torch.nn.functional.mse_loss(fc, yb)
la = R.relational_target_loss(aux["h_td"], yb, aux["mean"], aux["std"]) + R.relational_consistency_loss(aux["h_td"], aux["h_fd"])
gt = [torch.zeros_like(p) if g is None else g for g, p in zip(torch.autograd.grad(task, params, retain_graph=True, allow_unused=True), params)]; ga = torch.autograd.grad(la, params, allow_unused=True)
g_ref, info = R.conflict_free_combine(list(gt), list(ga), 0.3)
same = all(torch.allclose(x, y, atol=1e-6, rtol=1e-4) for x, y in zip(g_sur, g_ref))
lhs = float(sum((x * y).sum() for x, y in zip(g_sur, gt))); nt2 = float(sum((x * x).sum() for x in gt))
check("X6 surrogate loss yields exactly the conflict-free gradient; <g, g_task> >= ||g_task||^2 on the real model",
      same and lhs >= nt2 * (1 - 1e-4), f"<g,g_t>/||g_t||^2 = {lhs / nt2:.4f}; conflict={info['conflict']}; cos={info['cos']:.3f}")

# X7 alpha = 0 and fdfix: total loss is the task loss
for v, al in (("raft", 0.0), ("fdfix", 0.0)):
    o = R.make_compute_losses(v, al, {})(f, tfb, xb, xr, yb, alpha=al)
    ok = torch.allclose(o["total_loss"], o["task_loss"])
check("X7 alpha = 0 (and fdfix): total loss == task loss", ok)

# X8 BenchmarkDataset reproduces the ETTh1 loader exactly
ok = True
for flag in ("train", "val", "test"):
    d1, d2 = ETTh1Dataset(a.csv, flag=flag), BenchmarkDataset(a.csv, flag=flag)
    ok &= len(d1) == len(d2)
    for i in (0, len(d1) // 2, len(d1) - 1):
        ok &= torch.equal(d1[i]["x"], d2[i]["x"]) and torch.equal(d1[i]["y"], d2[i]["y"])
check("X8 BenchmarkDataset == ETTh1Dataset on ETTh1 (all splits, sampled windows identical)", ok)

# X9 custom 70/10/20 borders (Time-Series-Library formula) and targets start at the split boundary
tmp = tempfile.mkdtemp()
n = 1000
df = pd.DataFrame(np.arange(n * 8, dtype=float).reshape(n, 8), columns=[f"c{i}" for i in range(7)] + ["OT"])
df.insert(0, "date", pd.date_range("2000-01-01", periods=n, freq="D").astype(str))
p = os.path.join(tmp, "exchange_rate.csv"); df.to_csv(p, index=False)
b1, b2 = borders("Exchange", n, 96)
dv = BenchmarkDataset(p, flag="val", seq_len=96, pred_len=24)
first_target_row = int(round(dv[0]["y"][0, 0].item() * dv.std[0] + dv.mean[0])) // 8
check("X9 custom split 70/10/20 with train-only scaling; first validation target = first validation row",
      b2 == [700, 800, 1000] and b1 == [0, 604, 704] and first_target_row == 700, f"borders {b1} {b2}")

# X10 few-shot: earliest 10 % of the training windows
d10 = BenchmarkDataset(a.csv, flag="train", percent=10); dfull = BenchmarkDataset(a.csv, flag="train")
check("X10 few-shot 10%: earliest windows, same scaling, unchanged validation",
      len(d10) == (8640 - 512) * 10 // 100 + 512 - 512 - 96 + 1 and torch.equal(d10[0]["x"], dfull[0]["x"])
      and len(BenchmarkDataset(a.csv, flag="val", percent=10)) == 2785, f"{len(d10)} windows")

# X11 end-to-end: train() with RAFT on a tiny subset (CPU)
T.ETTh1Dataset = lambda csv, flag, seq_len, pred_len, patch_len: torch.utils.data.Subset(
    BenchmarkDataset(csv, flag=flag, seq_len=seq_len, pred_len=pred_len, patch_len=patch_len), range(8))
T.build_model = R.build_raft_model
st = {}
T.compute_losses = R.make_compute_losses("raft", 0.3, st)
tmpd = tempfile.mkdtemp()
_, _, r = T.train(csv_path=a.csv, checkpoint_dir=tmpd, batch_size=4, epochs=2, patience=5, min_epochs=1,
                  loss_mode="fixed_weighted", alpha=0.3, seed=1, head_type="flatten", use_llm=False, llm_policy="none",
                  run_sanity_check=False, device="cpu", num_features=7)
check("X11 train() end to end with RAFT (2 epochs, tiny subset, CPU): finite losses, projection statistics logged",
      r["test_mse"] is not None and np.isfinite(r["test_mse"]) and st.get("steps", 0) == 4,
      f"test MSE {r['test_mse']:.3f}; steps {st.get('steps')}; conflicts {st.get('conflicts')}")

print(f"\n{sum(results)}/{len(results)} RAFT checks passed")
sys.exit(0 if all(results) else 1)
