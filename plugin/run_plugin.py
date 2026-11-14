"""
Stage P: RAFT as a plug-in objective for published forecasters (declared in configs/plugin_protocol.json).

For each backbone (PatchTST, iTransformer), dataset-horizon setting and seed, trains
  base : the backbone on the forecasting loss (its official settings)
  raft : the same backbone, same seed, with the target-aware relational loss L_RT on the representation the
         head reads and the conflict-free update g = g_task + alpha * Pi(g_aux)   (alpha = 1, tau = 0.1: the values
         selected for F in Stage X, NOT re-tuned). These backbones have no frequency branch, so L_TF does not apply.
Only validation error selects the checkpoint; the test set is evaluated once.

  python plugin/run_plugin.py --out /kaggle/working/stage_p --gpus 0,1 --per_gpu 3
  python plugin/run_plugin.py --selftest
"""
import argparse, copy, glob, json, math, os, random, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SRC = os.path.join(REPO, "src")
PROT = os.environ.get("PLUGIN_PROT", os.path.join(REPO, "configs", "plugin_protocol.json"))
for d in ("", "data", "models", "losses"):
    sys.path.insert(0, os.path.join(SRC, d))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "scripts"))


def job_id(j):
    return f"{j['backbone']}__{j['setting']}__{j['arm']}__s{j['seed']}"


def parse_job(s):
    b, st, arm, seed = s.split("__")
    return dict(backbone=b, setting=st, arm=arm, seed=int(seed[1:]))


def jobs(P, phase, sel=None, only=None):
    """phase 'pilot': alpha candidates on the pilot settings; phase 'main': base vs the selected alpha per backbone."""
    J = []
    for s in P["seeds"]:                                   # seed-major: an interrupted session leaves complete pairs
        for b in P["backbones"]:
            if phase == "pilot":
                for st in P["pilot"]["settings"]:
                    for a in ["base"] + [f"raft_a{x:g}" for x in P["pilot"]["alphas"]]:
                        J.append(dict(backbone=b, setting=st, arm=a, seed=s))
            else:
                for st in P["settings"]:
                    for a in ("base", f"raft_a{sel[b]:g}"):
                        J.append(dict(backbone=b, setting=st, arm=a, seed=s))
    if only:
        keep = set(only.split(","))
        J = [j for j in J if j["backbone"] in keep or j["setting"] in keep]
    return J


def select(P, out):
    """Declared rule (same as Stage X): per backbone, the alpha with the lowest mean over the pilot settings of
    mean best-validation MSE / that of 'base' (same seeds). Ties (< 1e-4): smaller alpha. Test data not used."""
    import statistics as S
    sel, table = {}, {}
    for b in P["backbones"]:
        sc = {}
        for x in P["pilot"]["alphas"]:
            r = []
            for st in P["pilot"]["settings"]:
                v = lambda arm: [json.load(open(p))["best_val_mse"] for p in
                                 glob.glob(os.path.join(out, "runs", f"{b}__{st}__{arm}__s*", "result.json"))]
                vb, va = v("base"), v(f"raft_a{x:g}")
                if vb and va:
                    r.append(S.mean(va) / S.mean(vb))
            if len(r) == len(P["pilot"]["settings"]):
                sc[x] = S.mean(r)
        best = min(sc.values())
        sel[b] = min(x for x in sc if sc[x] - best < 1e-4)
        table[b] = sc
    json.dump(dict(selected=sel, mean_val_ratio=table), open(os.path.join(out, "selection.json"), "w"), indent=1)
    print("SELECTED alpha per backbone:", sel, "ratios:", table, flush=True)
    return sel


def seed_all(s):
    import numpy as np, torch
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def worker(job, out, data_dir, dry=False):
    import torch
    import torch.nn.functional as Fn
    from torch.utils.data import DataLoader
    import raft as R
    import backbones as BB
    from datasets import BenchmarkDataset, SPECS
    from metrics import evaluate_full
    P = json.load(open(PROT))
    dry = dry or os.environ.get("PLUGIN_DRY") == "1"
    j = parse_job(job)
    ds_name, H = j["setting"].split("_H"); H = int(H)
    V = SPECS[ds_name][2]
    dev = "cpu" if dry else os.environ.get("PLUGIN_DEVICE", "cuda")
    seed_all(j["seed"])
    model, cfg = BB.build(j["backbone"], ds_name, H, V)
    model = model.to(dev)
    L = cfg["seq_len"]
    csv = os.path.join(data_dir, SPECS[ds_name][0])
    mk = lambda flag: BenchmarkDataset(csv, flag=flag, seq_len=L, pred_len=H, patch_len=L, name=ds_name)
    tr, va, te = mk("train"), mk("val"), mk("test")
    T = P["training"][j["backbone"]]
    bs, epochs, patience = cfg["batch"], T["max_epochs"], T["patience"]
    if dry:
        sub = lambda d, n: torch.utils.data.Subset(d, range(min(len(d), n)))
        tr_l = DataLoader(sub(tr, 64), batch_size=16, shuffle=True, drop_last=True)
        va_s, te_s = sub(va, 32), sub(te, 32)
        for s_ in (va_s, te_s):
            s_.mean, s_.std = tr.mean, tr.std
        epochs = 2
    else:
        tr_l = DataLoader(tr, batch_size=bs, shuffle=True, drop_last=len(tr) > bs, num_workers=2)
        va_s, te_s = va, te
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    if j["arm"] == "base":
        alpha = 0.0
    elif j["arm"] == "raft":
        alpha = float(os.environ.get("PLUGIN_ALPHA", 1.0))
    else:                                               # raft_a<alpha> (pilot candidates)
        alpha = float(j["arm"].split("_a", 1)[1])
    params = [p for p in model.parameters() if p.requires_grad]
    stats = dict(steps=0, conflicts=0, cos_sum=0.0)

    def val_mse():
        model.eval(); se = n = 0.0
        with torch.no_grad():
            for b in DataLoader(va_s, batch_size=256, shuffle=False):
                yh, _ = model(None, b["x"].to(dev))
                se += float(((yh - b["y"].to(dev)) ** 2).sum()); n += yh.numel()
        return se / n

    best, best_ep, bad, state, hist = float("inf"), 0, 0, None, []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train(); tl = 0.0; nb = 0
        for b in tr_l:
            x, y = b["x"].to(dev), b["y"].to(dev)
            yh, aux = model(None, x)
            task = Fn.mse_loss(yh, y)
            opt.zero_grad(set_to_none=True)
            if alpha > 0:
                l_rt = R.relational_target_loss(aux["h"], y, aux["mean"], aux["std"], tau=P["raft"]["tau"],
                                                max_n=P["raft"]["max_n"])
                g_t = torch.autograd.grad(task, params, retain_graph=True, allow_unused=True)
                g_a = torch.autograd.grad(l_rt, params, allow_unused=True)
                g_t = [torch.zeros_like(p) if g is None else g for g, p in zip(g_t, params)]
                g, info = R.conflict_free_combine(g_t, list(g_a), alpha)
                for p, gi in zip(params, g):
                    p.grad = gi.detach()
                stats["steps"] += 1; stats["conflicts"] += int(info["conflict"]); stats["cos_sum"] += info["cos"]
            else:
                task.backward()
            opt.step()
            tl += float(task.detach()); nb += 1
        v = val_mse()
        hist.append(dict(epoch=ep, train_task=tl / max(nb, 1), val_mse=v))
        if v < best:
            best, best_ep, bad = v, ep, 0
            state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        for g_ in opt.param_groups:                       # official schedules: PatchTST 'type3', iTransformer 'type1'
            if j["backbone"] == "patchtst":
                g_["lr"] = cfg["lr"] * (0.9 ** max(0, ep - 2))
            else:
                g_["lr"] = cfg["lr"] * (0.5 ** ep)
        if bad >= patience:
            break
    model.load_state_dict(state)
    minutes = (time.time() - t0) / 60
    res = dict(stage="P", backbone=j["backbone"], setting=j["setting"], dataset=ds_name, pred_len=H, seq_len=L,
               arm=j["arm"], alpha=alpha, seed=j["seed"], best_epoch=best_ep, epochs_run=len(hist), best_val_mse=best,
               minutes=round(minutes, 2), n_params=sum(p.numel() for p in params), history=hist,
               config={k: v for k, v in cfg.items()})
    if stats["steps"]:
        res.update(proj_steps=stats["steps"], proj_conflict_rate=stats["conflicts"] / stats["steps"],
                   proj_mean_cos=stats["cos_sum"] / stats["steps"])
    model.eval()
    for split, d in (("val", va_s), ("test", te_s)):
        res.update({f"{split}_{k}": v for k, v in evaluate_full(model, DataLoader(d, batch_size=256, shuffle=False), dev).items()})
    run_dir = os.path.join(out, "runs", job)
    os.makedirs(run_dir, exist_ok=True)
    json.dump(res, open(os.path.join(run_dir, "result.json"), "w"), indent=1)
    print("RESULT", job, f"best_ep {best_ep} val {best:.4f} test MSE {res['test_mse']:.4f} ({minutes:.1f} min)", flush=True)


def prepare(P, out):
    import stage_x_raft as SX                            # same download + checksum code as Stage X
    from datasets import SPECS
    data_dir = os.path.join(out, "data"); os.makedirs(data_dir, exist_ok=True)
    import shutil, hashlib
    rec = {}
    for ds in sorted({st.split("_H")[0] for st in P["settings"]}):
        fn = SPECS[ds][0]
        p = SX.find_or_get(fn, out)
        dst = os.path.join(data_dir, fn)
        if os.path.abspath(p) != os.path.abspath(dst):
            shutil.copy(p, dst)
        sha = hashlib.sha256(open(dst, "rb").read()).hexdigest()
        if fn in SX.KNOWN_SHA and sha != SX.KNOWN_SHA[fn]:
            sys.exit(f"STOP: {fn} differs from the file used in Stage X")
        rec[fn] = sha
        print(f"data {fn}: sha256 {sha[:12]}", flush=True)
    return data_dir, rec


def summarize(out):
    import statistics as S
    rows = [json.load(open(p)) for p in glob.glob(os.path.join(out, "runs", "*", "result.json"))]
    by = {}
    for r in rows:
        by.setdefault((r["backbone"], r["setting"]), {}).setdefault(r["arm"], {})[r["seed"]] = r
    sel = {}
    if os.path.exists(os.path.join(out, "selection.json")):
        sel = json.load(open(os.path.join(out, "selection.json")))["selected"]
    L = [f"Stage P: {len(rows)} runs", ""]
    for (b, st), d in sorted(by.items()):
        ra = f"raft_a{sel[b]:g}" if b in sel else None
        if "base" in d and ra in d:
            d = dict(d, raft=d[ra])
            s = sorted(set(d["base"]) & set(d["raft"]))
            if s:
                dr = [d["raft"][k]["test_rmse"] - d["base"][k]["test_rmse"] for k in s]
                L.append(f"{b:12s} {st:14s} n={len(s)} base RMSE {S.mean(d['base'][k]['test_rmse'] for k in s):.4f} "
                         f"raft {S.mean(d['raft'][k]['test_rmse'] for k in s):.4f} d {S.mean(dr):+.4f} better {sum(x < 0 for x in dr)}/{len(s)}")
    open(os.path.join(out, "summary.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


def selftest():
    """CPU checks: shapes, h is what the head reads, conflict-free direction, one tiny run per backbone/arm."""
    import torch, tempfile, shutil
    import backbones as BB, raft as R
    ok = True
    for b in ("patchtst", "itransformer"):
        for ds in ("ETTh1", "ILI"):
            m, c = BB.build(b, ds, 24, 7)
            x = torch.randn(4, c["seq_len"], 7) * 3 + 5
            y, aux = m(None, x)
            good = y.shape == (4, 24, 7) and aux["h"].shape[0] == 28 and aux["mean"].shape == (4, 1, 7)
            # the head reads h: perturbing h-only path changes output -> check head input size
            hd = aux["h"].shape[1]
            exp = (m.n_patches * m.d_model) if b == "patchtst" else m.proj.in_features
            good &= hd == exp
            # RevIN: output invariant to shifting the input level by a constant (level is restored)
            y2, _ = m.eval()(None, x + 10.0); y1, _ = m(None, x)
            good &= bool(torch.allclose(y2 - 10.0, y1, atol=1e-3))
            print(f"[{'OK' if good else 'FAIL'}] {b} {ds}: shapes, head input = h, level equivariance"); ok &= good
    torch.manual_seed(0); c = True
    for _ in range(200):                                  # random pairs, many with conflict
        gt = [torch.randn(50, dtype=torch.float64)]; ga = [torch.randn(50, dtype=torch.float64) - 2 * gt[0] * torch.rand(1, dtype=torch.float64)]
        g, info = R.conflict_free_combine(gt, ga, 3.0)
        n2 = float((gt[0] * gt[0]).sum())
        c &= float((g[0] * gt[0]).sum()) >= n2 * (1 - 1e-9)
    print(f"[{'OK' if c else 'FAIL'}] conflict-free direction: <g, g_task> >= ||g_task||^2 (conflict={info['conflict']})"); ok &= c
    csv = None
    for p in glob.glob(os.path.join(REPO, "data", "ETTh1.csv")) + glob.glob("/kaggle/input/**/ETTh1.csv", recursive=True):
        csv = p; break
    if csv is None:
        import stage_x_raft as SX
        csv = SX.find_or_get("ETTh1.csv", tempfile.mkdtemp())
    tmp = tempfile.mkdtemp(); dd = os.path.join(tmp, "data"); os.makedirs(dd); shutil.copy(csv, dd)
    for b in ("patchtst", "itransformer"):
        for a in ("base", "raft_a1"):
            jid = job_id(dict(backbone=b, setting="ETTh1_H24", arm=a, seed=1))
            worker(jid, tmp, dd, dry=True)
            r = json.load(open(os.path.join(tmp, "runs", jid, "result.json")))
            good = math.isfinite(r["test_mse"]) and (a == "base" or r.get("proj_steps", 0) > 0)
            print(f"[{'OK' if good else 'FAIL'}] tiny run {b} {a}: test MSE {r['test_mse']:.3f}"); ok &= good
    print("PLUGIN SELFTEST", "PASSED" if ok else "FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out"); ap.add_argument("--gpus", default="0,1"); ap.add_argument("--per_gpu", type=int, default=3)
    ap.add_argument("--job"); ap.add_argument("--data_dir"); ap.add_argument("--only")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.job:
        return worker(a.job, a.out, a.data_dir)
    P = json.load(open(PROT))
    os.makedirs(os.path.join(a.out, "runs"), exist_ok=True)
    data_dir, rec = prepare(P, a.out)
    git = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    json.dump(dict(protocol=P, data=rec, git_commit=git), open(os.path.join(a.out, "protocol.json"), "w"), indent=1)
    lanes = [(g, k) for k in range(a.per_gpu) for g in a.gpus.split(",")]

    def run(J, tag):
        done = {os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(a.out, "runs", "*", "result.json"))}
        todo = [j for j in J if job_id(j) not in done]
        print(f"[{tag}] {len(J)} jobs, {len(J) - len(todo)} already done, {len(todo)} to run", flush=True)
        Q = {ln: [] for ln in lanes}
        for i, j in enumerate(todo):
            Q[lanes[i % len(lanes)]].append(j)
        procs = []
        for (g, k), q in Q.items():
            if q:
                cmd = " ; ".join(f'"{sys.executable}" "{os.path.abspath(__file__)}" --out "{a.out}" --data_dir "{data_dir}" '
                                 f'--job {job_id(j)} > "{a.out}/runs/{job_id(j)}.log" 2>&1' for j in q)
                procs.append(subprocess.Popen(cmd, shell=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=g, PYTHONUNBUFFERED="1")))
        t0 = time.time()
        while any(p.poll() is None for p in procs):
            time.sleep(60)
            n = sum(os.path.exists(os.path.join(a.out, "runs", job_id(j), "result.json")) for j in J)
            print(f"[{tag} {(time.time() - t0) / 60:.0f} min] {n}/{len(J)} result files", flush=True)

    run(jobs(P, "pilot", only=a.only), "pilot")
    sel = select(P, a.out)
    run(jobs(P, "main", sel, a.only), "main")
    summarize(a.out)


if __name__ == "__main__":
    main()
