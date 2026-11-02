"""
Stage R1 (reassessment after S3): why does the TFB objective fail, and is the failure general?

    python scripts/stage_r_probes.py --part mechanism|scope_f|scope_e --out <dir> [--csv ETTh1.csv] [--gpus 0,1]
    python scripts/stage_r_probes.py --part selftest            (CPU checks of this script; run by the preflight)

Protocol: configs/stage_r_protocol.json (declared before any R1 run). The forecaster, code and training protocol
are exactly those of S2/S3 (configs/frozen_forecaster.json; src/ is checked to be unchanged). This script only
changes, per run: the dataset (ETTh1/ETTh2), the horizon, the fixed TFB weight alpha, which TFB component is
optimised (part mechanism; done by replacing train.tfb_total_loss inside the run process, src/ untouched), and the
model (E only in part scope_e). After training it computes the declared probes on the selected checkpoint:
  P1 relative circular-shift sensitivity of z_F, z_T and the TD patch representation,
  P2 linear decodability of the hour-of-day phase from z_T and from the last TD patch embedding,
  P3 (alpha = 0 runs of part mechanism) cosine between task-loss and TFB-component gradients on the encoders.
Per run: run_config.json, epoch_log.json (written by train()), result.json. No checkpoints are kept.
"""
import argparse, csv, glob, hashlib, json, math, os, statistics, subprocess, sys, time, urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")
CFG = os.path.join(REPO, "configs", "frozen_forecaster.json")
PROT = os.path.join(REPO, "configs", "stage_r_protocol.json")
STAGE2_COMMIT = "b32fe77d4c6c496ace27c74cc63dbb5ecaa93e37"
ETTH2_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv"
ETTH2_SHA = "a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b"
LAM = 0.5
TAUS = (1, 4, 12)


# ----------------------------------------------------------------------------------------------- jobs
def jobs_for(part, C, Pr):
    """Ordered seed by seed, so that an interrupted session still leaves complete paired sets."""
    P = Pr["parts"]
    J = []
    if part == "mechanism":
        for s in P["mechanism"]["seeds"]:
            for arm in P["mechanism"]["arms"]:
                J.append(dict(part=part, setting="ETTh1_H96", arm=arm, seed=s))
    elif part == "scope_f":
        for s in C["seeds"]:
            for name, st in P["scope_f"]["settings"].items():
                for a in st["alphas"]:
                    J.append(dict(part=part, setting=name, arm=f"a{a:g}", seed=s))
    elif part == "scope_e":
        for s in P["scope_e"]["seeds"]:
            for a in P["scope_e"]["alphas"]:
                J.append(dict(part=part, setting="ETTh1_H96_E", arm=f"a{a:g}", seed=s))
    return J


def job_id(j):
    return f"{j['setting']}__{j['arm']}__s{j['seed']}"


def parse_job(s):
    setting, arm, seed = s.split("__")
    return dict(setting=setting, arm=arm, seed=int(seed[1:]))


def arm_spec(arm):
    """-> (alpha, component) with component in {'full', 'T', 'F', 'B'}."""
    if arm == "a0":
        return 0.0, "full"
    comp, a = (arm.split("_a") if "_a" in arm else ("full", arm[1:]))
    return float(a), comp


def setting_spec(setting):
    if setting == "ETTh1_H96":
        return "ETTh1", 96, False
    if setting == "ETTh1_H96_E":
        return "ETTh1", 96, True
    if setting == "ETTh2_H96":
        return "ETTh2", 96, False
    if setting == "ETTh1_H336":
        return "ETTh1", 336, False
    raise ValueError(setting)


# ----------------------------------------------------------------------------------------------- probes
def component_loss(comp):
    """Replacement for train.tfb_total_loss: keeps each term's coefficient inside the full objective."""
    if comp == "full":
        return None
    def f(l_t, l_f, l_b, lam=LAM):
        return {"T": lam * l_t, "F": lam * l_f, "B": (1 - lam) * l_b}[comp]
    return f


def encode(f, xb):
    """Encoder outputs exactly as in the forecaster's forward pass (RevIN -> TD / FD encoders)."""
    import torch
    if f.use_revin:
        mean, std = f.revin.stats(xb)
        xb = f.revin.norm(xb, mean, std)
    B, L, V = xb.shape
    xp = xb.reshape(B, L // f.patch_len, f.patch_len, V)          # = loaders.patchify per window
    z_T, pe = f.td_encoder(xp)                                     # (B, d), (B, P, V, d)
    z_F, _ = f.fd_encoder(xb)
    h_td = pe.mean(2).reshape(B, -1)                                # TD representation per patch, averaged over variables
    h_last = pe[:, -1].mean(1)                                      # last patch
    return dict(z_T=z_T, z_F=z_F, h_TD=h_td, h_last=h_last)


def window_hours(csv_path, flag, seq_len, pred_len, n):
    """Hour of day of the first forecast step for each window of a split (same borders as src/data/loaders.py)."""
    import pandas as pd
    d = pd.to_datetime(pd.read_csv(csv_path, usecols=["date"])["date"])
    nt, nv = 12 * 30 * 24, 4 * 30 * 24
    b1 = {"train": 0, "val": nt - seq_len, "test": nt + nv - seq_len}[flag]
    return d.iloc[b1 + seq_len: b1 + seq_len + n].dt.hour.to_numpy()


def collect(f, ds, dev, bs=256):
    import torch
    out = {}
    with torch.no_grad():
        for i in range(0, len(ds), bs):
            xb = torch.stack([ds[k]["x"] for k in range(i, min(i + bs, len(ds)))]).to(dev)
            for k, v in encode(f, xb).items():
                out.setdefault(k, []).append(v.double().cpu())
    return {k: torch.cat(v) for k, v in out.items()}


def shift_sensitivity(f, ds, dev, bs=256):
    """P1: R_e = mean_tau mean cosdist(e(x), e(roll(x, tau))) / mean cosdist(e(x), e(x'))."""
    import numpy as np, torch
    n = len(ds)
    rng = np.random.default_rng(0)
    perm = np.roll(rng.permutation(n), 1)
    base = collect(f, ds, dev, bs)
    cd = lambda a, b: (1 - torch.nn.functional.cosine_similarity(a, b, dim=1))
    ref = {k: float(cd(v, v[perm]).mean()) for k, v in base.items()}
    res = {}
    for tau in TAUS:
        acc = {}
        with torch.no_grad():
            for i in range(0, n, bs):
                xb = torch.stack([ds[k]["x"] for k in range(i, min(i + bs, n))]).to(dev)
                for k, v in encode(f, torch.roll(xb, shifts=tau, dims=1)).items():
                    acc.setdefault(k, []).append(v.double().cpu())
        for k in base:
            res.setdefault(k, []).append(float(cd(base[k], torch.cat(acc[k])).mean()) / max(ref[k], 1e-12))
    return {f"R_{k}": float(np.mean(v)) for k, v in res.items()} | {f"R_{k}_per_tau": v for k, v in res.items()} \
        | {f"refdist_{k}": ref[k] for k in ref}


def ridge_r2(Xtr, Ytr, Xte, Yte, lam=1.0):
    import numpy as np
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = Ytr.mean(0)
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (Ytr - ym))
    P = B @ W + ym
    r2 = 1 - ((Yte - P) ** 2).sum(0) / ((Yte - Yte.mean(0)) ** 2).sum(0)
    return float(np.mean(r2))


def phase_probe(f, ds_tr, ds_va, hr_tr, hr_va, dev):
    """P2: R^2 of (sin, cos)(2 pi h / 24) from z_T and from h_last, ridge fitted on train windows, scored on val."""
    import numpy as np
    tr, va = collect(f, ds_tr, dev), collect(f, ds_va, dev)
    Y = lambda h: np.stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)], 1)
    return {f"phaseR2_{k}": ridge_r2(tr[k].numpy(), Y(hr_tr), va[k].numpy(), Y(hr_va)) for k in ("z_T", "h_last", "z_F")}


def grad_alignment(T, f, tfb, ds_tr, dev, n_batches=20):
    """P3: cos(grad task, grad component) on TD- and FD-encoder parameters, averaged over batches."""
    import torch
    from torch.utils.data import DataLoader
    import random as _r, numpy as _np
    _r.seed(0); _np.random.seed(0); torch.manual_seed(0)
    f.eval(); tfb.eval()
    groups = {"TD": [p for p in f.td_encoder.parameters() if p.requires_grad],
              "FD": [p for p in f.fd_encoder.parameters() if p.requires_grad]}
    sums = {}
    dl = DataLoader(ds_tr, batch_size=32, shuffle=False, drop_last=True)
    for b_i, b in enumerate(dl):
        if b_i >= n_batches:
            break
        xp, xr, y = b["x_patched"].to(dev), b["x"].to(dev), b["y"].to(dev)
        forecast, aux = f(xp, xr)
        task = torch.nn.functional.mse_loss(forecast, y)
        z_T, z_F = aux["z_T"], aux["z_F"]
        z_T_aug, _ = f.td_encoder(T.apply_td_augmentation(aux["x_patched_in"]))
        z_F_aug, _ = f.fd_encoder(T.apply_fd_augmentation(aux["x_raw_in"]))
        p_T, p_F = tfb(z_T, z_F); p_Ta, p_Fa = tfb(z_T_aug, z_F_aug)
        comps = {"T": LAM * T.nt_xent_loss(z_T, z_T_aug), "F": LAM * T.nt_xent_loss(z_F, z_F_aug),
                 "B": (1 - LAM) * T.balance_loss(p_T, p_Ta, p_F, p_Fa)}
        for g, ps in groups.items():
            gt = torch.autograd.grad(task, ps, retain_graph=True, allow_unused=True)
            vt = torch.cat([(x if x is not None else torch.zeros_like(p)).flatten() for x, p in zip(gt, ps)])
            for c, l in comps.items():
                gc = torch.autograd.grad(l, ps, retain_graph=True, allow_unused=True)
                vc = torch.cat([(x if x is not None else torch.zeros_like(p)).flatten() for x, p in zip(gc, ps)])
                if float(vc.norm()) < 1e-12 or float(vt.norm()) < 1e-12:
                    continue
                sums.setdefault(f"cos_task_{c}_{g}", []).append(float(torch.nn.functional.cosine_similarity(vt, vc, dim=0)))
    return {k: sum(v) / len(v) for k, v in sums.items()}


# ----------------------------------------------------------------------------------------------- worker
def worker(args):
    sys.path.insert(0, SRC)
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    import train as T
    from metrics import evaluate_full
    j = parse_job(args.job)
    C = json.load(open(CFG)); A, R = C["architecture"], C["training"]
    dataset, pred_len, use_e = setting_spec(j["setting"])
    alpha, comp = arm_spec(j["arm"])
    seed = j["seed"]
    csv_path = args.csv if dataset == "ETTh1" else args.csv2
    seq_len = C["data"]["seq_len"]
    E = C["confirmation_model"]
    head, use_llm, policy = (E["head_type"], E["use_llm"], E["llm_policy"]) if use_e else \
        (A["head_type"], A["use_llm"], A["llm_policy"])
    DRY = os.environ.get("STAGE_DRY") == "1"
    dev = "cpu" if DRY else "cuda"
    if DRY:
        _len = T.ETTh1Dataset.__len__; T.ETTh1Dataset.__len__ = lambda self: min(_len(self), 16)
        R = dict(R, max_epochs=2, batch_size=4, min_epochs=1)
    if DRY and use_e and os.environ.get("STAGE_GPT2_RANDOM") == "1":     # local plumbing test without internet only
        from transformers import GPT2Model, GPT2Config
        GPT2Model.from_pretrained = classmethod(lambda cls, *a, **k: cls(GPT2Config()))
    patch = component_loss(comp)
    if patch is not None:
        T.tfb_total_loss = patch                      # this run process only; src/ is unchanged
    run_dir = os.path.join(args.out, "runs", job_id(j))
    if not DRY:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    f, tfb, r = T.train(csv_path=csv_path, checkpoint_dir=run_dir, seq_len=seq_len, pred_len=pred_len,
                        batch_size=R["batch_size"], learning_rate=R["learning_rate"], weight_decay=R["weight_decay"],
                        epochs=R["max_epochs"], patience=R["early_stopping_patience"], min_epochs=R["min_epochs"],
                        loss_mode="fixed_weighted", alpha=alpha, lam=LAM, use_revin=A["use_revin"], seed=seed,
                        head_type=head, use_llm=use_llm, llm_policy=policy, run_sanity_check=False, device=dev)
    minutes = (time.time() - t0) / 60
    f.eval()
    out = dict(part=args.part, setting=j["setting"], dataset=dataset, pred_len=pred_len, model="E" if use_e else "F",
               arm=j["arm"], alpha=alpha, component=comp, seed=seed, best_epoch=r["best_epoch"],
               epochs_run=r["epochs_run"], best_val_mse=r["best_val_mse"], minutes=round(minutes, 1),
               peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if not DRY else 0.0,
               config=E["name"] if use_e else C["name"], csv=os.path.basename(csv_path))
    dss = {}
    for split in ("train", "val", "test"):
        dss[split] = T.ETTh1Dataset(csv_path, flag=split, seq_len=seq_len, pred_len=pred_len, patch_len=A["patch_len"])
    for split in ("val", "test"):
        dl = DataLoader(dss[split], batch_size=64, shuffle=False)
        out.update({f"{split}_{k}": v for k, v in evaluate_full(f, dl, dev).items()})
    se_t = se_v = None
    with torch.no_grad():
        for b in DataLoader(dss["test"], batch_size=64, shuffle=False):
            y, _ = f(b["x_patched"].to(dev), b["x"].to(dev))
            e = (y.double().cpu() - b["y"].double()) ** 2
            se_t = e.sum((0, 2)) if se_t is None else se_t + e.sum((0, 2))
            se_v = e.sum((0, 1)) if se_v is None else se_v + e.sum((0, 1))
    n = len(dss["test"])
    out["test_mse_per_step"] = [round(float(v) / (n * 7), 6) for v in se_t]
    out["test_mse_per_variable"] = [round(float(v) / (n * pred_len), 6) for v in se_v]
    # ---- probes (declared): part mechanism and scope_e
    if args.part in ("mechanism", "scope_e"):
        hr_tr = window_hours(csv_path, "train", seq_len, pred_len, len(dss["train"]))
        hr_va = window_hours(csv_path, "val", seq_len, pred_len, len(dss["val"]))
        out.update(shift_sensitivity(f, dss["val"], dev))
        out.update(phase_probe(f, dss["train"], dss["val"], hr_tr, hr_va, dev))
        if args.part == "mechanism" and j["arm"] == "a0":
            out.update({f"{k}_selected": v for k, v in grad_alignment(T, f, tfb, dss["train"], dev).items()})
            T.set_seed(seed)
            f0, tfb0 = T.build_model(d_emb=64, patch_len=A["patch_len"], num_features=7, pred_len=pred_len,
                                     use_revin=A["use_revin"], seq_len=seq_len, head_type=head, use_llm=use_llm,
                                     llm_policy=policy)
            f0.to(dev); tfb0.to(dev)
            out.update({f"{k}_init": v for k, v in grad_alignment(T, f0, tfb0, dss["train"], dev).items()})
    json.dump(out, open(os.path.join(run_dir, "result.json"), "w"), indent=1)
    p = os.path.join(run_dir, "best_model.pt")
    if os.path.exists(p):
        os.remove(p)                                   # no checkpoints kept (Output size)
    print("RESULT", json.dumps({k: v for k, v in out.items() if not isinstance(v, list)}), flush=True)


# ----------------------------------------------------------------------------------------------- self-test
def selftest(csv):
    """CPU checks of this script's own logic (component objective, probes, hour alignment, ridge)."""
    sys.path.insert(0, SRC)
    import numpy as np, torch
    import train as T
    orig = T.tfb_total_loss
    ok = True
    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)
    ds = T.ETTh1Dataset(csv, flag="train", seq_len=512, pred_len=96, patch_len=16)
    b = {k: torch.stack([ds[i][k] for i in range(8)]) for k in ("x", "x_patched", "y")}
    T.set_seed(0)
    f, tfb = T.build_model(d_emb=64, patch_len=16, num_features=7, pred_len=96, use_revin=True, seq_len=512,
                           head_type="flatten", use_llm=False, llm_policy="none")
    f.eval(); tfb.eval()
    td = [p for p in f.td_encoder.parameters()]; fd = [p for p in f.fd_encoder.parameters()]
    parts = {}
    for comp in ("full", "T", "F", "B"):
        T.tfb_total_loss = component_loss(comp) or orig
        T.set_seed(1)
        L = T.compute_losses(f, tfb, b["x_patched"], b["x"], b["y"], lam=LAM, loss_mode="fixed_weighted", alpha=0.1)
        extra = L["total_loss"] - L["task_loss"]
        parts[comp] = float(extra.detach())
        g = lambda ps: sum(float(x.abs().sum()) for x in torch.autograd.grad(extra, ps, retain_graph=True, allow_unused=True) if x is not None)
        gtd, gfd = g(td), g(fd)
        want = {"full": (True, True), "T": (True, False), "F": (False, True), "B": (True, True)}[comp]
        check(f"component {comp}: gradient reaches TD={want[0]} FD={want[1]}", (gtd > 0) == want[0] and (gfd > 0) == want[1],
              f"(|grad| TD {gtd:.3g}, FD {gfd:.3g})")
        if comp != "full":
            exp = {"T": LAM * float(L["l_t"]), "F": LAM * float(L["l_f"]), "B": (1 - LAM) * float(L["l_b"])}[comp]
            check(f"component {comp}: total = task + 0.1 x component", abs(float(extra) - 0.1 * exp) < 1e-5,
                  f"({float(extra):.6f} vs {0.1 * exp:.6f})")
    T.tfb_total_loss = orig
    check("components sum to the full objective", abs(parts["T"] + parts["F"] + parts["B"] - parts["full"]) < 1e-5,
          f"({parts['T'] + parts['F'] + parts['B']:.6f} vs {parts['full']:.6f})")
    _len = T.ETTh1Dataset.__len__; T.ETTh1Dataset.__len__ = lambda self: min(_len(self), 64)
    dsv = T.ETTh1Dataset(csv, flag="val", seq_len=512, pred_len=96, patch_len=16)
    R = shift_sensitivity(f, dsv, "cpu", bs=32)
    check("P1: z_F is circular-shift invariant (R_z_F < 1e-4)", R["R_z_F"] < 1e-4, f"(R_z_F {R['R_z_F']:.2e})")
    check("P1: z_T is shift sensitive at initialisation (R_z_T > 0.01)", R["R_z_T"] > 0.01, f"(R_z_T {R['R_z_T']:.3f})")
    xb = torch.stack([dsv[i]["x"] for i in range(4)])
    e1 = encode(f, xb)
    with torch.no_grad():
        _, aux = f(torch.stack([dsv[i]["x_patched"] for i in range(4)]), xb)
    check("P1: encode() reproduces the forward pass embeddings",
          float((e1["z_T"] - aux["z_T"]).abs().max()) < 1e-6 and float((e1["z_F"] - aux["z_F"]).abs().max()) < 1e-6)
    import pandas as pd
    raw = pd.read_csv(csv)
    for flag in ("train", "val", "test"):
        dsx = T.ETTh1Dataset(csv, flag=flag, seq_len=512, pred_len=96, patch_len=16)
        h = window_hours(csv, flag, 512, 96, len(dsx))
        nt, nv = 12 * 30 * 24, 4 * 30 * 24
        b1 = {"train": 0, "val": nt - 512, "test": nt + nv - 512}[flag]
        row = raw.iloc[b1 + 512 + 5]
        yv = (row[dsx.feature_cols].to_numpy(np.float32) - dsx.mean) / dsx.std
        check(f"hours aligned with windows ({flag})",
              np.allclose(yv, dsx[5]["y"][0].numpy(), atol=1e-5) and h[5] == pd.to_datetime(row["date"]).hour)
    G = grad_alignment(T, f, tfb, ds, "cpu", n_batches=2)
    want = {"cos_task_T_TD", "cos_task_B_TD", "cos_task_F_FD", "cos_task_B_FD"}
    check("P3: gradient cosines computed for every component/encoder pair that interacts",
          want <= set(G) and all(-1 - 1e-9 <= G[k] <= 1 + 1e-9 for k in G) and "cos_task_F_TD" not in G, f"({sorted(G)})")
    T.ETTh1Dataset.__len__ = _len
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 10)); W = rng.normal(size=(10, 2)); Y = X @ W
    check("ridge recovers a linear relation (R^2 > 0.99)", ridge_r2(X[:400], Y[:400], X[400:], Y[400:]) > 0.99)
    print("SELFTEST", "OK" if ok else "FAILED", flush=True)
    return ok


# ----------------------------------------------------------------------------------------------- driver
def src_ok():
    r = subprocess.run(["git", "-C", REPO, "diff", "--quiet", STAGE2_COMMIT, "HEAD", "--", "src"],
                       capture_output=True)
    if r.returncode == 0:
        return True, "src/ identical to the S1-S3 commit b32fe77"
    h = hashlib.sha256()
    fs = sorted(os.path.relpath(os.path.join(d, x), SRC) for d, _, F in os.walk(SRC) for x in F if x.endswith(".py"))
    for x in fs:
        h.update(x.encode() + b"\0" + open(os.path.join(SRC, x), "rb").read() + b"\0")
    want = json.load(open(PROT))["src_content_sha256"]
    return h.hexdigest() == want, f"src/ content hash {h.hexdigest()[:12]} (declared {want[:12]})"


def get_etth2(out):
    p = os.path.join(out, "data", "ETTh2.csv")
    found = sorted(glob.glob("/kaggle/input/**/ETTh2.csv", recursive=True) + glob.glob(os.path.join(REPO, "**", "ETTh2.csv"), recursive=True))
    if found:
        p = found[0]
    elif not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        urllib.request.urlretrieve(ETTH2_URL, p)
    sha = hashlib.sha256(open(p, "rb").read()).hexdigest()
    if sha != ETTH2_SHA:
        sys.exit(f"STOP: ETTh2.csv sha256 {sha} differs from the declared file")
    return p


def summarize(out):
    rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(out, "runs", "*", "result.json")))]
    if not rows:
        return
    keys = sorted({k for r in rows for k, v in r.items() if not isinstance(v, (list, dict))})
    with open(os.path.join(out, "results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    lines = [f"Stage R1 part {rows[0]['part']}: {len(rows)} runs", ""]
    for key in sorted({(r["setting"], r["arm"]) for r in rows}):
        rs = [r for r in rows if (r["setting"], r["arm"]) == key]
        m = lambda k: statistics.mean(r[k] for r in rs) if all(k in r for r in rs) else float("nan")
        s = lambda k: statistics.stdev(r[k] for r in rs) if len(rs) > 1 else 0.0
        extra = f"  R_zT {m('R_z_T'):.3f}  R_zF {m('R_z_F'):.1e}  phaseR2_zT {m('phaseR2_z_T'):.3f}" if "R_z_T" in rs[0] else ""
        lines.append(f"{key[0]:12s} {key[1]:10s} n={len(rs):2d}  best val MSE {m('best_val_mse'):.4f}±{s('best_val_mse'):.4f}  "
                     f"test MSE {m('test_mse'):.4f}±{s('test_mse'):.4f}  test RMSE {m('test_rmse'):.4f}{extra}")
    open(os.path.join(out, "summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["mechanism", "scope_f", "scope_e", "selftest"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--csv2", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--job", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--skip_preflight", action="store_true")
    args = ap.parse_args()
    args.csv = args.csv or sorted(glob.glob("/kaggle/input/**/ETTh1.csv", recursive=True)
                                  + glob.glob(os.path.join(REPO, "**", "ETTh1.csv"), recursive=True))[0]
    if args.part == "selftest":
        sys.exit(0 if selftest(args.csv) else 1)
    if args.job:
        return worker(args)
    assert args.out, "--out is required"
    C, Pr = json.load(open(CFG)), json.load(open(PROT))
    ok, msg = src_ok()
    print("code check:", msg, flush=True)
    if not ok:
        sys.exit("STOP: src/ differs from the code of S1-S3; R1 must use the same code.")
    os.makedirs(os.path.join(args.out, "runs"), exist_ok=True)
    args.csv2 = get_etth2(args.out) if args.part == "scope_f" else ""
    jobs = jobs_for(args.part, C, Pr)
    json.dump(dict(frozen=C, protocol=Pr, part=args.part, jobs=[job_id(j) for j in jobs], csv=args.csv, csv2=args.csv2,
                   git_commit=subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()),
              open(os.path.join(args.out, "protocol.json"), "w"), indent=1)
    if not args.skip_preflight:
        print("Preflight 1/2: self-test of this script (CPU) ...", flush=True)
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--part", "selftest", "--csv", args.csv],
                           capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
        open(os.path.join(args.out, "selftest_log.txt"), "w").write(r.stdout + r.stderr)
        print(r.stdout[-2500:], flush=True)
        if r.returncode != 0:
            sys.exit("Preflight FAILED (self-test): see selftest_log.txt; no experiment was run.")
        print("Preflight 2/2: Stage 1 tests on cuda ...", flush=True)
        r = subprocess.run([sys.executable, os.path.join(REPO, "tests", "test_stage1.py"), SRC, "--gpt2", "real",
                            "--device", "cuda", "--csv", args.csv], cwd=args.out,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus.split(",")[0]))
        if r.returncode != 0:
            sys.exit("Preflight FAILED: see stage1_test_log_real_cuda.txt; no experiment was run.")
    gpus = args.gpus.split(",")
    queues = {g: [] for g in gpus}
    for i, j in enumerate(jobs):
        queues[gpus[i % len(gpus)]].append(j)
    procs = []
    for g, q in queues.items():
        cmds = " ; ".join(f'"{sys.executable}" "{os.path.abspath(__file__)}" --part {args.part} --out "{args.out}" '
                          f'--csv "{args.csv}" --csv2 "{args.csv2}" --job {job_id(j)} > "{args.out}/runs/{job_id(j)}.log" 2>&1'
                          for j in q)
        procs.append(subprocess.Popen(cmds, shell=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=g, PYTHONUNBUFFERED="1")))
        print(f"GPU {g}: {len(q)} runs", flush=True)
    for p in procs:
        p.wait()
    summarize(args.out)


if __name__ == "__main__":
    main()
