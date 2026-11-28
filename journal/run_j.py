"""
Stage J (journal) experiment runner. Declared in configs/stage_j_protocol.json before any run.

  python journal/run_j.py --part comp  --out /kaggle/working/stage_j_comp   (competitor objectives and combiners)
  python journal/run_j.py --part sens  --out ...                            (sensitivity: alpha = 3, tau)
  python journal/run_j.py --part probe --out ...                            (representation probes + embeddings)
  python journal/run_j.py --part ecl   --out ...                            (Electricity, look-back 96, stride 4)
  python journal/run_j.py --part llm   --out ...                            (GPT-2 forecaster E)
  python journal/run_j.py --part cost  --out ... --gpus 0                   (3-epoch timing, one run at a time)
  python journal/run_j.py --part selftest

src/ (forecaster F, TFB, RAFT) is NOT modified; it must match the code declared for Stage X. Arms:
  none | tfb_a<x> | raft_a<x> | raft_rt_a<x> | raft_noproj_a<x>   -> exactly the Stage X code paths
  <aux>_<comb>_a<x>[_tau<t>]  aux in {raft, rnc, rnctf}, comb in {cf, pc, gs, sum}  -> journal/raft_j.py
  e_<arm>                       the same arm on the GPT-2 forecaster E (flatten head, LLM policy norms_pos)
"""
import argparse, glob, hashlib, importlib.util, json, os, subprocess, sys, tempfile, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")
JDIR = os.path.join(REPO, "journal")
CFG = os.path.join(REPO, "configs", "frozen_forecaster.json")
PROT = os.path.join(REPO, "configs", "stage_j_protocol.json")
LAM = 0.5

_spec = importlib.util.spec_from_file_location("sx", os.path.join(REPO, "scripts", "stage_x_raft.py"))
sx = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(sx)
sx.URLS["electricity.csv"] = [
    "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/electricity/electricity.csv",
    "https://huggingface.co/Melady/TEMPO/resolve/main/all_six_datasets/electricity/electricity.csv"]


# ----------------------------------------------------------------------------------------------- arms
AUX_NAMES = ("raft", "rnc", "rnctf")            # must equal raft_j.AUX (checked in test_j.py)
COMB_NAMES = ("cf", "pc", "gs", "sum")          # must equal raft_j.COMBINERS (checked in test_j.py)
def arm_parse(arm):
    """-> dict(llm, kind, variant, aux, comb, alpha, tau). kind: none | tfb | raft (Stage X path) | j (Stage J path)."""
    llm = arm.startswith("e_")
    a = arm[2:] if llm else arm
    d = dict(llm=llm, kind=None, variant=None, aux=None, comb=None, alpha=0.0, tau=0.1)
    if a == "none":
        d["kind"] = "none"; return d
    if a.startswith("tfb_a"):
        d.update(kind="tfb", alpha=float(a[5:])); return d
    tau = 0.1
    if "_tau" in a:
        a, t = a.split("_tau"); tau = float(t)
    head, al = a.rsplit("_a", 1)
    al = float(al)
    if head in ("raft", "raft_rt", "raft_noproj") and tau == 0.1:
        d.update(kind="raft", variant=head, alpha=al); return d
    aux, comb = head.split("_")
    assert aux in AUX_NAMES and comb in COMB_NAMES, arm
    d.update(kind="j", aux=aux, comb=comb, alpha=al, tau=tau)
    return d


def jobs_for(part, P):
    pp = P["parts"][part]
    J = [dict(setting=st, arm=a, seed=s) for s in pp["seeds"] for st in pp["settings"] for a in pp["arms"]]
    for j in J:
        arm_parse(j["arm"])
    return J


def job_id(j):
    return f"{j['setting']}__{j['arm']}__s{j['seed']}"


# ----------------------------------------------------------------------------------------------- worker
def worker(args):
    for d in ("", "data", "models", "losses"):
        sys.path.insert(0, os.path.join(SRC, d))
    sys.path.insert(0, JDIR)
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    if os.environ.get("STAGE_GPT2_MOCK") == "1":                 # CPU self-test only: tiny random GPT-2
        from transformers import GPT2Config, GPT2Model
        GPT2Model.from_pretrained = classmethod(lambda cls, name, **k: GPT2Model(GPT2Config(n_layer=1, n_head=2)))
    import train as T
    import raft as R
    import raft_j as RJ
    from datasets import BenchmarkDataset, SPECS
    from metrics import evaluate_full
    j = sx.parse_job(args.job)
    C = json.load(open(CFG)); A, Rt = C["architecture"], C["training"]
    P = json.load(open(PROT))
    dataset, pred_len, pct = sx.setting_parse(j["setting"])
    dcfg = P["datasets"][dataset]
    seq_len, stride = dcfg["lookback"], dcfg.get("train_stride", 1)
    V = SPECS[dataset][2]
    a = arm_parse(j["arm"])
    csv_path = sx.csv_for(dataset, args.data_dir)
    DRY = os.environ.get("STAGE_DRY") == "1"
    dev = "cpu" if DRY else "cuda"
    epochs, bs, min_ep = Rt["max_epochs"], dcfg.get("batch_size", Rt["batch_size"]), Rt["min_epochs"]
    eval_bs = dcfg.get("batch_size", 64)                         # many-variable datasets: smaller batches (GPU memory)
    if "max_epochs" in P["parts"][args.part]:                    # part 'cost': a few epochs only (timing, not accuracy)
        epochs = min_ep = P["parts"][args.part]["max_epochs"]
    if DRY:
        epochs, bs, min_ep = 2, 4, 1

    def make_ds(csv, flag, seq_len, pred_len, patch_len):
        ds = BenchmarkDataset(csv, flag=flag, seq_len=seq_len, pred_len=pred_len, patch_len=patch_len,
                              percent=pct if flag == "train" else 100, name=dataset)
        idx = None
        if flag == "train" and stride > 1:
            idx = list(range(0, len(ds), stride))
        if DRY:
            idx = (idx or list(range(len(ds))))[:8]
        if idx is not None:
            sub = torch.utils.data.Subset(ds, idx)
            sub.mean, sub.std = ds.mean, ds.std
            return sub
        return ds
    T.ETTh1Dataset = make_ds
    stats = {}
    if a["kind"] == "raft":
        T.build_model = RJ.build_raft_model_any
        T.compute_losses = R.make_compute_losses(a["variant"], a["alpha"], stats)
    elif a["kind"] == "j":
        T.build_model = RJ.build_raft_model_any
        T.compute_losses = RJ.make_compute_losses_j(a["aux"], a["comb"], a["alpha"], stats, tau=a["tau"])
    if a["llm"] and a["kind"] in ("raft", "j"):                  # amendment A1: release each step's graph (GPT-2 memory)
        T.compute_losses = RJ.release_graph(T.compute_losses)
    use_llm = a["llm"]
    E = C["confirmation_model"]
    run_dir = os.path.join(args.out, "runs", job_id(j))
    if not DRY:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    loss_alpha = a["alpha"]                                   # 0 for "none"; RAFT/J losses carry their own alpha
    f, tfb, r = T.train(csv_path=csv_path, checkpoint_dir=run_dir, seq_len=seq_len, pred_len=pred_len,
                        num_features=V, batch_size=bs, learning_rate=Rt["learning_rate"],
                        weight_decay=Rt["weight_decay"], epochs=epochs, patience=Rt["early_stopping_patience"],
                        min_epochs=min_ep, loss_mode="fixed_weighted", alpha=loss_alpha, lam=LAM, use_revin=A["use_revin"],
                        seed=j["seed"], head_type=A["head_type"], use_llm=use_llm,
                        llm_policy=E["llm_policy"] if use_llm else A["llm_policy"],
                        run_sanity_check=False, device=dev)
    minutes = (time.time() - t0) / 60
    out = dict(part=args.part, setting=j["setting"], dataset=dataset, pred_len=pred_len, seq_len=seq_len,
               train_stride=stride, batch_size=bs, percent=pct, arm=j["arm"], llm=use_llm, kind=a["kind"], variant=a["variant"],
               aux=a["aux"], comb=a["comb"], alpha=a["alpha"], tau=a["tau"], seed=j["seed"],
               best_epoch=r["best_epoch"], epochs_run=r["epochs_run"], best_val_mse=r["best_val_mse"],
               minutes=round(minutes, 2), sec_per_epoch=round(60 * minutes / max(r["epochs_run"], 1), 1),
               peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if not DRY else 0.0,
               n_trainable=r["n_trainable"], csv=os.path.basename(csv_path))
    if stats.get("steps"):
        out.update(proj_steps=stats["steps"], proj_conflict_rate=stats["conflicts"] / stats["steps"],
                   proj_mean_cos=stats["cos_sum"] / stats["steps"])
    f.eval()
    for split in ("val", "test"):
        ds = make_ds(csv_path, split, seq_len, pred_len, A["patch_len"])
        out.update({f"{split}_{k}": v for k, v in evaluate_full(f, DataLoader(ds, batch_size=eval_bs, shuffle=False), dev).items()})
    ds = make_ds(csv_path, "test", seq_len, pred_len, A["patch_len"])
    se_t = se_v = None
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=eval_bs, shuffle=False):
            yh, _ = f(b["x_patched"].to(dev), b["x"].to(dev))
            e = (yh.double().cpu() - b["y"].double()) ** 2
            se_t = e.sum((0, 2)) if se_t is None else se_t + e.sum((0, 2))
            se_v = e.sum((0, 1)) if se_v is None else se_v + e.sum((0, 1))
    n = len(ds)
    out["test_mse_per_step"] = [round(float(v) / (n * V), 6) for v in se_t]
    out["test_mse_per_variable"] = [round(float(v) / (n * pred_len), 6) for v in se_v]
    if args.part == "cost":
        loader = DataLoader(ds, batch_size=eval_bs, shuffle=False)
        with torch.no_grad():
            for b in loader:                                     # warm-up
                f(b["x_patched"].to(dev), b["x"].to(dev)); break
            if dev == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
            for b in loader:
                f(b["x_patched"].to(dev), b["x"].to(dev))
            if dev == "cuda":
                torch.cuda.synchronize()
        out["test_inference_sec"] = round(time.time() - t1, 3)
        out["test_windows"] = len(ds)
    if args.part == "probe":
        out.update(sx.representation_probe(f, ds, dev))
        if j["seed"] == P["parts"]["probe"]["embedding_seed"]:
            np.savez_compressed(os.path.join(run_dir, "embeddings.npz"), **dump_embeddings(f, ds, dev))
    json.dump(out, open(os.path.join(run_dir, "result.json"), "w"), indent=1)
    p = os.path.join(run_dir, "best_model.pt")
    if os.path.exists(p):
        os.remove(p)
    print("RESULT", json.dumps({k: v for k, v in out.items() if not isinstance(v, list)}), flush=True)


def dump_embeddings(f, ds, dev, n_windows=64):
    """TD (h) and FD (u) representations and normalised futures of n_windows evenly spaced TEST windows (float16)."""
    import numpy as np
    import torch
    idx = np.linspace(0, len(ds) - 1, n_windows).round().astype(int).tolist()
    b = [ds[i] for i in idx]
    x = torch.stack([e["x"] for e in b]).to(dev); xp = torch.stack([e["x_patched"] for e in b]).to(dev)
    y = torch.stack([e["y"] for e in b]).to(dev)
    with torch.no_grad():
        _, aux = f(xp, x)
        td = aux["td_patch_embeddings"]
        B, P_, V, d = td.shape
        h = td.permute(0, 2, 1, 3).reshape(B * V, -1)
        mean, std = f.revin.stats(x)
        u = f.fd_encoder.pooled_per_variable(f.revin.norm(x, mean, std)).reshape(B * V, -1)
        yr = ((y - mean) / std).permute(0, 2, 1).reshape(B * V, -1)
    var = np.tile(np.arange(V), B); win = np.repeat(np.array(idx), V)
    return dict(h=h.cpu().numpy().astype(np.float16), u=u.cpu().numpy().astype(np.float16),
                y=yr.cpu().numpy().astype(np.float16), variable=var, window=win)


# ----------------------------------------------------------------------------------------------- checks
def code_hash(root):
    h = hashlib.sha256()
    fs = sorted(os.path.relpath(os.path.join(d, x), root) for d, _, F in os.walk(root) for x in F if x.endswith(".py"))
    for x in fs:
        h.update(x.encode() + b"\0" + open(os.path.join(root, x), "rb").read() + b"\0")
    return h.hexdigest()


def selftest(csv):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, os.path.join(JDIR, "test_j.py"), SRC, JDIR], capture_output=True, text=True, env=env)
    print(r.stdout[-5000:], r.stderr[-3000:])
    if r.returncode != 0:
        return False
    import shutil
    import numpy as np, pandas as pd
    tmp = tempfile.mkdtemp()
    data_dir = os.path.join(tmp, "data"); os.makedirs(data_dir)
    shutil.copy(csv, os.path.join(data_dir, "ETTh1.csv"))
    rng = np.random.default_rng(0)                    # synthetic stand-in with the shape of electricity.csv
    t = np.arange(1200)
    vals = np.sin(t[:, None] / 24.0 * 2 * np.pi + rng.uniform(0, 6, 321)) + 0.1 * rng.standard_normal((1200, 321))
    df = pd.DataFrame(vals, columns=[f"{k}" for k in range(320)] + ["OT"]); df.insert(0, "date", pd.date_range("2016-07-01", periods=1200, freq="h"))
    df.to_csv(os.path.join(data_dir, "electricity.csv"), index=False)
    ok = True
    cases = [("probe", "ETTh1_H96__none__s42"), ("comp", "ETTh1_H96__rnc_cf_a0.3__s1"), ("comp", "ETTh1_H96__rnctf_cf_a1__s1"),
             ("comp", "ETTh1_H96__raft_pc_a1__s1"), ("comp", "ETTh1_H96__raft_gs_a1__s1"), ("sens", "ETTh1_H96__raft_a3__s1"),
             ("sens", "ETTh1_H96__raft_cf_a1_tau0.05__s1"), ("ecl", "ECL_H96__tfb_a0.01__s1"), ("ecl", "ECL_H96__raft_a1__s1"),
             ("llm", "ETTh1_H96__e_tfb_a0.01__s1"), ("llm", "ETTh1_H96__e_raft_a1__s1"), ("probe", "ETTh1_H96__raft_a1__s42"),
             ("cost", "ETTh1_H96__raft_a1__s42")]
    for part, jid in cases:
        os.makedirs(os.path.join(tmp, "runs", jid), exist_ok=True)
        q = subprocess.run([sys.executable, os.path.abspath(__file__), "--part", part, "--out", tmp, "--data_dir", data_dir,
                            "--job", jid], capture_output=True, text=True,
                           env=dict(env, STAGE_DRY="1", STAGE_GPT2_MOCK="1"))
        res = os.path.join(tmp, "runs", jid, "result.json")
        good = q.returncode == 0 and os.path.exists(res)
        if good:
            rr = json.load(open(res))
            H = int(jid.split("__")[0].split("_H")[1])
            good = rr["test_rmse"] == rr["test_rmse"] and len(rr["test_mse_per_step"]) == H
            if "raft" in jid or "rnc" in jid:
                good = good and rr.get("proj_steps", 0) > 0
            if jid.startswith("ECL"):
                good = good and rr["seq_len"] == 96 and rr["train_stride"] == 4 and len(rr["test_mse_per_variable"]) == 321
                good = good and rr["batch_size"] == 4                         # DRY batch (the real ECL batch is 8)
            good = good and rr["llm"] is jid.split("__")[1].startswith("e_")
            if part == "cost":
                good = good and rr.get("test_inference_sec", -1) >= 0 and rr["epochs_run"] <= 3
            if part == "probe":
                good = good and "probe_D_h" in rr and os.path.exists(os.path.join(tmp, "runs", jid, "embeddings.npz"))
        print(f"[{'PASS' if good else 'FAIL'}] runner plumbing (CPU, tiny): {part} {jid}", flush=True)
        if not good:
            print(q.stdout[-2000:], q.stderr[-2000:])
        ok &= good
    return ok


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["comp", "sens", "probe", "ecl", "llm", "cost", "selftest"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--per_gpu", type=int, default=None)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--job", default=None)
    ap.add_argument("--skip_preflight", action="store_true")
    args = ap.parse_args()
    if args.job:
        return worker(args)
    if args.part == "selftest":
        csv = sx.find_or_get("ETTh1.csv", args.out or tempfile.mkdtemp())
        sys.exit(0 if selftest(csv) else 1)
    assert args.out, "--out is required"
    P = json.load(open(PROT))
    hs, hj = code_hash(SRC), code_hash(JDIR)
    print(f"code check: src {hs[:12]} (declared {P['src_content_sha256'][:12]}), journal {hj[:12]} "
          f"(declared {P['journal_content_sha256'][:12]})", flush=True)
    if hs != P["src_content_sha256"] or hj != P["journal_content_sha256"]:
        sys.exit("STOP: code differs from the code declared for Stage J.")
    os.makedirs(os.path.join(args.out, "runs"), exist_ok=True)
    J = jobs_for(args.part, P)
    data_dir, data_rec = sx.prepare_data(J, args.out)
    done = {os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(args.out, "runs", "*", "result.json"))}
    todo = [j for j in J if job_id(j) not in done]
    json.dump(dict(protocol=P, part=args.part, jobs=[job_id(j) for j in J], data=data_rec, src_sha256=hs, journal_sha256=hj),
              open(os.path.join(args.out, "protocol.json"), "w"), indent=1)
    print(f"{len(J)} jobs, {len(done & {job_id(j) for j in J})} already done, {len(todo)} to run", flush=True)
    if not args.skip_preflight:
        print("Preflight: Stage J self-test (CPU) ...", flush=True)
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--part", "selftest", "--out", os.path.join(args.out, "selftest")],
                           capture_output=True, text=True)
        open(os.path.join(args.out, "selftest_log.txt"), "w").write(r.stdout + r.stderr)
        print(r.stdout[-3000:], flush=True)
        if r.returncode != 0:
            sys.exit("Preflight FAILED (self-test): see selftest_log.txt; no experiment was run.")
    per_gpu = args.per_gpu or P["parts"][args.part]["per_gpu"]
    lanes = [(g, k) for k in range(per_gpu) for g in args.gpus.split(",")]
    queues = {ln: [] for ln in lanes}
    for i, j in enumerate(todo):
        queues[lanes[i % len(lanes)]].append(j)
    procs = []
    for (g, k), q in queues.items():
        if not q:
            continue
        cmds = " ; ".join(f'"{sys.executable}" "{os.path.abspath(__file__)}" --part {args.part} --out "{args.out}" '
                          f'--data_dir "{data_dir}" --job {job_id(j)} > "{args.out}/runs/{job_id(j)}.log" 2>&1' for j in q)
        procs.append(subprocess.Popen(cmds, shell=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=g, PYTHONUNBUFFERED="1")))
        print(f"GPU {g} lane {k}: {len(q)} runs", flush=True)
    t0 = time.time()
    while any(p.poll() is None for p in procs):
        time.sleep(300)
        n = len(glob.glob(os.path.join(args.out, "runs", "*", "result.json")))
        print(f"[{(time.time() - t0) / 60:.0f} min] {n} result files", flush=True)
    sx.summarize(args.out)


if __name__ == "__main__":
    main()
