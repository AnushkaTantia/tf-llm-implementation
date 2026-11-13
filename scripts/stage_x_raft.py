"""
Stage X (RAFT) experiment runner. Declared in configs/stage_x_protocol.json before any run.

  python scripts/stage_x_raft.py --part pilot    --out /kaggle/working/stage_x_pilot
  python scripts/stage_x_raft.py --part core     --raft_arm raft_a0.3 --out ...
  python scripts/stage_x_raft.py --part ablation --raft_arm raft_a0.3 --out ...
  python scripts/stage_x_raft.py --part breadth  --raft_arm raft_a0.3 --group a --out ...
  python scripts/stage_x_raft.py --part fewshot  --raft_arm raft_a0.3 --out ...
  python scripts/stage_x_raft.py --part selftest
  python scripts/stage_x_raft.py --part timing   --out ...      (one short run per dataset: seconds per epoch)

Runs are spread over the GPUs (--gpus 0,1) with --per_gpu concurrent queues per GPU; jobs are ordered seed by seed
so an interrupted session still leaves complete paired sets. Each run writes runs/<job>/result.json; checkpoints
are deleted after evaluation. A finished run is skipped when the part is restarted with the same --out.
"""
import argparse, csv, glob, hashlib, json, os, statistics, subprocess, sys, time, urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")
CFG = os.path.join(REPO, "configs", "frozen_forecaster.json")
PROT = os.path.join(REPO, "configs", "stage_x_protocol.json")
LAM = 0.5

URLS = {
    "ETTh1.csv": ["https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"],
    "ETTh2.csv": ["https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv"],
    "ETTm1.csv": ["https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv"],
    "ETTm2.csv": ["https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv"],
    "exchange_rate.csv": ["https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/exchange_rate/exchange_rate.csv",
                          "https://huggingface.co/Melady/TEMPO/resolve/main/all_six_datasets/exchange_rate/exchange_rate.csv"],
    "weather.csv": ["https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/weather/weather.csv",
                    "https://huggingface.co/Melady/TEMPO/resolve/main/all_six_datasets/weather/weather.csv"],
    "national_illness.csv": ["https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/illness/national_illness.csv",
                             "https://huggingface.co/Melady/TEMPO/resolve/main/all_six_datasets/illness/national_illness.csv"],
}
KNOWN_SHA = {   # files already used in S1-R1
    "ETTh1.csv": "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066",
    "ETTh2.csv": "a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b",
}


# ----------------------------------------------------------------------------------------------- jobs
def lookback(P, dataset):
    return P["datasets"][dataset]["lookback"]


def setting_parse(s):
    """'ETTm1_H96' or 'ETTh1_H96_p10' -> (dataset, horizon, percent)."""
    parts = s.split("_")
    pct = 100
    if parts[-1].startswith("p"):
        pct = int(parts[-1][1:]); parts = parts[:-1]
    return parts[0], int(parts[1][1:]), pct


def jobs_for(args, P):
    pp = P["parts"]
    J = []
    def add(settings, arms, seeds):
        for s in seeds:
            for st in settings:
                for a in arms:
                    J.append(dict(setting=st, arm=a, seed=s))
    if args.part == "pilot":
        add(pp["pilot"]["settings"], pp["pilot"]["arms"], pp["pilot"]["seeds"])
    elif args.part == "core":
        for s in pp["core"]["seeds"]:
            for st in pp["core"]["settings"]:
                J.append(dict(setting=st, arm=args.raft_arm, seed=s))
            J.append(dict(setting="ETTh1_H336", arm="tfb_a0.01", seed=s))
    elif args.part == "ablation":
        a = args.raft_arm.split("_a")[-1]
        arms = [f"raft_rt_a{a}", f"raft_tf_a{a}", f"raft_noproj_a{a}", "fdfix", args.raft_arm, "none"]
        add(pp["ablation"]["settings"], arms, pp["ablation"]["seeds"])
    elif args.part == "breadth":
        groups = args.group.split(",")
        settings = [st for g in groups for st in pp["breadth"]["groups"][g]]
        add(settings, ["none", "tfb_a0.01", args.raft_arm], pp["breadth"]["seeds"])
    elif args.part == "fewshot":
        pc = pp["fewshot"]["percent"]
        add([f"{s}_p{pc}" for s in pp["fewshot"]["settings"]], ["none", "tfb_a0.01", args.raft_arm], pp["fewshot"]["seeds"])
    elif args.part == "timing":
        for st in ["ETTh1_H96", "ETTm1_H96", "Exchange_H96", "ILI_H24", "Weather_H96"]:
            for a in ["none", "tfb_a0.01", args.raft_arm or "raft_a0.3"]:
                J.append(dict(setting=st, arm=a, seed=42))
    if args.settings:
        keep = set(args.settings.split(","))
        J = [j for j in J if j["setting"] in keep]
    if getattr(args, "arms", None):
        keep = set(a.strip().replace("-", "_") for a in args.arms.split(","))
        J = [j for j in J if j["arm"] in keep]
    for j in J:
        arm_parse(j["arm"])                       # every arm must parse before anything runs
    return J


def job_id(j):
    return f"{j['setting']}__{j['arm']}__s{j['seed']}"


def parse_job(s):
    st, arm, seed = s.split("__")
    return dict(setting=st, arm=arm, seed=int(seed[1:]))


def arm_parse(arm):
    """-> (kind, variant, alpha): kind in {none, tfb, raft}."""
    if arm == "none":
        return "none", None, 0.0
    if arm == "fdfix":
        return "raft", "fdfix", 0.0
    if arm.startswith("tfb_a"):
        return "tfb", None, float(arm[5:])
    variant, a = arm.rsplit("_a", 1)
    from_variants = {"raft", "raft_rt", "raft_tf", "raft_noproj"}
    assert variant in from_variants, arm
    return "raft", variant, float(a)


# ----------------------------------------------------------------------------------------------- data
def find_or_get(fn, out):
    hits = sorted(glob.glob(f"/kaggle/input/**/{fn}", recursive=True) + glob.glob(os.path.join(REPO, "data", fn)))
    if hits:
        return hits[0]
    p = os.path.join(out, "data", fn)
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        err = None
        for u in URLS[fn]:
            try:
                urllib.request.urlretrieve(u, p); err = None; break
            except Exception as e:
                err = e
        if err is not None:
            sys.exit(f"STOP: could not obtain {fn} ({err}). Attach it as a Kaggle input dataset.")
    return p


def csv_for(dataset, data_dir):
    sys.path.insert(0, os.path.join(SRC, "data"))
    from datasets import SPECS
    return os.path.join(data_dir, SPECS[dataset][0])


def prepare_data(J, out):
    sys.path.insert(0, os.path.join(SRC, "data"))
    from datasets import SPECS
    import pandas as pd
    data_dir = os.path.join(out, "data"); os.makedirs(data_dir, exist_ok=True)
    rec = {}
    for ds in sorted({setting_parse(j["setting"])[0] for j in J}):
        fn = SPECS[ds][0]
        p = find_or_get(fn, out)
        dst = os.path.join(data_dir, fn)
        if os.path.abspath(p) != os.path.abspath(dst):
            import shutil; shutil.copy(p, dst)
        sha = hashlib.sha256(open(dst, "rb").read()).hexdigest()
        if fn in KNOWN_SHA and sha != KNOWN_SHA[fn]:
            sys.exit(f"STOP: {fn} sha256 {sha} differs from the file used in S1-R1")
        df = pd.read_csv(dst, nrows=3)
        nvar = len([c for c in df.columns if c != "date"])
        if nvar != SPECS[ds][2]:
            sys.exit(f"STOP: {fn} has {nvar} variables, expected {SPECS[ds][2]}")
        rec[fn] = dict(sha256=sha, rows=sum(1 for _ in open(dst)) - 1, source=p)
        print(f"data {fn}: {rec[fn]['rows']} rows, sha256 {sha[:12]}", flush=True)
    return data_dir, rec


# ----------------------------------------------------------------------------------------------- worker
def worker(args):
    for d in ("", "data", "models", "losses"):
        sys.path.insert(0, os.path.join(SRC, d))
    import torch
    from torch.utils.data import DataLoader
    import train as T
    import raft as R
    from datasets import BenchmarkDataset, SPECS
    from metrics import evaluate_full
    j = parse_job(args.job)
    C = json.load(open(CFG)); A, Rt = C["architecture"], C["training"]
    P = json.load(open(PROT))
    dataset, pred_len, pct = setting_parse(j["setting"])
    seq_len = lookback(P, dataset)
    V = SPECS[dataset][2]
    kind, variant, alpha = arm_parse(j["arm"])
    csv_path = csv_for(dataset, args.data_dir)
    DRY = os.environ.get("STAGE_DRY") == "1"
    dev = "cpu" if DRY else "cuda"
    epochs, bs, min_ep = Rt["max_epochs"], Rt["batch_size"], Rt["min_epochs"]
    if DRY:
        epochs, bs, min_ep = 2, 4, 1

    def make_ds(csv, flag, seq_len, pred_len, patch_len):
        ds = BenchmarkDataset(csv, flag=flag, seq_len=seq_len, pred_len=pred_len, patch_len=patch_len,
                              percent=pct if flag == "train" else 100, name=dataset)
        if DRY:
            ds = torch.utils.data.Subset(ds, range(min(len(ds), 8)))
            ds.mean, ds.std = ds.dataset.mean, ds.dataset.std
        return ds
    T.ETTh1Dataset = make_ds                                  # this run process only
    stats = {}
    if kind == "raft":
        T.build_model = R.build_raft_model
        T.compute_losses = R.make_compute_losses(variant, alpha, stats)
    run_dir = os.path.join(args.out, "runs", job_id(j))
    if not DRY:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    f, tfb, r = T.train(csv_path=csv_path, checkpoint_dir=run_dir, seq_len=seq_len, pred_len=pred_len,
                        num_features=V, batch_size=bs, learning_rate=Rt["learning_rate"],
                        weight_decay=Rt["weight_decay"], epochs=epochs, patience=Rt["early_stopping_patience"],
                        min_epochs=min_ep, loss_mode="fixed_weighted", alpha=alpha, lam=LAM, use_revin=A["use_revin"],
                        seed=j["seed"], head_type=A["head_type"], use_llm=A["use_llm"], llm_policy=A["llm_policy"],
                        run_sanity_check=False, device=dev)
    minutes = (time.time() - t0) / 60
    out = dict(part=args.part, setting=j["setting"], dataset=dataset, pred_len=pred_len, seq_len=seq_len,
               percent=pct, arm=j["arm"], kind=kind, variant=variant, alpha=alpha, seed=j["seed"],
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
        out.update({f"{split}_{k}": v for k, v in evaluate_full(f, DataLoader(ds, batch_size=64, shuffle=False), dev).items()})
    ds = make_ds(csv_path, "test", seq_len, pred_len, A["patch_len"])
    se_t = se_v = None
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=64, shuffle=False):
            yh, _ = f(b["x_patched"].to(dev), b["x"].to(dev))
            e = (yh.double().cpu() - b["y"].double()) ** 2
            se_t = e.sum((0, 2)) if se_t is None else se_t + e.sum((0, 2))
            se_v = e.sum((0, 1)) if se_v is None else se_v + e.sum((0, 1))
    n = len(ds)
    out["test_mse_per_step"] = [round(float(v) / (n * V), 6) for v in se_t]
    out["test_mse_per_variable"] = [round(float(v) / (n * pred_len), 6) for v in se_v]
    json.dump(out, open(os.path.join(run_dir, "result.json"), "w"), indent=1)
    p = os.path.join(run_dir, "best_model.pt")
    if os.path.exists(p):
        os.remove(p)
    print("RESULT", json.dumps({k: v for k, v in out.items() if not isinstance(v, list)}), flush=True)


# ----------------------------------------------------------------------------------------------- checks
def src_hash():
    h = hashlib.sha256()
    fs = sorted(os.path.relpath(os.path.join(d, x), SRC) for d, _, F in os.walk(SRC) for x in F if x.endswith(".py"))
    for x in fs:
        h.update(x.encode() + b"\0" + open(os.path.join(SRC, x), "rb").read() + b"\0")
    return h.hexdigest()


def selftest(csv):
    r = subprocess.run([sys.executable, os.path.join(REPO, "tests", "test_raft.py"), SRC, "--csv", csv],
                       capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    print(r.stdout[-4000:], r.stderr[-2000:])
    if r.returncode != 0:
        return False
    # runner plumbing on CPU: one tiny run per arm kind
    import tempfile
    tmp = tempfile.mkdtemp()
    data_dir = os.path.join(tmp, "data"); os.makedirs(data_dir)
    import shutil; shutil.copy(csv, os.path.join(data_dir, "ETTh1.csv"))
    ok = True
    for arm in ("none", "tfb_a0.01", "raft_a0.3", "raft_noproj_a0.3", "fdfix"):
        jid = f"ETTh1_H96__{arm}__s1"
        os.makedirs(os.path.join(tmp, "runs", jid), exist_ok=True)
        q = subprocess.run([sys.executable, os.path.abspath(__file__), "--part", "selftest_job", "--out", tmp,
                            "--data_dir", data_dir, "--job", jid], capture_output=True, text=True,
                           env=dict(os.environ, STAGE_DRY="1", CUDA_VISIBLE_DEVICES=""))
        res = os.path.join(tmp, "runs", jid, "result.json")
        good = q.returncode == 0 and os.path.exists(res)
        if good:
            rr = json.load(open(res))
            good = rr["test_rmse"] == rr["test_rmse"] and len(rr["test_mse_per_step"]) == 96
            if arm == "raft_a0.3":
                good = good and rr.get("proj_steps", 0) > 0
        print(f"[{'PASS' if good else 'FAIL'}] runner plumbing (CPU, tiny): {arm}", flush=True)
        if not good:
            print(q.stdout[-1500:], q.stderr[-1500:])
        ok &= good
    return ok


# ----------------------------------------------------------------------------------------------- summary
def summarize(out):
    rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(out, "runs", "*", "result.json")))]
    if not rows:
        return
    keys = sorted({k for r in rows for k, v in r.items() if not isinstance(v, (list, dict))})
    with open(os.path.join(out, "results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    lines = [f"Stage X part {rows[0]['part']}: {len(rows)} runs", ""]
    for key in sorted({(r["setting"], r["arm"]) for r in rows}):
        rs = [r for r in rows if (r["setting"], r["arm"]) == key]
        m = lambda k: statistics.mean(r[k] for r in rs)
        s = lambda k: statistics.stdev(r[k] for r in rs) if len(rs) > 1 else 0.0
        pj = f"  conflict {m('proj_conflict_rate'):.2f}" if "proj_conflict_rate" in rs[0] else ""
        lines.append(f"{key[0]:14s} {key[1]:18s} n={len(rs):2d}  val {m('best_val_mse'):.4f}±{s('best_val_mse'):.4f}  "
                     f"test MSE {m('test_mse'):.4f}±{s('test_mse'):.4f}  RMSE {m('test_rmse'):.4f}  SMAPE {m('test_smape'):.2f}  "
                     f"{m('sec_per_epoch'):.0f}s/ep{pj}")
    open(os.path.join(out, "summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True,
                    choices=["pilot", "core", "ablation", "breadth", "fewshot", "timing", "selftest", "selftest_job"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--raft_arm", default=None, help="selected RAFT arm from the pilot, e.g. raft_a0.3")
    ap.add_argument("--group", default="a", help="breadth groups, e.g. a or a,b")
    ap.add_argument("--settings", default=None, help="optional comma list to restrict settings")
    ap.add_argument("--arms", default=None, help="optional comma list to restrict arms (e.g. only the missing RAFT arms)")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--per_gpu", type=int, default=2)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--job", default=None)
    ap.add_argument("--skip_preflight", action="store_true")
    args = ap.parse_args()
    if args.part == "selftest_job":
        args.part = "selftest"
        return worker(args)
    if args.job:
        return worker(args)
    if args.part == "selftest":
        csv = find_or_get("ETTh1.csv", args.out or "/tmp/stage_x_selftest")
        sys.exit(0 if selftest(csv) else 1)
    assert args.out, "--out is required"
    if args.raft_arm:
        args.raft_arm = args.raft_arm.strip().replace("-", "_")
        try:
            k, v, al = arm_parse(args.raft_arm)
            assert k == "raft" and v == "raft" and al > 0
        except Exception:
            sys.exit(f"STOP: --raft_arm {args.raft_arm!r} is not a valid RAFT arm (expected e.g. raft_a1)")
    if args.part in ("core", "ablation", "breadth", "fewshot"):
        assert args.raft_arm, "--raft_arm (selected in the pilot) is required"
    P = json.load(open(PROT))
    h = src_hash()
    print(f"code check: src/ content sha256 {h[:12]} (declared {P['src_content_sha256'][:12]})", flush=True)
    if h != P["src_content_sha256"]:
        sys.exit("STOP: src/ differs from the code declared for Stage X.")
    os.makedirs(os.path.join(args.out, "runs"), exist_ok=True)
    J = jobs_for(args, P)
    data_dir, data_rec = prepare_data(J, args.out)
    done = {os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(args.out, "runs", "*", "result.json"))}
    todo = [j for j in J if job_id(j) not in done]
    git = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    json.dump(dict(protocol=P, part=args.part, raft_arm=args.raft_arm, group=args.group, jobs=[job_id(j) for j in J],
                   data=data_rec, git_commit=git, src_sha256=h), open(os.path.join(args.out, "protocol.json"), "w"), indent=1)
    print(f"{len(J)} jobs, {len(done & {job_id(j) for j in J})} already done, {len(todo)} to run", flush=True)
    if not args.skip_preflight:
        print("Preflight 1/2: RAFT checks + runner plumbing (CPU) ...", flush=True)
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--part", "selftest",
                            "--out", os.path.join(args.out, "selftest")], capture_output=True, text=True)
        open(os.path.join(args.out, "selftest_log.txt"), "w").write(r.stdout + r.stderr)
        print(r.stdout[-3000:], flush=True)
        if r.returncode != 0:
            sys.exit("Preflight FAILED (self-test): see selftest_log.txt; no experiment was run.")
        print("Preflight 2/2: Stage 1 tests on cuda ...", flush=True)
        r = subprocess.run([sys.executable, os.path.join(REPO, "tests", "test_stage1.py"), SRC, "--gpt2", "real",
                            "--device", "cuda", "--csv", os.path.join(data_dir, "ETTh1.csv") if os.path.exists(os.path.join(data_dir, "ETTh1.csv")) else find_or_get("ETTh1.csv", args.out)],
                           cwd=args.out, env=dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus.split(",")[0]))
        if r.returncode != 0:
            sys.exit("Preflight FAILED: Stage 1 tests on cuda; no experiment was run.")
    lanes = [(g, k) for k in range(args.per_gpu) for g in args.gpus.split(",")]
    queues = {ln: [] for ln in lanes}
    for i, j in enumerate(todo):
        queues[lanes[i % len(lanes)]].append(j)
    procs = []
    for (g, k), q in queues.items():
        if not q:
            continue
        cmds = " ; ".join(f'"{sys.executable}" "{os.path.abspath(__file__)}" --part {args.part} --out "{args.out}" '
                          f'--data_dir "{data_dir}" --job {job_id(j)} > "{args.out}/runs/{job_id(j)}.log" 2>&1'
                          for j in q)
        procs.append(subprocess.Popen(cmds, shell=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=g, PYTHONUNBUFFERED="1")))
        print(f"GPU {g} lane {k}: {len(q)} runs", flush=True)
    t0 = time.time()
    while any(p.poll() is None for p in procs):
        time.sleep(300)
        n = len(glob.glob(os.path.join(args.out, "runs", "*", "result.json")))
        print(f"[{(time.time() - t0) / 60:.0f} min] {n} result files", flush=True)
    summarize(args.out)


if __name__ == "__main__":
    main()
