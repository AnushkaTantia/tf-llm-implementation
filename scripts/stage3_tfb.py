"""
Stage 3 (checklist S3): does the auxiliary TFB objective help? Fixed-alpha grid on the frozen forecaster.

    python scripts/stage3_tfb.py --out <dir> [--csv ETTh1.csv] [--gpus 0,1]

Protocol: configs/stage3_protocol.json (declared before any run). Forecaster and training: exactly
configs/frozen_forecaster.json; only alpha (loss_mode fixed_weighted) changes. The alpha = 0 arm is re-used
from Stage 2, so this script first checks that src/ is identical to the code Stage 2 ran (commit b32fe77):
if it is not, it stops, because the re-use would no longer be valid.

Jobs: for each of the 10 seeds, alpha in (0.01, 0.03, 0.1, 0.3, 1.0) -> 50 runs, two GPUs in parallel,
ordered seed by seed so that an interrupted session still leaves complete paired sets.
Per run: run_config.json, epoch_log.json (incl. train TFB loss per epoch), result.json (validation and test
metrics, per-step and per-variable test MSE). No checkpoints are kept (size).
"""
import argparse, csv, glob, json, os, statistics, subprocess, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")
CFG = os.path.join(REPO, "configs", "frozen_forecaster.json")
PROT = os.path.join(REPO, "configs", "stage3_protocol.json")
STAGE2_COMMIT = "b32fe77d4c6c496ace27c74cc63dbb5ecaa93e37"


def worker(args):
    sys.path.insert(0, SRC)
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    import train as T
    from metrics import evaluate_full
    a_str, s_str = args.job.split(":"); alpha, seed = float(a_str), int(s_str)
    C = json.load(open(CFG)); A, D, R = C["architecture"], C["data"], C["training"]
    DRY = os.environ.get("STAGE_DRY") == "1"
    dev = "cpu" if DRY else "cuda"
    if DRY:
        _len = T.ETTh1Dataset.__len__; T.ETTh1Dataset.__len__ = lambda self: min(_len(self), 16)
        R = dict(R, max_epochs=2, batch_size=4, min_epochs=1)
    run_dir = os.path.join(args.out, "runs", f"a{a_str}_s{seed}")
    if not DRY:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    f, _, r = T.train(csv_path=args.csv, checkpoint_dir=run_dir, seq_len=D["seq_len"], pred_len=D["pred_len"],
                      batch_size=R["batch_size"], learning_rate=R["learning_rate"], weight_decay=R["weight_decay"],
                      epochs=R["max_epochs"], patience=R["early_stopping_patience"], min_epochs=R["min_epochs"],
                      loss_mode="fixed_weighted", alpha=alpha, use_revin=A["use_revin"], seed=seed,
                      head_type=A["head_type"], use_llm=A["use_llm"], llm_policy=A["llm_policy"],
                      run_sanity_check=False, device=dev)
    f.eval()
    out = dict(alpha=alpha, seed=seed, best_epoch=r["best_epoch"], epochs_run=r["epochs_run"],
               best_val_mse=r["best_val_mse"], minutes=round((time.time() - t0) / 60, 1),
               peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if not DRY else 0.0, config=C["name"])
    for split in ("val", "test"):
        ds = T.ETTh1Dataset(args.csv, flag=split, seq_len=D["seq_len"], pred_len=D["pred_len"], patch_len=A["patch_len"])
        dl = DataLoader(ds, batch_size=64, shuffle=False)
        out.update({f"{split}_{k}": v for k, v in evaluate_full(f, dl, dev).items()})
        if split == "test":
            se_t = se_v = None
            with torch.no_grad():
                for b in dl:
                    y, _ = f(b["x_patched"].to(dev), b["x"].to(dev))
                    e = (y.double().cpu() - b["y"].double()) ** 2
                    se_t = e.sum((0, 2)) if se_t is None else se_t + e.sum((0, 2))
                    se_v = e.sum((0, 1)) if se_v is None else se_v + e.sum((0, 1))
            n = len(ds)
            out["test_mse_per_step"] = [round(float(v) / (n * D["variables"]), 6) for v in se_t]
            out["test_mse_per_variable"] = [round(float(v) / (n * D["pred_len"]), 6) for v in se_v]
    json.dump(out, open(os.path.join(run_dir, "result.json"), "w"), indent=1)
    p = os.path.join(run_dir, "best_model.pt")
    if os.path.exists(p):
        os.remove(p)                                   # no checkpoints kept (Output size)
    print("RESULT", json.dumps({k: v for k, v in out.items() if not isinstance(v, list)}), flush=True)


def summarize(out):
    rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(out, "runs", "*", "result.json")))]
    if not rows:
        return
    keys = [k for k in rows[0] if not isinstance(rows[0][k], list)]
    with open(os.path.join(out, "results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    lines = ["Stage 3: fixed alpha grid on the frozen forecaster (alpha = 0 is in the Stage 2 results)", ""]
    for a in sorted({r["alpha"] for r in rows}):
        rs = [r for r in rows if r["alpha"] == a]
        f = lambda k: (statistics.mean(r[k] for r in rs), statistics.stdev(r[k] for r in rs) if len(rs) > 1 else 0)
        (vm, vs), (tm, ts), (sm, _) = f("best_val_mse"), f("test_mse"), f("test_smape")
        lines.append(f"alpha {a:<5}  n={len(rs):2d}  best val MSE {vm:.4f}±{vs:.4f}  test MSE {tm:.4f}±{ts:.4f}  SMAPE {sm:.2f}")
    open(os.path.join(out, "summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--job", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--skip_preflight", action="store_true")
    ap.add_argument("--skip_code_check", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    args.csv = args.csv or sorted(glob.glob("/kaggle/input/**/ETTh1.csv", recursive=True)
                                  + glob.glob(os.path.join(REPO, "**", "ETTh1.csv"), recursive=True))[0]
    if args.job:
        return worker(args)
    C, Pr = json.load(open(CFG)), json.load(open(PROT))
    if not args.skip_code_check:
        r = subprocess.run(["git", "-C", REPO, "diff", "--quiet", STAGE2_COMMIT, "HEAD", "--", "src"])
        if r.returncode != 0:
            sys.exit("STOP: src/ differs from the code Stage 2 ran (b32fe77); the alpha = 0 arm cannot be re-used.")
        print("code check: src/ identical to the Stage 2 commit -> alpha = 0 arm re-used from Stage 2")
    os.makedirs(os.path.join(args.out, "runs"), exist_ok=True)
    json.dump(dict(frozen=C, protocol=Pr, csv=args.csv, git_commit=subprocess.run(
        ["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()),
        open(os.path.join(args.out, "protocol.json"), "w"), indent=1)
    if not args.skip_preflight:
        print("Preflight: Stage 1 tests on cuda ...", flush=True)
        r = subprocess.run([sys.executable, os.path.join(REPO, "tests", "test_stage1.py"), SRC, "--gpt2", "real",
                            "--device", "cuda", "--csv", args.csv], cwd=args.out,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus.split(",")[0]))
        if r.returncode != 0:
            sys.exit("Preflight FAILED: see stage1_test_log_real_cuda.txt; no experiment was run.")
    jobs = [(a, s) for s in C["seeds"] for a in Pr["arms"]["fixed_alpha_grid"]]
    gpus = args.gpus.split(",")
    queues = {g: [] for g in gpus}
    for i, j in enumerate(jobs):
        queues[gpus[i % len(gpus)]].append(j)
    procs = []
    for g, q in queues.items():
        cmds = " ; ".join(f'"{sys.executable}" "{os.path.abspath(__file__)}" --out "{args.out}" --csv "{args.csv}" '
                          f'--job {a}:{s} > "{args.out}/runs/a{a}_s{s}.log" 2>&1' for a, s in q)
        procs.append(subprocess.Popen(cmds, shell=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=g, PYTHONUNBUFFERED="1")))
        print(f"GPU {g}: {len(q)} runs", flush=True)
    for p in procs:
        p.wait()
    summarize(args.out)


if __name__ == "__main__":
    main()
