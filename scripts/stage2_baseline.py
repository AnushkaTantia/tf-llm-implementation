"""
Stage 2 (checklist S2): credible baseline -- the frozen corrected forecaster over the 10 predeclared seeds.

    python scripts/stage2_baseline.py --out <dir> [--csv ETTh1.csv] [--gpus 0,1]

Uses configs/frozen_forecaster.json (architecture, protocol and seed list frozen at the S1 exit); nothing in
this script changes the model or the protocol. alpha = 0 (task loss only): the forecaster is assessed before
any auxiliary objective, and these runs are re-used as the alpha = 0 arm of S3/S4 only if the frozen
configuration and code are unchanged.

Per seed it saves
  run_config.json, epoch_log.json          (written by train())
  preds.npz     test forecasts of the selected (best-validation) checkpoint, float16 (validation metrics in result.json)
  result.json   test metrics (src/metrics.py), per-horizon-step and per-variable test MSE, and the
                input-order test: test MSE when the 32 input patches are presented in a shuffled order
                (fixed permutation). A forecaster that uses temporal order should get clearly worse.
Then summary.txt and results.csv over seeds. Analysis against the controls is done offline
(s2/stage2_analysis.py) from these files.
"""
import argparse, csv, glob, json, os, statistics, subprocess, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")
CFG = os.path.join(REPO, "configs", "frozen_forecaster.json")


def worker(args):
    sys.path.insert(0, SRC)
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    import train as T
    from metrics import evaluate_full
    seed = int(args.job)
    C = json.load(open(CFG)); A, D, R = C["architecture"], C["data"], C["training"]
    DRY = os.environ.get("STAGE_DRY") == "1"          # plumbing test only: CPU, 16 windows, 2 epochs
    dev = "cpu" if DRY else "cuda"
    if DRY:
        _len = T.ETTh1Dataset.__len__; T.ETTh1Dataset.__len__ = lambda self: min(_len(self), 16)
        R = dict(R, max_epochs=2, batch_size=4, min_epochs=1)
    run_dir = os.path.join(args.out, "runs", f"s{seed}")
    if not DRY:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    f, _, r = T.train(csv_path=args.csv, checkpoint_dir=run_dir, seq_len=D["seq_len"], pred_len=D["pred_len"],
                      batch_size=R["batch_size"], learning_rate=R["learning_rate"], weight_decay=R["weight_decay"],
                      epochs=R["max_epochs"], patience=R["early_stopping_patience"], min_epochs=R["min_epochs"],
                      loss_mode="fixed_weighted", alpha=0.0, use_revin=A["use_revin"], seed=seed,
                      head_type=A["head_type"], use_llm=A["use_llm"], llm_policy=A["llm_policy"],
                      run_sanity_check=False, device=dev)
    minutes = (time.time() - t0) / 60
    f.eval()                                   # train() has re-loaded the best-validation checkpoint
    out = dict(seed=seed, best_epoch=r["best_epoch"], epochs_run=r["epochs_run"], best_val_mse=r["best_val_mse"],
               minutes=round(minutes, 1), peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if not DRY else 0.0,
               n_trainable=r["n_trainable"], config=C["name"])
    preds = {}
    for split in ("val", "test"):
        ds = T.ETTh1Dataset(args.csv, flag=split, seq_len=D["seq_len"], pred_len=D["pred_len"], patch_len=A["patch_len"])
        dl = DataLoader(ds, batch_size=64, shuffle=False)
        m = evaluate_full(f, dl, dev)
        out.update({f"{split}_{k}": v for k, v in m.items()})
        P = []
        with torch.no_grad():
            for b in dl:
                y, _ = f(b["x_patched"].to(dev), b["x"].to(dev)); P.append(y.float().cpu().numpy())
        preds[split] = np.concatenate(P).astype(np.float16)
        if split == "test":
            Y = np.stack([np.asarray(ds[i]["y"]) for i in range(len(ds))]).astype(np.float64)
            e = (preds[split].astype(np.float64) - Y) ** 2
            out["test_mse_per_step"] = [round(float(v), 6) for v in e.mean((0, 2))]
            out["test_mse_per_variable"] = [round(float(v), 6) for v in e.mean((0, 1))]
            # input-order test: same windows, the 32 input patches in a fixed shuffled order
            g = torch.Generator().manual_seed(0)
            perm = torch.randperm(D["seq_len"] // A["patch_len"], generator=g)
            se, n = 0.0, 0
            with torch.no_grad():
                for b in dl:
                    xp = b["x_patched"][:, perm]                                  # (B, P, patch, V)
                    xr = xp.reshape(xp.shape[0], -1, xp.shape[-1])
                    y, _ = f(xp.to(dev), xr.to(dev))
                    se += float(((y.double().cpu() - b["y"].double()) ** 2).sum()); n += b["y"].numel()
            out["test_mse_shuffled_patch_order"] = se / n
    np.savez_compressed(os.path.join(run_dir, "preds.npz"), test=preds["test"])   # test forecasts only (size)
    json.dump(out, open(os.path.join(run_dir, "result.json"), "w"), indent=1)
    print("RESULT", json.dumps({k: v for k, v in out.items() if not isinstance(v, list)}), flush=True)
    os.remove(os.path.join(run_dir, "best_model.pt")) if seed != 42 else None   # keep one checkpoint


def summarize(out):
    rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(out, "runs", "*", "result.json")))]
    if not rows:
        return
    keys = [k for k in rows[0] if not isinstance(rows[0][k], list)]
    with open(os.path.join(out, "results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    lines = [f"Stage 2: frozen forecaster ({rows[0]['config']}), alpha = 0, ETTh1 H=96, n = {len(rows)} seeds", ""]
    for k in ("test_mse", "test_rmse", "test_mae", "test_smape", "test_mape", "val_mse", "best_epoch",
              "test_mse_shuffled_patch_order", "minutes", "peak_gpu_gb"):
        v = [float(r[k]) for r in rows]
        lines.append(f"   {k:30s} mean {statistics.mean(v):9.4f}  sd {statistics.stdev(v) if len(v) > 1 else 0:8.4f}  "
                     f"min {min(v):8.4f}  max {max(v):8.4f}")
    lines.append("\nper seed: " + "  ".join(f"s{r['seed']}={r['test_mse']:.4f}" for r in rows))
    open(os.path.join(out, "summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--job", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--skip_preflight", action="store_true")
    args = ap.parse_args()
    args.csv = args.csv or sorted(glob.glob("/kaggle/input/**/ETTh1.csv", recursive=True)
                                  + glob.glob(os.path.join(REPO, "**", "ETTh1.csv"), recursive=True))[0]
    if args.job:
        return worker(args)
    C = json.load(open(CFG))
    os.makedirs(os.path.join(args.out, "runs"), exist_ok=True)
    json.dump(dict(C, csv=args.csv, git_commit=subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
              capture_output=True, text=True).stdout.strip()), open(os.path.join(args.out, "protocol.json"), "w"), indent=1)
    if not args.skip_preflight:
        print("Preflight: Stage 1 tests on cuda ...", flush=True)
        r = subprocess.run([sys.executable, os.path.join(REPO, "tests", "test_stage1.py"), SRC, "--gpt2", "real",
                            "--device", "cuda", "--csv", args.csv], cwd=args.out,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus.split(",")[0]))
        if r.returncode != 0:
            sys.exit("Preflight FAILED: see stage1_test_log_real_cuda.txt; no experiment was run.")
    gpus = args.gpus.split(",")
    queues = {g: [] for g in gpus}
    for i, s in enumerate(C["seeds"]):
        queues[gpus[i % len(gpus)]].append(s)
    procs = []
    for g, q in queues.items():
        cmds = " ; ".join(f'"{sys.executable}" "{os.path.abspath(__file__)}" --out "{args.out}" --csv "{args.csv}" '
                          f'--job {s} > "{args.out}/runs/s{s}.log" 2>&1' for s in q)
        procs.append(subprocess.Popen(cmds, shell=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=g, PYTHONUNBUFFERED="1")))
        print(f"GPU {g}: seeds {q}", flush=True)
    for p in procs:
        p.wait()
    summarize(args.out)


if __name__ == "__main__":
    main()
