"""
Stage 1 (checklist S1): controlled E vs F comparison on GPU with the real pretrained GPT-2.

    python scripts/stage1_ef.py --out <dir> [--csv ETTh1.csv] [--gpus 0,1]

What it does
  1. Preflight: runs tests/test_stage1.py with the REAL pretrained GPT-2 on the GPU. Stops if any check fails.
  2. Trains E and F under one protocol (below), task loss only (alpha = 0), same seeds, two GPUs in parallel.
  3. Evaluates every best-validation checkpoint on the test split (RMSE, MAE, SMAPE, MAPE, MSE) and writes
     results.csv, summary.txt (per-architecture mean/std, per-seed paired differences E - F) and
     curves.png (train/validation loss per epoch for every run).

Protocol (frozen in protocol.json, written before any run starts)
  E: flatten head + pretrained GPT-2, GPT-2 layer norms and position embeddings trained (llm_policy=norms_pos)
  F: flatten head, no LLM
  ETTh1, lookback 512, horizon 96, chronological 12/4/4-month split, train-only z-score, input/output RevIN,
  Adam lr 1e-4, weight decay 1e-4, batch 32, at most 30 epochs, early stopping patience 5 (not before
  epoch 5), checkpoint = lowest validation MSE, test evaluated once after training. Seeds 42, 123, 456.
  alpha = 0: the TFB loss is computed but has weight 0, so the comparison concerns the forecaster only.
"""
import argparse, csv, glob, json, os, statistics, subprocess, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")

PROTOCOL = dict(
    stage="S1 E vs F", dataset="ETTh1", seq_len=512, pred_len=96, batch_size=32, learning_rate=1e-4,
    weight_decay=1e-4, epochs=30, patience=5, min_epochs=5, loss_mode="fixed_weighted", alpha=0.0,
    use_revin=True, seeds=[42, 123, 456],
    architectures={
        "E": dict(head_type="flatten", use_llm=True, llm_policy="norms_pos"),
        "F": dict(head_type="flatten", use_llm=False, llm_policy="none"),
    },
)
# longest jobs first; GPU queues are filled greedily
JOBS = [("E", 42), ("E", 123), ("E", 456), ("F", 42), ("F", 123), ("F", 456)]


def worker(args):
    """Runs one (architecture, seed) in this process; called as a subprocess."""
    sys.path.insert(0, SRC)
    import torch
    DRY = os.environ.get("STAGE1_DRY") == "1"     # plumbing test only: random GPT-2, 16 windows, CPU
    if DRY:
        from transformers import GPT2Config, GPT2Model
        GPT2Model.from_pretrained = classmethod(lambda cls, n, **k: GPT2Model(GPT2Config()))
    import train as T
    if DRY:
        _len = T.ETTh1Dataset.__len__; T.ETTh1Dataset.__len__ = lambda self: min(_len(self), 16)
    from metrics import eval_checkpoint
    arch, seed = args.job.split(":"); seed = int(seed)
    P = json.load(open(os.path.join(args.out, "protocol.json")))
    dev = "cpu" if DRY else "cuda"
    if DRY:
        P.update(epochs=2, batch_size=4, min_epochs=1)
    run_dir = os.path.join(args.out, "runs", f"{arch}_s{seed}")
    if not DRY:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    _, _, r = T.train(csv_path=args.csv, checkpoint_dir=run_dir, seq_len=P["seq_len"], pred_len=P["pred_len"],
                      batch_size=P["batch_size"], learning_rate=P["learning_rate"], weight_decay=P["weight_decay"],
                      epochs=P["epochs"], patience=P["patience"], min_epochs=P["min_epochs"],
                      loss_mode=P["loss_mode"], alpha=P["alpha"], use_revin=P["use_revin"], seed=seed,
                      run_sanity_check=False, device=dev, **P["architectures"][arch])
    minutes = (time.time() - t0) / 60
    m = eval_checkpoint(os.path.join(run_dir, "best_model.pt"), args.csv, pred_len=P["pred_len"], device=dev)
    h = r["history"]
    row = dict(arch=arch, seed=seed, best_epoch=r["best_epoch"], epochs_run=r["epochs_run"],
               best_val_mse=r["best_val_mse"], test_mse=m["mse"], test_rmse=m["rmse"], test_mae=m["mae"],
               test_smape=m["smape"], test_mape=m["mape"], n_trainable=r["n_trainable"],
               minutes=round(minutes, 1), sec_per_epoch=round(statistics.mean(x["seconds"] for x in h), 1),
               peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if not DRY else None)
    json.dump(row, open(os.path.join(run_dir, "result.json"), "w"), indent=1)
    print("RESULT", json.dumps(row), flush=True)
    if not (arch == "E" and seed == 42) and not (arch == "F" and seed == 42):
        os.remove(os.path.join(run_dir, "best_model.pt"))          # keep one checkpoint per architecture


def summarize(out):
    rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(out, "runs", "*", "result.json")))]
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(os.path.join(out, "results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    lines = ["Stage 1: E (flatten head + GPT-2) vs F (flatten head, no LLM); ETTh1, H=96, alpha=0", ""]
    for arch in ("E", "F"):
        rs = [r for r in rows if r["arch"] == arch]
        if not rs:
            continue
        lines.append(f"{arch}: n={len(rs)}")
        for k in ("test_mse", "test_rmse", "test_mae", "test_smape", "best_val_mse", "best_epoch", "sec_per_epoch", "minutes", "peak_gpu_gb", "n_trainable"):
            v = [float(r[k]) for r in rs if r[k] is not None]
            if not v:
                continue
            sd = statistics.stdev(v) if len(v) > 1 else float("nan")
            lines.append(f"   {k:14s} mean {statistics.mean(v):10.4f}  sd {sd:8.4f}  per seed {[round(x, 4) for x in v]}")
    lines += ["", "Paired differences E - F (same seed): negative = E better"]
    for s in PROTOCOL["seeds"]:
        e = [r for r in rows if r["arch"] == "E" and r["seed"] == s]
        f = [r for r in rows if r["arch"] == "F" and r["seed"] == s]
        if e and f:
            lines.append(f"   seed {s}: test MSE {e[0]['test_mse'] - f[0]['test_mse']:+.4f}   "
                         f"RMSE {e[0]['test_rmse'] - f[0]['test_rmse']:+.4f}   SMAPE {e[0]['test_smape'] - f[0]['test_smape']:+.3f}")
    lines += ["", "Reference controls (same pipeline, computed earlier): window mean 0.709, seasonal naive (24 h) 0.512, "
              "DLinear 0.373 (test MSE)", "Three seeds describe variability; they are not a significance test."]
    open(os.path.join(out, "summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for i, arch in enumerate(("E", "F")):
            for d in sorted(glob.glob(os.path.join(out, "runs", f"{arch}_s*"))):
                p = os.path.join(d, "epoch_log.json")
                if not os.path.exists(p):
                    continue
                h = json.load(open(p)); ep = [x["epoch"] for x in h]
                l, = ax[i].plot(ep, [x["val_mse"] for x in h], label=f"val {os.path.basename(d)}")
                ax[i].plot(ep, [x["train_task"] for x in h], "--", color=l.get_color(), label=f"train {os.path.basename(d)}")
            ax[i].set_title(f"{arch}: train (dashed) and validation MSE"); ax[i].set_xlabel("epoch"); ax[i].legend(fontsize=7)
        ax[0].set_ylabel("MSE (z-scored)")
        fig.tight_layout(); fig.savefig(os.path.join(out, "curves.png"), dpi=130)
    except Exception as e:
        print("plot failed:", e)


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
    os.makedirs(os.path.join(args.out, "runs"), exist_ok=True)
    json.dump(dict(PROTOCOL, csv=args.csv, git_commit=subprocess.run(
        ["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()),
        open(os.path.join(args.out, "protocol.json"), "w"), indent=1)
    if not args.skip_preflight:
        print("Preflight: Stage 1 tests with the real pretrained GPT-2 on cuda ...", flush=True)
        r = subprocess.run([sys.executable, os.path.join(REPO, "tests", "test_stage1.py"), SRC, "--gpt2", "real",
                            "--device", "cuda", "--csv", args.csv], cwd=args.out,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus.split(",")[0]))
        if r.returncode != 0:
            sys.exit("Preflight FAILED: see stage1_test_log_real_cuda.txt; no experiment was run.")
    gpus = args.gpus.split(",")
    queues = {g: [] for g in gpus}
    for i, j in enumerate(JOBS):                      # E jobs alternate GPUs, F jobs follow
        queues[gpus[i % len(gpus)]].append(j)
    procs = []
    for g, q in queues.items():
        cmds = " ; ".join(f'"{sys.executable}" "{os.path.abspath(__file__)}" --out "{args.out}" --csv "{args.csv}" '
                           f'--job {a}:{s} > "{args.out}/runs/{a}_s{s}.log" 2>&1' for a, s in q)
        procs.append(subprocess.Popen(cmds, shell=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=g, PYTHONUNBUFFERED="1")))
        print(f"GPU {g}: {q}", flush=True)
    for p in procs:
        p.wait()
    summarize(args.out)


if __name__ == "__main__":
    main()
