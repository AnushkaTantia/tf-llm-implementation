"""
run_all.py -- unattended experiment queue for the journal paper (ETTh1, RevIN, lr=1e-5).

Run from the repo root:   python scripts\run_all.py --supervise

* Resumable: every finished run is a row in logs/final_results.csv and is skipped on restart.
* A run that fails or crashes is retried (after a 30-min wait); after 3 attempts it is skipped
  (see final_failures.csv). Start with  --supervise  so a hard crash restarts the script.
* Runs are in PRIORITY order, so if time runs out the most important results exist.
* Disk: checkpoints are kept only for seed 42 and for hmm / oahmm_0.5 at H=96; others are deleted
  after their test score is recorded (each checkpoint is ~0.5 GB).
* After every run, logs/summary.txt is rewritten (mean +- std per method), and
  logs/progress.txt shows what is running / done.
"""
import os, sys, csv, glob, gc, random, time, shutil, traceback, math, statistics
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
import numpy as np
import torch
import train as T                           # src/train.py
import oahmm_scheduler as OA                # src/oahmm_scheduler.py

LR, BATCH = 1e-5, 8
MAX_EPOCHS, PATIENCE, MIN_EPOCHS = 30, 8, 10
MAX_TRIES = 3                                   # attempts per run before it is skipped
FAIL_WAIT = int(os.environ.get("FAIL_WAIT", 1800))   # seconds to wait after a failure
LOG_DIR = os.path.join(REPO, "logs"); os.makedirs(LOG_DIR, exist_ok=True)
RES = os.path.join(LOG_DIR, "final_results.csv")
FAIL = os.path.join(LOG_DIR, "final_failures.csv")
ATTEMPTS = os.path.join(LOG_DIR, "final_attempts.csv")
PROGRESS = os.path.join(LOG_DIR, "progress.txt")
SUMMARY = os.path.join(LOG_DIR, "summary.txt")
CSV_PATH = sorted(glob.glob(os.path.join(REPO, "**", "ETTh1.csv"), recursive=True))[0]


class OAHMMNoSlope(OA.OAHMMScheduler):
    """Ablation: OA-HMM with the overfitting-velocity signal switched off (slope always 0)."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.window = 10 ** 9           # slope is used only when len(gap) >= window -> never


METHODS = {
    "sum":            dict(loss_mode="sum"),
    "task_only":      dict(loss_mode="fixed_weighted", alpha=0.0),
    "fixed_0.3":      dict(loss_mode="fixed_weighted", alpha=0.3),
    "kendall":        dict(loss_mode="variable_weighted"),
    "warmup":         dict(loss_mode="time_varying", alpha=0.3, warmup_epochs=15),
    "kalman":         dict(loss_mode="kalman_varying"),
    "hmm":            dict(loss_mode="hmm_varying"),
    "oahmm_0.5":      dict(loss_mode="oahmm_varying", alpha_over=0.5),
    # ablations
    "oahmm_0.05":     dict(loss_mode="oahmm_varying", alpha_over=0.05),
    "oahmm_noslope":  dict(loss_mode="oahmm_varying", alpha_over=0.5),
}
MAIN = ["sum", "task_only", "fixed_0.3", "kendall", "warmup", "kalman", "hmm", "oahmm_0.5"]
ABL = ["oahmm_0.05", "oahmm_noslope"]
HORIZON_METHODS = ["sum", "hmm", "oahmm_0.5"]

# (method, seed, pred_len) in priority order
QUEUE = []
QUEUE += [(m, s, 96) for s in (42, 123, 456) for m in MAIN]          # A: main, 3 seeds   (24)
QUEUE += [(m, s, 96) for s in (42, 123, 456) for m in ABL]           # B: ablations       (6)
QUEUE += [(m, s, 96) for s in (789, 2024) for m in MAIN]             # C: main, seeds 4-5 (16)
QUEUE += [(m, s, 96) for s in (789, 2024) for m in ABL]              # D: ablations 4-5   (4)
QUEUE += [(m, s, h) for h in (192, 336, 720) for s in (42, 123, 456) for m in HORIZON_METHODS]  # E (27)

FIELDS = ["method", "seed", "pred_len", "lr", "best_val_mse", "test_mse", "test_mae",
          "best_epoch", "epochs_run", "minutes", "finished_at"]


def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def append_row(path, fields, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)


def key(r):
    return (r["method"], int(r["seed"]), int(r["pred_len"]))


def write_summary():
    groups = defaultdict(list)
    for r in read_rows(RES):
        groups[(int(r["pred_len"]), r["method"])].append(r)
    lines = [f"Summary (lr={LR}, updated {time.strftime('%Y-%m-%d %H:%M')})",
             f"{'H':>4}  {'method':15s} {'n':>2}  {'test MSE mean+-std':>20}  {'test MAE mean+-std':>20}  best epochs"]
    for (h, m) in sorted(groups):
        rs = groups[(h, m)]
        mse = [float(r["test_mse"]) for r in rs]; mae = [float(r["test_mae"]) for r in rs]
        sd = lambda v: statistics.stdev(v) if len(v) > 1 else 0.0
        lines.append(f"{h:>4}  {m:15s} {len(rs):>2}  {statistics.mean(mse):.4f} +- {sd(mse):.4f}"
                     f"      {statistics.mean(mae):.4f} +- {sd(mae):.4f}   "
                     + ",".join(r["best_epoch"] for r in rs))
    open(SUMMARY, "w").write("\n".join(lines) + "\n")


class Tee:
    def __init__(self, *f): self.f = f
    def write(self, o):
        for x in self.f:
            try: x.write(o); x.flush()
            except Exception: pass
    def flush(self):
        for x in self.f:
            try: x.flush()
            except Exception: pass


def main():
    done = {key(r) for r in read_rows(RES)}
    tries = defaultdict(int)                  # counts starts, so hard crashes (process killed) count too
    for r in read_rows(ATTEMPTS):
        tries[key(r)] += 1
    todo = [q for q in QUEUE if q not in done and tries[q] < MAX_TRIES]
    print(f"{len(done)} done, {len(todo)} to run, {sum(1 for q in QUEUE if q not in done and tries[q] >= MAX_TRIES)} given up. Data: {CSV_PATH}")
    orig_sched = T.OAHMMScheduler
    for i, (m, seed, h) in enumerate(todo, 1):
        tag = f"final_{m}_s{seed}_H{h}"
        open(PROGRESS, "w").write(f"{time.strftime('%Y-%m-%d %H:%M')}  running {i}/{len(todo)}: {tag}\n"
                                  f"finished so far: {len(done)} of {len(QUEUE)}\n")
        print(f"\n[{i}/{len(todo)}] {tag}", flush=True)
        append_row(ATTEMPTS, ["method", "seed", "pred_len", "at"],
                   dict(method=m, seed=seed, pred_len=h, at=time.strftime("%Y-%m-%d %H:%M")))
        T.OAHMMScheduler = OAHMMNoSlope if m == "oahmm_noslope" else orig_sched
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        ckpt_dir = os.path.join(REPO, "checkpoints", tag); os.makedirs(ckpt_dir, exist_ok=True)
        log = open(os.path.join(LOG_DIR, f"{tag}_log.txt"), "w", encoding="utf-8")
        old = sys.stdout; sys.stdout = Tee(old, log)
        t0 = time.time()
        try:
            _, _, res = T.train(csv_path=CSV_PATH, checkpoint_dir=ckpt_dir, device="cuda",
                                pred_len=h, epochs=MAX_EPOCHS, patience=PATIENCE, min_epochs=MIN_EPOCHS,
                                batch_size=BATCH, learning_rate=LR, use_revin=True, log_every=200,
                                **METHODS[m])
            print("RESULT", tag, res)
            append_row(RES, FIELDS, dict(method=m, seed=seed, pred_len=h, lr=LR,
                       **{k: res[k] for k in ("best_val_mse", "test_mse", "test_mae", "best_epoch", "epochs_run")},
                       minutes=round((time.time() - t0) / 60, 1), finished_at=time.strftime("%Y-%m-%d %H:%M")))
            done.add((m, seed, h))
            if not (seed == 42 or (m in ("hmm", "oahmm_0.5") and h == 96)):
                shutil.rmtree(ckpt_dir, ignore_errors=True)
        except Exception as e:
            print("FAILED", tag, repr(e)); traceback.print_exc(file=sys.stdout)
            append_row(FAIL, ["method", "seed", "pred_len", "error", "at"],
                       dict(method=m, seed=seed, pred_len=h, error=repr(e)[:300], at=time.strftime("%Y-%m-%d %H:%M")))
            print(f"Waiting {FAIL_WAIT // 60} min before continuing (e.g. GPU busy with someone else's job)...")
            time.sleep(FAIL_WAIT)
        finally:
            sys.stdout = old
            log.close()
            T.OAHMMScheduler = orig_sched
        gc.collect(); torch.cuda.empty_cache()
        try: write_summary()
        except Exception as e: print("summary failed:", e)
    write_summary()
    tries = defaultdict(int)
    for r in read_rows(ATTEMPTS):
        tries[key(r)] += 1
    if any(q not in done and tries[q] < MAX_TRIES for q in QUEUE):
        print("Some runs failed once; exiting so the supervisor restarts and retries them.")
        sys.exit(3)
    left = [q for q in QUEUE if q not in done]
    open(PROGRESS, "w").write(f"{time.strftime('%Y-%m-%d %H:%M')}  QUEUE FINISHED. {len(done)} of {len(QUEUE)} done; "
                              f"{len(left)} not done (see final_failures.csv).\n")
    open(os.path.join(LOG_DIR, "ALL_DONE.flag"), "w").write("done\n")
    print("ALL DONE. Send me the logs folder.")


def supervise():
    import subprocess
    flag = os.path.join(LOG_DIR, "ALL_DONE.flag")
    if os.path.exists(flag):
        os.remove(flag)
    for attempt in range(1, 41):
        subprocess.run([sys.executable, os.path.abspath(__file__)], cwd=REPO)
        if os.path.exists(flag):
            print("\n===== ALL DONE. Results: logs/final_results.csv and logs/summary.txt =====")
            return
        print(f"Worker stopped (restart {attempt}/40); restarting in 60 s...", flush=True)
        time.sleep(60)


if __name__ == "__main__":
    supervise() if "--supervise" in sys.argv else main()
