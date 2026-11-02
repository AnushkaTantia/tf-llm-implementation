"""
Stage 3 diagnostics (offline, descriptive; no new runs, no test-based selection).
Explains HOW the fixed TFB weight changes training, to inform the reassessment of the central premise.

  python s3/stage3_diagnostics.py --s3 <stage3 dir> --s2 <stage2 dir> --out s3/report
"""
import argparse, glob, json, os
import numpy as np
from scipy.stats import spearmanr

VARS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


def load(a):
    runs = {}
    for p in glob.glob(os.path.join(a.s2, "runs", "s*", "result.json")):
        r = json.load(open(p)); r["alpha"] = 0.0
        r["log"] = json.load(open(os.path.join(os.path.dirname(p), "epoch_log.json"))); runs[(0.0, r["seed"])] = r
    for p in glob.glob(os.path.join(a.s3, "runs", "a*_s*", "result.json")):
        r = json.load(open(p))
        r["log"] = json.load(open(os.path.join(os.path.dirname(p), "epoch_log.json"))); runs[(float(r["alpha"]), r["seed"])] = r
    return runs


def slope(y):
    x = np.arange(len(y)); return float(np.polyfit(x, y, 1)[0]) if len(y) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s3", required=True); ap.add_argument("--s2", required=True); ap.add_argument("--out", default="s3/report")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    runs = load(a)
    alphas = sorted({k[0] for k in runs}); seeds = sorted({k[1] for k in runs if k[0] == 0.0})
    L = []; say = lambda s="": (print(s), L.append(s))
    m = lambda xs: (float(np.mean(xs)), float(np.std(xs, ddof=1)))
    out = {}

    say("# Stage 3 diagnostics: how the fixed TFB weight changes training\n")
    say("Descriptive analysis of the logged trajectories of all 60 runs (6 arms x 10 seeds). Nothing here selects a "
        "configuration; it explains the Stage 3 result.\n")

    # D1 optimisation of the task: training task loss at matched epochs
    say("## D1. Forecasting fit and generalisation per arm (mean over seeds)\n")
    say("| alpha | val MSE ep.1 | best epoch | train task loss ep.1 | train task loss ep.5 | train task at best | "
        "gap at best (val-train) | gap slope ep.1-5 | TFB loss ep.1 | TFB loss at best | alpha*TFB / task at best |")
    say("|---|---|---|---|---|---|---|---|---|---|---|")
    for al in alphas:
        rs = [runs[(al, s)] for s in seeds]
        v1 = [r["log"][0]["val_mse"] for r in rs]; be = [r["best_epoch"] for r in rs]
        t1 = [r["log"][0]["train_task"] for r in rs]; t5 = [r["log"][4]["train_task"] for r in rs]
        tb = [r["log"][r["best_epoch"] - 1]["train_task"] for r in rs]
        gb = [r["log"][r["best_epoch"] - 1]["gap"] for r in rs]
        gs = [slope([e["gap"] for e in r["log"][:5]]) for r in rs]
        f1 = [r["log"][0]["train_tfb"] for r in rs]; fb = [r["log"][r["best_epoch"] - 1]["train_tfb"] for r in rs]
        ratio = [al * f / t for f, t in zip(fb, tb)]
        row = dict(val1=m(v1), best_epoch=m(be), task1=m(t1), task5=m(t5), task_best=m(tb), gap_best=m(gb),
                   gap_slope=m(gs), tfb1=m(f1), tfb_best=m(fb), ratio=m(ratio))
        out[str(al)] = row
        say(f"| {al} | {row['val1'][0]:.4f} | {row['best_epoch'][0]:.1f} | {row['task1'][0]:.4f} | {row['task5'][0]:.4f} | "
            f"{row['task_best'][0]:.4f} | {row['gap_best'][0]:.4f} | {row['gap_slope'][0]:+.4f} | {row['tfb1'][0]:.3f} | "
            f"{row['tfb_best'][0]:.3f} | {row['ratio'][0]:.3f} |")
    say()

    # D2 paired per-epoch validation curves: does TFB hurt from the first epoch?
    say("## D2. Validation MSE by epoch, paired difference to alpha = 0 (mean over seeds; epochs all runs reached)\n")
    emin = min(len(r["log"]) for r in runs.values())
    say("| alpha | " + " | ".join(f"ep {e}" for e in range(1, emin + 1)) + " |")
    say("|---|" + "---|" * emin)
    for al in alphas[1:]:
        d = [[runs[(al, s)]["log"][e]["val_mse"] - runs[(0.0, s)]["log"][e]["val_mse"] for s in seeds] for e in range(emin)]
        say(f"| {al} | " + " | ".join(f"{np.mean(x):+.4f}" for x in d) + " |")
        out.setdefault("val_diff_by_epoch", {})[str(al)] = [float(np.mean(x)) for x in d]
    say()

    # D3 where the harm occurs: per variable and per horizon, test MSE paired diff vs alpha = 0
    say("## D3. Where the error increases: test MSE, paired difference to alpha = 0 (mean over seeds)\n")
    say("| alpha | " + " | ".join(VARS) + " | steps 1-24 | 25-48 | 49-72 | 73-96 |")
    say("|---|" + "---|" * 11)
    for al in alphas[1:]:
        pv = np.mean([np.array(runs[(al, s)]["test_mse_per_variable"]) - np.array(runs[(0.0, s)]["test_mse_per_variable"]) for s in seeds], 0)
        ps = np.mean([np.array(runs[(al, s)]["test_mse_per_step"]) - np.array(runs[(0.0, s)]["test_mse_per_step"]) for s in seeds], 0)
        q = [ps[i * 24:(i + 1) * 24].mean() for i in range(4)]
        out.setdefault("per_var", {})[str(al)] = pv.tolist(); out.setdefault("per_quarter", {})[str(al)] = q
        say(f"| {al} | " + " | ".join(f"{x:+.4f}" for x in pv) + " | " + " | ".join(f"{x:+.4f}" for x in q) + " |")
    say()

    # D4 alignment vs accuracy across all runs
    tfb = [r["log"][r["best_epoch"] - 1]["train_tfb"] for r in runs.values()]
    val = [r["best_val_mse"] for r in runs.values()]
    rho, p = spearmanr(tfb, val)
    pos = [(r["log"][r["best_epoch"] - 1]["train_tfb"], r["best_val_mse"]) for k, r in runs.items() if k[0] > 0]
    rho2, p2 = spearmanr([x for x, _ in pos], [y for _, y in pos])
    out["spearman_tfb_val_all"] = [float(rho), float(p)]; out["spearman_tfb_val_pos"] = [float(rho2), float(p2)]
    say("## D4. Alignment vs accuracy\n")
    say(f"- Spearman correlation between the TFB loss at the selected epoch and the best validation MSE, all 60 runs: "
        f"rho = {rho:+.3f} (p = {p:.2g}); alpha > 0 runs only: rho = {rho2:+.3f} (p = {p2:.2g}). Negative rho: "
        f"runs whose time and frequency embeddings are more aligned (lower TFB loss) forecast worse.")
    t0 = [e["train_tfb"] for s in seeds for e in runs[(0.0, s)]["log"]]
    say(f"- At alpha = 0 the TFB loss is computed but not optimised; it stays at {np.mean(t0):.3f} "
        f"(range {np.min(t0):.3f}-{np.max(t0):.3f}) throughout training: the forecasting objective alone does not align the views.\n")

    json.dump(out, open(os.path.join(a.out, "s3_diagnostics.json"), "w"), indent=1)
    open(os.path.join(a.out, "STAGE3_DIAGNOSTICS.md"), "w").write("\n".join(L) + "\n")

    # figure: validation curves per arm + dose-response of harm by variable
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    cm = plt.cm.viridis(np.linspace(0, 0.9, len(alphas)))
    for c, al in zip(cm, alphas):
        mv = np.mean([[runs[(al, s)]["log"][e]["val_mse"] for e in range(emin)] for s in seeds], 0)
        mt = np.mean([[runs[(al, s)]["log"][e]["train_task"] for e in range(emin)] for s in seeds], 0)
        ax[0].plot(range(1, emin + 1), mv, "-o", ms=3, color=c, label=f"alpha={al}")
        ax[1].plot(range(1, emin + 1), mt, "-o", ms=3, color=c, label=f"alpha={al}")
    ax[0].set_title("Validation MSE (mean over 10 seeds)"); ax[0].set_xlabel("epoch"); ax[0].legend(fontsize=7)
    ax[1].set_title("Training forecasting loss (mean over 10 seeds)"); ax[1].set_xlabel("epoch")
    ax[2].scatter([x for x, _ in pos], [y for _, y in pos], s=12, c=[np.log10(k[0]) for k in runs if k[0] > 0], cmap="viridis")
    ax[2].scatter([runs[(0.0, s)]["log"][runs[(0.0, s)]["best_epoch"] - 1]["train_tfb"] for s in seeds],
                  [runs[(0.0, s)]["best_val_mse"] for s in seeds], s=16, marker="x", color="k", label="alpha=0")
    ax[2].set_xlabel("TFB loss at selected epoch"); ax[2].set_ylabel("best validation MSE"); ax[2].legend(fontsize=7)
    ax[2].set_title(f"Alignment vs accuracy (Spearman rho={rho:+.2f})")
    for x in ax: x.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_s3_diagnostics.png"), dpi=150)


if __name__ == "__main__":
    main()
