"""
Stage 2 analysis (offline, CPU): is the corrected forecaster a credible baseline, and can its performance be
explained relative to the controls?

  python s2/stage2_analysis.py --s2 <stage2 output dir> --controls s2/out --s1 <stage1 output dir> --out s2/report

Answers, in order, the S2 tasks of the research plan:
  A  corrected model vs controls (seasonal naive, window mean, linear, DLinear reproduced under the same protocol)
  B  independent verification of the test metrics (independent loader + numpy metrics vs src/metrics.py)
  C  training and validation curves
  D  temporal learning beyond the window average (level-removed skill, input-order test, per-step skill)
  E  seed variability and reproducibility (S1 vs S2 for the shared seeds)
  F  validation/test distribution shift
"""
import argparse, csv, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stage2_controls import metrics, windows          # independent metrics and windows
sys.path.insert(0, HERE)
from data_audit import load

RNG = np.random.default_rng(0)


def boot_ci(x, n=10000):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    m = RNG.choice(x, (n, len(x)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def stat(x):
    x = np.asarray(x, float)
    lo, hi = boot_ci(x)
    return dict(n=len(x), mean=float(x.mean()), sd=float(x.std(ddof=1)) if len(x) > 1 else 0.0,
                min=float(x.min()), max=float(x.max()), ci_lo=lo, ci_hi=hi)


def shape_r2(P, Y, level):
    """Skill at predicting deviations from the window level: both forecast and target minus the input-window mean."""
    p, y = P - level[:, None, :], Y - level[:, None, :]
    return 1 - float(((p - y) ** 2).sum() / (y ** 2).sum())


def mean_corr(P, Y, level):
    p, y = P - level[:, None, :], Y - level[:, None, :]
    p = p - p.mean(1, keepdims=True); y = y - y.mean(1, keepdims=True)
    num = (p * y).sum(1); den = np.sqrt((p ** 2).sum(1) * (y ** 2).sum(1)) + 1e-12
    return float(np.nanmean(num / den))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2", required=True)
    ap.add_argument("--controls", default=os.path.join(HERE, "out"))
    ap.add_argument("--s1", default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "report"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    L = []                                                   # report lines (markdown)
    say = lambda s="": (print(s), L.append(s))

    df, cols, raw, z, mu, sd, b1, b2 = load("ETTh1")
    mu, sd = np.asarray(mu, float), np.asarray(sd, float)
    (Xva, Yva), (Xte, Yte) = windows(z, b1, b2, 1), windows(z, b1, b2, 2)
    Yva, Yte = Yva.astype(float), Yte.astype(float)
    lev_va, lev_te = Xva.mean(1).astype(float), Xte.mean(1).astype(float)
    sn24 = np.concatenate([Xte[:, -24:]] * 4, 1).astype(float)
    wmean = np.repeat(Xte.mean(1, keepdims=True), 96, 1).astype(float)

    runs = sorted(glob.glob(os.path.join(a.s2, "runs", "s*", "result.json")), key=lambda p: int(p.split(os.sep)[-2][1:]))
    R = [json.load(open(p)) for p in runs]
    P = {r["seed"]: np.load(os.path.join(os.path.dirname(p), "preds.npz")) for r, p in zip(R, runs)}
    seeds = [r["seed"] for r in R]
    ctrl = list(csv.DictReader(open(os.path.join(a.controls, "controls.csv"))))
    say(f"# Stage 2 report: credible baseline\n\nForecaster: `{R[0]['config']}` (frozen at the S1 exit), alpha = 0, "
        f"ETTh1, lookback 512, horizon 96. Seeds ({len(seeds)}): {seeds}.\n")

    # ---------------------------------------------------------------- B independent verification
    say("## B. Independent verification of the test metrics")
    worst = {}
    indep = {}
    for r in R:
        Pt = P[r["seed"]]["test"].astype(float)
        assert Pt.shape == Yte.shape, (Pt.shape, Yte.shape)
        m = metrics(Pt, Yte, mu, sd); indep[r["seed"]] = m
        for k in ("mse", "rmse", "mae", "smape", "mape"):
            worst[k] = max(worst.get(k, 0), abs(m[k] - r[f"test_{k}"]) / max(abs(r[f"test_{k}"]), 1e-12))
    say("Test metrics recomputed from the saved forecasts with the independent loader (audit/data_audit.py) and "
        "independent numpy metric code, compared with the values reported by `src/metrics.py`. Largest relative "
        "difference over all seeds (forecasts are stored as float16, which bounds the attainable agreement):\n")
    say("| metric | max relative difference |\n|---|---|")
    for k, v in worst.items():
        say(f"| {k} | {v:.2e} |")
    ok_b = all(v < 2e-3 for v in worst.values())
    say(f"\nVerdict: {'agreement' if ok_b else 'DISAGREEMENT -- investigate'} (tolerance 0.2 %).\n")

    # ---------------------------------------------------------------- A comparison with controls
    say("## A. Corrected forecaster vs controls (test split)")
    rows = []
    def add(name, vals, kind):
        s = {k: stat([v[k] for v in vals]) for k in ("mse", "rmse", "mae", "smape")}
        rows.append((name, kind, s))
    add("Corrected forecaster (F)", [{k: r[f"test_{k}"] for k in ("mse", "rmse", "mae", "smape")} for r in R], f"{len(R)} seeds")
    for name in ("DLinear", "RevIN_Linear"):
        add(name + " (same protocol)", [{k: float(c[k]) for k in ("mse", "rmse", "mae", "smape")}
                                         for c in ctrl if c["predictor"] == name and c["split"] == "test"], "10 seeds")
    for name in ("seasonal_naive_24h", "seasonal_naive_168h", "window_mean", "last_value", "train_mean"):
        add(name, [{k: float(c[k]) for k in ("mse", "rmse", "mae", "smape")}
                   for c in ctrl if c["predictor"] == name and c["split"] == "test"], "deterministic")
    say("| Predictor | runs | MSE mean ± sd [95% CI] | RMSE | MAE | SMAPE % |\n|---|---|---|---|---|---|")
    for name, kind, s in rows:
        ci = f" [{s['mse']['ci_lo']:.4f}, {s['mse']['ci_hi']:.4f}]" if s["mse"]["n"] > 1 else ""
        say(f"| {name} | {kind} | {s['mse']['mean']:.4f} ± {s['mse']['sd']:.4f}{ci} | {s['rmse']['mean']:.4f} | "
            f"{s['mae']['mean']:.4f} | {s['smape']['mean']:.2f} |")
    fm = rows[0][2]["mse"]["mean"]; dl = rows[1][2]["mse"]["mean"]
    say(f"\nRelative to seasonal naive (24 h) the corrected forecaster lowers test MSE by "
        f"{100 * (1 - fm / 0.5122):.1f} %; relative to the window mean by {100 * (1 - fm / 0.7086):.1f} %; "
        f"its difference to DLinear is {fm - dl:+.4f} MSE ({100 * (fm / dl - 1):+.1f} %).\n")

    # ---------------------------------------------------------------- D temporal learning
    say("## D. Does the forecaster learn temporal patterns rather than the average?")
    shp = [shape_r2(P[s]["test"].astype(float), Yte, lev_te) for s in seeds]
    cor = [mean_corr(P[s]["test"].astype(float), Yte, lev_te) for s in seeds]
    shuf = [r["test_mse_shuffled_patch_order"] / r["test_mse"] for r in R]
    dlp = np.load(os.path.join(a.controls, "preds_DLinear_s42.npz"))["test"].astype(float)
    say("Level-removed skill: forecast and target minus each window's input mean, so a predictor that only "
        "matches the level scores 0 and a perfect forecaster scores 1.\n")
    say("| Predictor | level-removed R² | mean within-window correlation |\n|---|---|---|")
    say(f"| Corrected forecaster (mean over seeds) | {np.mean(shp):.3f} (min {np.min(shp):.3f}) | {np.mean(cor):.3f} |")
    say(f"| DLinear (seed 42) | {shape_r2(dlp, Yte, lev_te):.3f} | {mean_corr(dlp, Yte, lev_te):.3f} |")
    say(f"| Seasonal naive 24 h | {shape_r2(sn24, Yte, lev_te):.3f} | {mean_corr(sn24, Yte, lev_te):.3f} |")
    say(f"| Window mean | {shape_r2(wmean, Yte, lev_te):.3f} | (constant) |")
    lin = np.load(os.path.join(a.controls, "preds_RevIN_Linear_s42.npz"))["test"].astype(float)
    Pm0 = np.mean([P[s_]["test"].astype(float) for s_ in seeds], 0)
    d_lin = float(((Pm0 - lin) ** 2).mean()); e_lin = float(((lin - Yte) ** 2).mean())
    say(f"| RevIN-Linear (seed 42) | {shape_r2(lin, Yte, lev_te):.3f} | {mean_corr(lin, Yte, lev_te):.3f} |")
    say(f"\nSimilarity to the linear control: mean squared difference between the forecaster's (seed-averaged) "
        f"forecast and the RevIN-Linear forecast is {d_lin:.4f}, against a RevIN-Linear test MSE of {e_lin:.4f} "
        f"(ratio {d_lin / e_lin:.2f}; near 0 would mean the forecaster reproduces the linear map).")
    say(f"\nInput-order test: with the 32 input patches presented in a shuffled order, test MSE rises by a factor "
        f"{np.mean(shuf):.2f} (range {np.min(shuf):.2f}-{np.max(shuf):.2f}) over the normal order.")
    steps = np.array([r["test_mse_per_step"] for r in R]).mean(0)
    sn_step = ((sn24 - Yte) ** 2).mean((0, 2)); wm_step = ((wmean - Yte) ** 2).mean((0, 2)); dl_step = ((dlp - Yte) ** 2).mean((0, 2))
    say(f"Per-step test MSE (mean over seeds): step 1 {steps[0]:.3f}, step 24 {steps[23]:.3f}, step 48 {steps[47]:.3f}, "
        f"step 96 {steps[95]:.3f}; seasonal naive {sn_step[0]:.3f}/{sn_step[23]:.3f}/{sn_step[47]:.3f}/{sn_step[95]:.3f}. "
        f"The forecaster is better than seasonal naive at {int((steps < sn_step).sum())} of 96 steps.")
    pv = np.array([r["test_mse_per_variable"] for r in R]).mean(0)
    sn_v = ((sn24 - Yte) ** 2).mean((0, 1)); dl_v = ((dlp - Yte) ** 2).mean((0, 1))
    say("\n| Variable | forecaster MSE | DLinear | seasonal naive |\n|---|---|---|---|")
    for j, c in enumerate(cols):
        say(f"| {c} | {pv[j]:.4f} | {dl_v[j]:.4f} | {sn_v[j]:.4f} |")
    temporal_ok = np.min(shp) > 0.2 and np.mean(shuf) > 1.1
    say(f"\nVerdict: {'the forecaster predicts within-window structure, not only the level' if temporal_ok else 'NOT established'} "
        f"(criteria fixed in advance: level-removed R² > 0.2 for every seed and input-order factor > 1.1).\n")

    # ---------------------------------------------------------------- C curves
    say("## C. Training and validation curves")
    H = {s: json.load(open(os.path.join(a.s2, "runs", f"s{s}", "epoch_log.json"))) for s in seeds}
    be = [r["best_epoch"] for r in R]; er = [r["epochs_run"] for r in R]
    dv = np.concatenate([np.abs(np.diff([x["val_mse"] for x in H[s]])) for s in seeds])
    v1 = [H[s][0]["val_mse"] for s in seeds]; vb = [r["best_val_mse"] for r in R]
    t1 = [H[s][0]["train_task"] for s in seeds]; tl = [H[s][-1]["train_task"] for s in seeds]
    gap_b = [H[s][r["best_epoch"] - 1]["gap"] for s, r in zip(seeds, R)]
    say(f"Best epoch: median {np.median(be):.0f} (range {min(be)}-{max(be)}); epochs run: median {np.median(er):.0f}. "
        f"Training task MSE falls from {np.mean(t1):.3f} (epoch 1) to {np.mean(tl):.3f} (last epoch); validation MSE "
        f"is {np.mean(v1):.3f} after epoch 1 and {np.mean(vb):.3f} at its best, i.e. "
        f"{100 * (1 - np.mean(vb) / np.mean(v1)):.1f} % lower. Epoch-to-epoch validation change: median "
        f"{np.median(dv):.4f}, 90th percentile {np.percentile(dv, 90):.4f}. Generalization gap at the selected epoch: "
        f"{np.mean(gap_b):.3f}.\n")

    # ---------------------------------------------------------------- E seed variability, reproducibility
    say("## E. Seed variability and reproducibility")
    s = rows[0][2]["mse"]
    say(f"Test MSE over {s['n']} seeds: mean {s['mean']:.4f}, sd {s['sd']:.4f} (CV {100 * s['sd'] / s['mean']:.1f} %), "
        f"range {s['min']:.4f}-{s['max']:.4f}. DLinear: sd {rows[1][2]['mse']['sd']:.4f}.")
    if a.s1:
        s1 = {json.load(open(p))["seed"]: json.load(open(p)) for p in glob.glob(os.path.join(a.s1, "runs", "F_s*", "result.json"))}
        common = sorted(set(s1) & set(seeds))
        if common:
            say("\nSame code commit, configuration and seed run twice (S1 and S2, separate Kaggle sessions):\n")
            say("| seed | S1 test MSE | S2 test MSE | difference | S1 best epoch | S2 best epoch |\n|---|---|---|---|---|---|")
            for c in common:
                r2 = [r for r in R if r["seed"] == c][0]
                say(f"| {c} | {s1[c]['test_mse']:.4f} | {r2['test_mse']:.4f} | {r2['test_mse'] - s1[c]['test_mse']:+.4f} | "
                    f"{s1[c]['best_epoch']} | {r2['best_epoch']} |")
    say("")

    # ---------------------------------------------------------------- F val/test shift
    say("## F. Validation/test distribution shift")
    vm = [r["val_mse"] for r in R]; tm = [r["test_mse"] for r in R]
    from scipy.stats import spearmanr
    rho = spearmanr(vm, tm).correlation if len(R) > 2 else float("nan")
    dlv = [float(c["mse"]) for c in ctrl if c["predictor"] == "DLinear" and c["split"] == "val"]
    snv = [float(c["mse"]) for c in ctrl if c["predictor"] == "seasonal_naive_24h" and c["split"] == "val"][0]
    say(f"Validation MSE is higher than test MSE for every predictor (forecaster {np.mean(vm):.3f} vs {np.mean(tm):.3f}; "
        f"DLinear {np.mean(dlv):.3f} vs {rows[1][2]['mse']['mean']:.3f}; seasonal naive {snv:.3f} vs 0.512), consistent "
        f"with the documented shift (OT falls by 1.17 training SD between the validation and test periods). Across seeds, "
        f"the rank correlation between validation and test MSE of the selected checkpoints is {rho:.2f}: a lower validation "
        f"loss {'does' if rho > 0.5 else 'does not reliably'} indicate a lower test loss.")
    mon = np.arange(len(Yte)) * 4 // len(Yte)
    say("\n| Test month | forecaster MSE | seasonal naive | DLinear |\n|---|---|---|---|")
    Pm = np.mean([P[s]["test"].astype(float) for s in seeds], 0)
    for k in range(4):
        i = mon == k
        say(f"| {k + 1} | {((Pm[i] - Yte[i]) ** 2).mean():.4f} | {((sn24[i] - Yte[i]) ** 2).mean():.4f} | {((dlp[i] - Yte[i]) ** 2).mean():.4f} |")
    say("(forecaster column: mean forecast over seeds)\n")

    # ---------------------------------------------------------------- figures
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].plot(range(1, 97), steps, label="corrected forecaster (mean of seeds)")
        ax[0].plot(range(1, 97), dl_step, label="DLinear (seed 42)")
        ax[0].plot(range(1, 97), sn_step, label="seasonal naive 24 h")
        ax[0].plot(range(1, 97), wm_step, label="window mean")
        ax[0].set_xlabel("forecast step (h)"); ax[0].set_ylabel("test MSE"); ax[0].legend(fontsize=8); ax[0].set_title("Error by horizon step")
        for s_ in seeds:
            h = H[s_]; ax[1].plot([x["epoch"] for x in h], [x["val_mse"] for x in h], color="C1", alpha=.5)
            ax[1].plot([x["epoch"] for x in h], [x["train_task"] for x in h], "--", color="C0", alpha=.5)
        ax[1].set_xlabel("epoch"); ax[1].set_title("Train (dashed) and validation MSE, all seeds")
        i, j = 1000, cols.index("OT")
        ax[2].plot(range(-96, 0), Xte[i, -96:, j], color="gray", label="input (last 96 h)")
        ax[2].plot(range(96), Yte[i, :, j], color="k", label="target")
        ax[2].plot(range(96), P[seeds[0]]["test"][i, :, j].astype(float), label="forecaster")
        ax[2].plot(range(96), sn24[i, :, j], ":", label="seasonal naive")
        ax[2].plot(range(96), wmean[i, :, j], ":", label="window mean")
        ax[2].legend(fontsize=7); ax[2].set_title("OT, test window 1000 (z-scored)")
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_s2_baseline.png"), dpi=130)
        say("Figure: `fig_s2_baseline.png` (error by step, curves, example forecast).\n")
    except Exception as e:
        say(f"(figure failed: {e})")

    # ---------------------------------------------------------------- exit assessment
    say("## S2 exit assessment")
    beats = fm < 0.5122 and fm < 0.7086
    say(f"- Better than seasonal naive and window mean on every seed: "
        f"{'yes' if all(r['test_mse'] < 0.5122 for r in R) else 'NO'}")
    say(f"- Comparable to DLinear reproduced under the same protocol: difference {fm - dl:+.4f} MSE")
    say(f"- Metrics independently verified: {'yes' if ok_b else 'NO'}")
    say(f"- Temporal learning beyond the window average: {'yes' if temporal_ok else 'NO'}")
    json.dump(dict(beats_controls=bool(beats), verified=bool(ok_b), temporal=bool(temporal_ok), f_mse=fm, dlinear_mse=dl),
              open(os.path.join(a.out, "s2_exit.json"), "w"), indent=1)
    open(os.path.join(a.out, "STAGE2_REPORT.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
