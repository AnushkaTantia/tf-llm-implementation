"""
Forecast-vs-actual plots + probabilistic (interval) forecasting.
Inputs: preds_test.npy, trues_test.npy, preds_train.npy, trues_train.npy  [n_windows, 96, 7]
Usage: python make_forecast_figures.py --dir results/preds_hmm --out results/figs_hmm
"""
import argparse, os, json
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

LEVELS = [0.90, 0.80, 0.70, 0.60]


def load(d, name):
    a = np.load(os.path.join(d, name))
    if a.ndim == 2:
        a = a[..., None]
    return a


def series(a, target, step):
    return a[:, step, target]


def pi_metrics(y, lo, hi, level, R):
    cov = (y >= lo) & (y <= hi)
    picp = cov.mean()
    pinaw = np.mean(hi - lo) / R
    w = hi - lo
    awd = np.where(y < lo, (lo - y) / w, np.where(y > hi, (y - hi) / w, 0.0)).mean()
    ace = picp - level
    return picp, pinaw, awd, ace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default="figs")
    ap.add_argument("--target", type=int, default=-1)
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--label", default="HMM scheduler (best run)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    Pt, Tt = load(a.dir, "preds_test.npy"), load(a.dir, "trues_test.npy")
    Pr, Tr = load(a.dir, "preds_train.npy"), load(a.dir, "trues_train.npy")
    mse, mae = np.mean((Pt - Tt) ** 2), np.mean(np.abs(Pt - Tt))
    print(f"Test MSE={mse:.4f}  MAE={mae:.4f}  (close to the log; Cell 4 of the lab notebook prints the exact value)")

    yt, pt = series(Tt, a.target, a.step), series(Pt, a.target, a.step)

    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.plot(yt, lw=0.8, color="black", label="True (observed)")
    ax.plot(pt, lw=0.8, color="tab:red", alpha=0.85, label=f"Forecast: {a.label}")
    ax.set_xlabel("Test-set time index (hours)"); ax.set_ylabel("OT (z-scored)")
    ax.legend(fontsize=7, frameon=False); fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig_forecast_line.pdf")); plt.close(fig)

    g = sns.jointplot(x=yt, y=pt, kind="reg", height=4,
                      scatter_kws=dict(s=4, alpha=0.4), line_kws=dict(color="tab:red", lw=1))
    lim = [min(yt.min(), pt.min()), max(yt.max(), pt.max())]
    g.ax_joint.plot(lim, lim, "k--", lw=0.7)
    r = np.corrcoef(yt, pt)[0, 1]
    g.ax_joint.set_xlabel("True OT (z-scored)"); g.ax_joint.set_ylabel("Forecast OT (z-scored)")
    g.ax_joint.text(0.03, 0.95, f"r = {r:.3f}", transform=g.ax_joint.transAxes, va="top", fontsize=8)
    g.savefig(os.path.join(a.out, "fig_forecast_joint.pdf")); plt.close("all")

    e = series(Tr, a.target, a.step) - series(Pr, a.target, a.step)
    g_loc, g_scale = stats.norm.fit(e)
    t_df, t_loc, t_scale = stats.t.fit(e)
    fig, ax = plt.subplots(figsize=(4, 2.8))
    ax.hist(e, bins=60, density=True, color="lightgray", edgecolor="gray", lw=0.3, label="Train residuals")
    xs = np.linspace(e.min(), e.max(), 400)
    ax.plot(xs, stats.norm.pdf(xs, g_loc, g_scale), label="Gaussian")
    ax.plot(xs, stats.t.pdf(xs, t_df, t_loc, t_scale), label="t Location-Scale")
    ax.set_xlabel("Residual"); ax.legend(fontsize=7, frameon=False); fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig_residual_fit.pdf")); plt.close(fig)

    dists = {
        "Gaussian": lambda q: stats.norm.ppf(q, g_loc, g_scale),
        "t Location-Scale": lambda q: stats.t.ppf(q, t_df, t_loc, t_scale),
    }
    R = yt.max() - yt.min()
    rows = []
    fig, axes = plt.subplots(2, 1, figsize=(7, 4.4), sharex=True)
    shades = dict(zip(LEVELS, [0.15, 0.25, 0.35, 0.45]))
    for ax, (name, ppf) in zip(axes, dists.items()):
        for lv in LEVELS:
            lo, hi = pt + ppf((1 - lv) / 2), pt + ppf(1 - (1 - lv) / 2)
            rows.append((name, lv, *pi_metrics(yt, lo, hi, lv, R)))
            ax.fill_between(np.arange(len(pt)), lo, hi, color="tab:blue", alpha=shades[lv], lw=0,
                            label=f"{int(lv*100)}% PI")
        ax.plot(yt, color="black", lw=0.6, label="True")
        ax.plot(pt, color="tab:red", lw=0.6, label="Forecast")
        ax.set_title(name, fontsize=8); ax.set_ylabel("OT (z)")
    axes[0].legend(fontsize=6, ncol=6, frameon=False, loc="upper left")
    axes[-1].set_xlabel("Test-set time index (hours)"); fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig_prediction_intervals.pdf")); plt.close(fig)

    with open(os.path.join(a.out, "table_pi_metrics.csv"), "w") as f:
        f.write("distribution,confidence,PICP,PINAW,AWD,ACE\n")
        for r_ in rows:
            f.write("%s,%d,%.4f,%.4f,%.4f,%.4f\n" % (r_[0], int(r_[1] * 100), *r_[2:]))
    tex = ["\\begin{tabular}{llcccc}", "\\toprule",
           "\\textbf{Distribution} & \\textbf{Conf.} & \\textbf{PICP}$\\uparrow$ & \\textbf{PINAW}$\\downarrow$ & \\textbf{AWD}$\\downarrow$ & \\textbf{ACE}$\\uparrow$ \\\\",
           "\\midrule"]
    for r_ in rows:
        tex.append("%s & %d\\%% & %.4f & %.4f & %.4f & %+.4f \\\\" % (r_[0], int(r_[1] * 100), *r_[2:]))
    tex += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(a.out, "table_pi_metrics.tex"), "w").write("\n".join(tex))

    json.dump(dict(test_mse=float(mse), test_mae=float(mae), corr_true_forecast=float(r),
                   gaussian=dict(loc=float(g_loc), scale=float(g_scale)),
                   t=dict(df=float(t_df), loc=float(t_loc), scale=float(t_scale))),
              open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    for r_ in rows:
        print("%-17s %d%%  PICP=%.3f PINAW=%.3f AWD=%.3f ACE=%+.3f" % (r_[0], int(r_[1] * 100), *r_[2:]))


if __name__ == "__main__":
    main()
