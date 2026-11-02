"""
Stage 3 analysis (offline): does the TFB objective help? Implements configs/stage3_protocol.json exactly.

  python s3/stage3_analysis.py --s3 <stage3 output dir> --s2 <stage2 output dir> --out s3/report
"""
import argparse, glob, json, os
import numpy as np
from scipy.stats import wilcoxon

RNG = np.random.default_rng(0)


def paired_stats(d):
    """d: paired differences (treatment - control), one per seed."""
    d = np.asarray(d, float)
    w = wilcoxon(d, alternative="two-sided", zero_method="wilcox") if np.any(d != 0) else None
    signs = RNG.choice([-1, 1], size=(10000, len(d)))
    perm_p = float((np.abs((signs * d).mean(1)) >= abs(d.mean()) - 1e-15).mean())
    boot = RNG.choice(d, (10000, len(d)), replace=True).mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(mean=float(d.mean()), sd=float(d.std(ddof=1)), wilcoxon_p=float(w.pvalue) if w else 1.0,
                perm_p=perm_p, ci_lo=float(lo), ci_hi=float(hi), n_better=int((d < 0).sum()), n=len(d))


def cliffs_delta(x, y):
    """P(x < y) - P(x > y): negative when x tends to be lower (better) than y."""
    x, y = np.asarray(x), np.asarray(y)
    return float(((x[:, None] < y[None]).sum() - (x[:, None] > y[None]).sum()) / (len(x) * len(y)) * -1)


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * ps[i])); adj[i] = run
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s3", required=True)
    ap.add_argument("--s2", required=True)
    ap.add_argument("--out", default="s3/report")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    L = []
    say = lambda s="": (print(s), L.append(s))

    runs = {}
    for p in glob.glob(os.path.join(a.s2, "runs", "s*", "result.json")):
        r = json.load(open(p)); r["alpha"] = 0.0; r["dir"] = os.path.dirname(p); runs[(0.0, r["seed"])] = r
    for p in glob.glob(os.path.join(a.s3, "runs", "a*_s*", "result.json")):
        r = json.load(open(p)); r["dir"] = os.path.dirname(p); runs[(float(r["alpha"]), r["seed"])] = r
    alphas = sorted({k[0] for k in runs})
    seeds = sorted({k[1] for k in runs if k[0] == 0.0})
    complete = [s for s in seeds if all((al, s) in runs for al in alphas)]
    say("# Stage 3 report: does the auxiliary TFB objective help? (H1)\n")
    say(f"Protocol: `configs/stage3_protocol.json`, declared before any run. Frozen forecaster F; the only thing that "
        f"changes between arms is the fixed weight alpha of the TFB loss. Arms: alpha in {alphas}; seeds with every arm "
        f"complete: {len(complete)} of {len(seeds)} ({complete}). Only complete paired sets are analysed.\n")
    if len(complete) < len(seeds):
        say(f"**Note:** {len(seeds) - len(complete)} seed(s) incomplete; results below use the complete seeds only.\n")

    # ---------------------------------------------------------------- arm table
    say("## Results by arm (mean ± sd over seeds)\n")
    say("| alpha | best val MSE | test MSE | test RMSE | test MAE | test SMAPE % | best epoch | final train TFB loss |\n|---|---|---|---|---|---|---|---|")
    arm = {}
    for al in alphas:
        rs = [runs[(al, s)] for s in complete]
        tfb = []
        for r in rs:
            h = json.load(open(os.path.join(r["dir"], "epoch_log.json")))
            tfb.append(h[r["best_epoch"] - 1]["train_tfb"])
        g = lambda k: np.array([r[k] for r in rs], float)
        arm[al] = dict(val=g("best_val_mse"), mse=g("test_mse"), rmse=g("test_rmse"), mae=g("test_mae"),
                       smape=g("test_smape"), be=g("best_epoch"), tfb=np.array(tfb))
        A = arm[al]
        say(f"| {al} | {A['val'].mean():.4f} ± {A['val'].std(ddof=1):.4f} | {A['mse'].mean():.4f} ± {A['mse'].std(ddof=1):.4f} | "
            f"{A['rmse'].mean():.4f} | {A['mae'].mean():.4f} | {A['smape'].mean():.2f} | {A['be'].mean():.1f} | {A['tfb'].mean():.3f} |")
    say("\n(final train TFB loss: the epoch-mean TFB loss at the selected epoch; at alpha = 0 it is computed but not optimised, "
        "so the difference shows whether training actually reduces the auxiliary objective.)\n")

    # ---------------------------------------------------------------- alpha* (validation only)
    val_means = {al: arm[al]["val"].mean() for al in alphas}
    a_star = min(val_means, key=val_means.get)
    say("## Validation-selected alpha*\n")
    say("Rule (declared): the arm with the lowest mean best-validation MSE over seeds, alpha = 0 included. Test results are not used.\n")
    say("| alpha | mean best-validation MSE |\n|---|---|")
    for al in alphas:
        say(f"| {al} | {val_means[al]:.4f}{'  ← alpha*' if al == a_star else ''} |")
    say(f"\n**alpha\\* = {a_star}**\n")

    # ---------------------------------------------------------------- primary test
    say("## Primary test: alpha* vs alpha = 0 (paired by seed, test RMSE)\n")
    if a_star == 0.0:
        prim = None
        say("alpha* = 0: validation does not favour any positive TFB weight. **H1 is not supported** by the declared "
            "decision rule (it requires alpha* > 0).\n")
    else:
        d = arm[a_star]["rmse"] - arm[0.0]["rmse"]
        prim = paired_stats(d); prim["cliffs_delta"] = cliffs_delta(arm[a_star]["rmse"], arm[0.0]["rmse"])
        dm = arm[a_star]["mse"] - arm[0.0]["mse"]
        say(f"Mean paired difference (alpha* − 0) in test RMSE: {prim['mean']:+.5f} "
            f"(95 % bootstrap CI [{prim['ci_lo']:+.5f}, {prim['ci_hi']:+.5f}]); in test MSE {dm.mean():+.5f}. "
            f"alpha* better on {prim['n_better']} of {prim['n']} seeds. Wilcoxon p = {prim['wilcoxon_p']:.4f}; paired "
            f"permutation p = {prim['perm_p']:.4f}; Cliff's delta = {prim['cliffs_delta']:+.2f}.\n")

    # ---------------------------------------------------------------- secondary: all arms vs 0 (Holm)
    say("## Secondary: every fixed alpha vs alpha = 0 (paired, test RMSE, Holm-corrected)\n")
    sec = {al: paired_stats(arm[al]["rmse"] - arm[0.0]["rmse"]) for al in alphas if al != 0.0}
    adj = holm(np.array([sec[al]["wilcoxon_p"] for al in sec]))
    say("| alpha | mean diff RMSE | 95% CI | better on | Wilcoxon p | Holm p | permutation p |\n|---|---|---|---|---|---|---|")
    for (al, s), ph in zip(sec.items(), adj):
        s["holm_p"] = float(ph)
        say(f"| {al} | {s['mean']:+.5f} | [{s['ci_lo']:+.5f}, {s['ci_hi']:+.5f}] | {s['n_better']}/{s['n']} | "
            f"{s['wilcoxon_p']:.4f} | {ph:.4f} | {s['perm_p']:.4f} |")

    # ---------------------------------------------------------------- decision
    sd0 = arm[0.0]["mse"].std(ddof=1)
    say("\n## Decision (rule declared in configs/stage3_protocol.json)\n")
    if prim is None:
        h1, meaningful = False, False
    else:
        h1 = prim["mean"] < 0 and prim["wilcoxon_p"] < 0.05 and prim["ci_hi"] < 0
        meaningful = h1 and (-(arm[a_star]["mse"] - arm[0.0]["mse"]).mean() > sd0)
    say(f"- H1 (TFB improves test error over alpha = 0): **{'SUPPORTED' if h1 else 'NOT SUPPORTED'}**")
    say(f"- Practically meaningful (improvement > seed sd {sd0:.4f} MSE): **{'yes' if meaningful else 'no'}**")
    say(f"- Calibrated fixed alpha for S4: **alpha* = {a_star}**")
    if not h1:
        say("- Per the research plan: the scheduler study (S4) is **not started**; the central premise must be reassessed "
            "with the supervisor (see the options in the Stage 3 summary).")
    else:
        say("- Per the research plan: proceed to S4 with alpha* as the critical comparator.")

    # ---------------------------------------------------------------- figure
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        xs = [max(al, 0.003) for al in alphas]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for k, (key, lab) in enumerate((("val", "best validation MSE"), ("mse", "test MSE"))):
            m = [arm[al][key].mean() for al in alphas]; s = [arm[al][key].std(ddof=1) for al in alphas]
            ax[k].errorbar(xs, m, yerr=s, marker="o", capsize=3)
            for al, x in zip(alphas, xs):
                ax[k].scatter([x] * len(complete), arm[al][key], s=8, color="gray", alpha=.5)
            ax[k].set_xscale("log"); ax[k].set_xlabel("alpha (alpha = 0 plotted at 0.003)"); ax[k].set_title(lab)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_s3_alpha.png"), dpi=130)
        say("\nFigure: `fig_s3_alpha.png` (validation and test MSE against alpha; dots = seeds, bars = sd).")
    except Exception as e:
        say(f"(figure failed: {e})")

    json.dump(dict(alpha_star=a_star, h1_supported=bool(h1), meaningful=bool(meaningful), primary=prim, secondary=sec,
                   val_means=val_means, complete_seeds=complete), open(os.path.join(a.out, "s3_decision.json"), "w"), indent=1)
    open(os.path.join(a.out, "STAGE3_REPORT.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
