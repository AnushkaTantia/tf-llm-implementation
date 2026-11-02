"""
Stage R1 analysis (offline): applies the decision rules of configs/stage_r_protocol.json exactly.

  python analysis/stage_r_analysis.py [--mechanism <dir>] [--scope_f <dir>] [--scope_e <dir>]
                                      [--s2 results/stage2] [--s3 results/stage3] --out <report dir>

Each part is optional, so the report can be produced after each Kaggle session. Only complete paired sets
(every arm of a setting present for a seed) are analysed.
"""
import argparse, glob, json, os
import numpy as np
from scipy.stats import wilcoxon

RNG = np.random.default_rng(0)


def paired(d):
    d = np.asarray(d, float)
    w = wilcoxon(d, alternative="two-sided", zero_method="wilcox") if np.any(d != 0) else None
    boot = RNG.choice(d, (10000, len(d)), replace=True).mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(mean=float(d.mean()), ci_lo=float(lo), ci_hi=float(hi), wilcoxon_p=float(w.pvalue) if w else 1.0,
                n_pos=int((d > 0).sum()), n_neg=int((d < 0).sum()), n=len(d))


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * ps[i])); adj[i] = run
    return adj


def load(d):
    R = {}
    for p in glob.glob(os.path.join(d, "runs", "*", "result.json")):
        r = json.load(open(p)); R[(r["setting"], r["arm"], r["seed"])] = r
    return R


def complete(R, setting, arms):
    seeds = sorted({k[2] for k in R if k[0] == setting})
    return [s for s in seeds if all((setting, a, s) in R for a in arms)]


def ms(x):
    return f"{np.mean(x):.4f} ± {np.std(x, ddof=1):.4f}" if len(x) > 1 else f"{np.mean(x):.4f}"


def mechanism(R, P, say, out, s2=None, s3=None):
    arms = list(P["arms"]); st = "ETTh1_H96"
    seeds = complete(R, st, arms)
    say(f"## Part mechanism (F, ETTh1, H = 96): why does TFB hurt?\n\nComplete paired seeds: {len(seeds)} {seeds}\n")
    if not seeds:
        return {}
    g = lambda a, k: np.array([R[(st, a, s)][k] for s in seeds], float)
    say("| arm | val MSE | test MSE | test RMSE | SMAPE % | ΔRMSE vs α=0 [95% CI] | seeds worse | R_zT | R_hTD | R_zF | phase R² (z_T) | phase R² (h_last) |")
    say("|---|---|---|---|---|---|---|---|---|---|---|---|")
    D = {}
    for a in arms:
        d = paired(g(a, "test_rmse") - g("a0", "test_rmse")) if a != "a0" else None
        D[a] = dict(delta=d, R_zT=g(a, "R_z_T").tolist(), R_zF=g(a, "R_z_F").tolist(), R_hTD=g(a, "R_h_TD").tolist(),
                    phase_zT=g(a, "phaseR2_z_T").tolist(), phase_last=g(a, "phaseR2_h_last").tolist(),
                    test_rmse=g(a, "test_rmse").tolist(), val=g(a, "best_val_mse").tolist())
        dtxt = f"{d['mean']:+.4f} [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}] | {d['n_pos']}/{d['n']}" if d else "– | –"
        say(f"| {a} | {ms(g(a,'best_val_mse'))} | {ms(g(a,'test_mse'))} | {np.mean(g(a,'test_rmse')):.4f} | "
            f"{np.mean(g(a,'test_smape')):.2f} | {dtxt} | {np.mean(g(a,'R_z_T')):.3f} | {np.mean(g(a,'R_h_TD')):.3f} | "
            f"{np.max(g(a,'R_z_F')):.1e} | {np.mean(g(a,'phaseR2_z_T')):.3f} | {np.mean(g(a,'phaseR2_h_last')):.3f} |")
    say("\n(R_e: relative circular-shift sensitivity, mean over tau in {1, 4, 12} h; 0 = the embedding ignores when "
        "events happen in the window. R_zF column: maximum over seeds. Phase R²: linear decodability of the hour of "
        "day of the first forecast step, validation windows.)\n")
    lower = lambda a, k: int(sum(x < y for x, y in zip(D[a][k], D["a0"][k])))
    n = len(seeds); need = n - 1 if n >= 5 else n
    Ma = all(R[(st, a, s)]["R_z_F"] < 0.01 for a in arms for s in seeds)
    Mb = lower("full_a0.1", "R_zT") >= need
    Mc = lower("full_a0.1", "phase_zT") >= need
    Md = D["B_a0.1"]["delta"]["n_pos"] >= need and lower("B_a0.1", "R_zT") >= need
    T_role = D["T_a0.1"]["delta"]["n_pos"] >= need and lower("T_a0.1", "R_zT") >= need
    if Ma and Mb and Mc and Md:
        verdict = "SUPPORTED: TFB removes timing information from the time-domain path, and the balance (alignment) term alone reproduces both the harm and the loss of timing information"
    elif Ma and Mb and Mc and T_role and not Md:
        verdict = "NOT SUPPORTED as alignment; the loss of timing information and the harm are attributed to the time-domain contrastive invariance (L_T)"
    else:
        verdict = "NOT SUPPORTED: the mechanism remains open"
    say("### Decision rules (declared)\n")
    say(f"- M_a premise (R_zF < 0.01 in every run): **{Ma}**")
    say(f"- M_b (R_zT lower at full α=0.1 than α=0 on ≥ {need}/{n} seeds): **{Mb}** ({lower('full_a0.1','R_zT')}/{n})")
    say(f"- M_c (phase R² from z_T lower at full α=0.1 on ≥ {need}/{n} seeds): **{Mc}** ({lower('full_a0.1','phase_zT')}/{n})")
    say(f"- M_d (balance term alone: higher test RMSE on ≥ {need}/{n} and lower R_zT on ≥ {need}/{n}): **{Md}** "
        f"(RMSE higher {D['B_a0.1']['delta']['n_pos']}/{n}; R_zT lower {lower('B_a0.1','R_zT')}/{n})")
    say(f"- time-domain contrastive term alone: RMSE higher {D['T_a0.1']['delta']['n_pos']}/{n}; R_zT lower {lower('T_a0.1','R_zT')}/{n}")
    say(f"\n**Timing-loss hypothesis: {verdict}.**\n")
    say("Component attribution (descriptive; 5 seeds give no significance test): paired ΔRMSE to α=0 above. "
        "The three component arms decompose the full α=0.1 objective term by term.\n")
    G = {}
    for when in ("selected", "init"):
        ks = sorted(k for k in R[(st, "a0", seeds[0])] if k.startswith("cos_task_") and k.endswith(when))
        if ks:
            G[when] = {k: float(np.mean([R[(st, "a0", s)][k] for s in seeds])) for k in ks}
    if G:
        say("### Gradient alignment (α = 0 runs; cosine between task-loss and component gradients; descriptive)\n")
        say("| pair | at initialisation | at selected checkpoint |\n|---|---|---|")
        for k in sorted({k.rsplit('_', 1)[0] for w in G for k in G[w]}):
            say(f"| {k.replace('cos_task_', 'task vs ')} | {G.get('init', {}).get(k + '_init', float('nan')):+.3f} | "
                f"{G.get('selected', {}).get(k + '_selected', float('nan')):+.3f} |")
        say("")
    if s2 or s3:
        say("### Cross-session check (descriptive): same configuration as S2 (α=0) and S3 (α=0.1)\n")
        for a, src, key in (("a0", s2, "runs/s{s}/result.json"), ("full_a0.1", s3, "runs/a0.1_s{s}/result.json")):
            if not src:
                continue
            diffs, ep = [], 0
            for s in seeds:
                p = os.path.join(src, key.format(s=s))
                if os.path.exists(p):
                    o = json.load(open(p)); diffs.append(abs(o["test_mse"] - R[(st, a, s)]["test_mse"]))
                    ep += o["best_epoch"] == R[(st, a, s)]["best_epoch"]
            if diffs:
                say(f"- {a}: max |test MSE difference| {max(diffs):.2e}; same selected epoch {ep}/{len(diffs)}")
        say("")
    _fig_mech(D, arms, out)
    return dict(seeds=seeds, M_a=Ma, M_b=Mb, M_c=Mc, M_d=Md, T_role=T_role, verdict=verdict, arms=D, grad=G)


def s3_rule(R, setting, alphas, say):
    arms = [f"a{a:g}" for a in alphas]
    seeds = complete(R, setting, arms)
    say(f"### {setting}\n\nComplete paired seeds: {len(seeds)} {seeds}\n")
    if len(seeds) < 2:
        return {}
    g = lambda a, k: np.array([R[(setting, a, s)][k] for s in seeds], float)
    vm = {a: float(np.mean(g(a, "best_val_mse"))) for a in arms}
    star = min(vm, key=vm.get)
    rows, ps = {}, []
    for a in arms[1:]:
        rows[a] = paired(g(a, "test_rmse") - g("a0", "test_rmse")); ps.append(rows[a]["wilcoxon_p"])
    hp = holm(np.array(ps)) if ps else []
    say("| α | best val MSE | test MSE | test RMSE | SMAPE % | ΔRMSE vs α=0 [95% CI] | seeds worse | Wilcoxon p | Holm p |")
    say("|---|---|---|---|---|---|---|---|---|")
    for i, a in enumerate(arms):
        tail = (f"{rows[a]['mean']:+.4f} [{rows[a]['ci_lo']:+.4f}, {rows[a]['ci_hi']:+.4f}] | {rows[a]['n_pos']}/{rows[a]['n']} | "
                f"{rows[a]['wilcoxon_p']:.4f} | {hp[i-1]:.4f}") if a != "a0" else "– | – | – | –"
        say(f"| {a[1:]}{' ←α*' if a == star else ''} | {ms(g(a,'best_val_mse'))} | {ms(g(a,'test_mse'))} | "
            f"{np.mean(g(a,'test_rmse')):.4f} | {np.mean(g(a,'test_smape')):.2f} | {tail} |")
    a_star = float(star[1:])
    h1 = False
    if a_star > 0:
        d = rows[star]; h1 = d["mean"] < 0 and d["wilcoxon_p"] < 0.05 and d["ci_hi"] < 0
    general = a_star == 0 and all(rows[a]["mean"] > 0 for a in rows) and all(p < 0.05 for p in hp)
    say(f"\nα* = {a_star:g}; H1 supported: **{h1}**; failure generalises to this setting: **{general}**\n")
    return dict(seeds=seeds, alpha_star=a_star, h1_supported=bool(h1), failure_generalises=bool(general),
                val_mean=vm, paired={a: dict(rows[a], holm_p=float(hp[i])) for i, a in enumerate(rows)})


def scope_e(R, P, say):
    st = "ETTh1_H96_E"; arms = ["a0", "a0.1"]
    seeds = complete(R, st, arms)
    say(f"### E (GPT-2), ETTh1, H = 96\n\nComplete paired seeds: {len(seeds)} {seeds}\n")
    if len(seeds) < 2:
        return {}
    g = lambda a, k: np.array([R[(st, a, s)][k] for s in seeds], float)
    d = paired(g("a0.1", "test_rmse") - g("a0", "test_rmse"))
    say("| α | best val MSE | test MSE | test RMSE | SMAPE % | R_zT | phase R² (z_T) | minutes/run |\n|---|---|---|---|---|---|---|---|")
    for a in arms:
        say(f"| {a[1:]} | {ms(g(a,'best_val_mse'))} | {ms(g(a,'test_mse'))} | {np.mean(g(a,'test_rmse')):.4f} | "
            f"{np.mean(g(a,'test_smape')):.2f} | {np.mean(g(a,'R_z_T')):.3f} | {np.mean(g(a,'phaseR2_z_T')):.3f} | {np.mean(g(a,'minutes')):.1f} |")
    n = len(seeds); need = n - 1 if n >= 5 else n
    general = d["n_pos"] >= need and d["ci_lo"] > 0
    helps = (np.mean(g("a0.1", "best_val_mse")) < np.mean(g("a0", "best_val_mse"))) and d["n_neg"] >= need and d["ci_hi"] < 0
    verdict = "failure GENERALISES to E" if general else ("TFB HELPS E" if helps else "INCONCLUSIVE")
    say(f"\nPaired ΔRMSE (α=0.1 − α=0): {d['mean']:+.4f} [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}], worse on {d['n_pos']}/{n} "
        f"seeds. **E: {verdict}** (5 seeds: no significance claim).\n")
    return dict(seeds=seeds, paired=d, verdict=verdict)


def _fig_mech(D, arms, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    lab = {"a0": "α=0", "full_a0.1": "full", "T_a0.1": "L_T only", "F_a0.1": "L_F only", "B_a0.1": "L_B only"}
    x = np.arange(len(arms))
    for i, a in enumerate(arms):
        if a != "a0":
            d = D[a]["delta"]; ax[0].bar(i, d["mean"], color="#c44e52" if d["mean"] > 0 else "#4c72b0")
            ax[0].errorbar(i, d["mean"], yerr=[[d["mean"] - d["ci_lo"]], [d["ci_hi"] - d["mean"]]], color="k", capsize=3)
        for j, (k, axx) in enumerate((("R_zT", ax[1]), ("phase_zT", ax[2]))):
            axx.bar(i, np.mean(D[a][k]), color="#8c8c8c" if a == "a0" else "#55a868")
            axx.scatter([i] * len(D[a][k]), D[a][k], s=10, color="k", zorder=3)
    for axx, t in zip(ax, ("Δ test RMSE vs α=0 (95% CI)", "Shift sensitivity of z_T (R_zT)", "Hour-of-day phase R² from z_T")):
        axx.set_xticks(x); axx.set_xticklabels([lab[a] for a in arms], fontsize=8); axx.set_title(t, fontsize=10); axx.grid(alpha=.3, axis="y")
    ax[0].axhline(0, color="k", lw=.8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_r_mechanism.png"), dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mechanism"); ap.add_argument("--scope_f"); ap.add_argument("--scope_e")
    ap.add_argument("--s2"); ap.add_argument("--s3")
    ap.add_argument("--protocol", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "stage_r_protocol.json"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    P = json.load(open(a.protocol))["parts"]
    L = []; say = lambda s="": (print(s), L.append(s))
    say("# Stage R1 report: why does TFB fail, and is the failure general?\n")
    say("Protocol: `configs/stage_r_protocol.json`, declared before any R1 run. Decision rules applied exactly; "
        "test results never select anything.\n")
    dec = {}
    if a.mechanism:
        dec["mechanism"] = mechanism(load(a.mechanism), P["mechanism"], say, a.out, a.s2, a.s3)
    if a.scope_f:
        R = load(a.scope_f)
        say("## Part scope_f (F): is the failure general across a second dataset and a longer horizon?\n")
        dec["scope_f"] = {name: s3_rule(R, name, st["alphas"], say) for name, st in P["scope_f"]["settings"].items()}
    if a.scope_e:
        say("## Part scope_e: the original GPT-2 setting\n")
        dec["scope_e"] = scope_e(load(a.scope_e), P["scope_e"], say)
    json.dump(dec, open(os.path.join(a.out, "r_decision.json"), "w"), indent=1, default=float)
    open(os.path.join(a.out, "STAGE_R_REPORT.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
