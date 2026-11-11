"""
Stage X (RAFT) declared analysis.   python analysis/stage_x_analysis.py --pilot <dir> [--out <dir>]
Applies the rules of configs/stage_x_protocol.json exactly; test results never select anything.
(Parts core / ablation / breadth / fewshot are analysed by the same script once they exist; options added then.)
"""
import argparse, glob, json, os, statistics

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(d):
    return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(d, "runs", "*", "result.json")))]


def pilot(rows, P, out):
    pp = P["parts"]["pilot"]
    cands = ["raft_a0.1", "raft_a0.3", "raft_a1", "raft_rt_a0.3"]
    L = ["# Stage X pilot (X0): selection of the RAFT configuration", "",
         "Rule (declared): " + pp["selection_rule"], "", "Go rule (declared): " + pp["go_rule"], ""]
    by = {}
    for r in rows:
        by.setdefault((r["setting"], r["arm"]), {})[r["seed"]] = r
    seeds = pp["seeds"]
    complete = all(len(by.get((s, a), {})) == len(seeds) for s in pp["settings"] for a in pp["arms"])
    L.append(f"Complete: {complete} ({len(rows)} runs)\n")
    L.append("| setting | arm | val MSE (mean) | r = val / val(none) | test MSE | test RMSE | SMAPE % | conflict rate |")
    L.append("|---|---|---|---|---|---|---|---|")
    ratio = {}
    present = [s for s in pp["settings"] if by.get((s, "none"))]
    for s in present:
        v0 = statistics.mean(x["best_val_mse"] for x in by[(s, "none")].values())
        for a in pp["arms"]:
            rs = [by[(s, a)][k] for k in seeds if k in by.get((s, a), {})]
            if not rs:
                continue
            v = statistics.mean(x["best_val_mse"] for x in rs)
            ratio[(s, a)] = v / v0
            cr = statistics.mean(x["proj_conflict_rate"] for x in rs) if "proj_conflict_rate" in rs[0] else float("nan")
            L.append(f"| {s} | {a} | {v:.4f} | {v / v0:.4f} | {statistics.mean(x['test_mse'] for x in rs):.4f} | "
                     f"{statistics.mean(x['test_rmse'] for x in rs):.4f} | {statistics.mean(x['test_smape'] for x in rs):.2f} | {cr:.2f} |")
    alpha = lambda a: float(a.rsplit("_a", 1)[1])
    score = {a: statistics.mean(ratio[(s, a)] for s in present) for a in cands if present and all((s, a) in ratio for s in present)}
    if not score:
        print("\n".join(L)); print("no complete candidate yet"); return None, False
    best = min(score.values())
    tied = [a for a in score if score[a] - best < 1e-4]
    sel = min(tied, key=alpha)
    go = complete and all(ratio[(s, sel)] <= 1.005 for s in pp["settings"])
    L += ["", "Mean ratio over settings: " + ", ".join(f"{a} {score[a]:.4f}" for a in cands if a in score),
          f"\n**Selected RAFT arm: {sel}**  ->  **{'GO' if go else 'NO-GO (carry forward; C2 not expected)'}**"]
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "STAGE_X_PILOT.md"), "w").write("\n".join(L) + "\n")
    json.dump(dict(selected=sel, go=go, score=score, ratio={f"{k[0]}|{k[1]}": v for k, v in ratio.items()}, complete=complete),
              open(os.path.join(out, "x_pilot_decision.json"), "w"), indent=1)
    print("\n".join(L))
    return sel, go


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    P = json.load(open(os.path.join(REPO, "configs", "stage_x_protocol.json")))
    pilot(load(a.pilot), P, a.out or os.path.join(a.pilot, "analysis"))


if __name__ == "__main__":
    main()
