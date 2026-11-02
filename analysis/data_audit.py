"""Independent audit of the ETT data pipeline (mirrors src/data/loaders.py exactly)."""
import sys, json, numpy as np, pandas as pd
from scipy import stats
import os
D = os.environ.get("ETT_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
SEQ, PRED = 512, 96
def borders(name):
    m = 4 if name.startswith("ETTm") else 1                     # ETTm*: 15-min data -> 4x rows
    tr, va, te = 12*30*24*m, 4*30*24*m, 4*30*24*m
    b1 = [0, tr - SEQ, tr + va - SEQ]; b2 = [tr, tr + va, tr + va + te]
    return b1, b2, (tr, va, te)
def load(name):
    df = pd.read_csv(f"{D}/{name}.csv"); cols = [c for c in df.columns if c != "date"]
    raw = df[cols].values.astype(np.float64)
    b1, b2, (tr, va, te) = borders(name)
    mu, sd = raw[:tr].mean(0), raw[:tr].std(0)
    z = (raw - mu) / sd
    return df, cols, raw, z, mu, sd, b1, b2
def windows(z, b1, b2, k, pred=PRED):
    seg = z[b1[k]:b2[k]]; n = len(seg) - SEQ - pred + 1
    return seg, n
def baselines(name, pred=PRED, split=2):
    df, cols, raw, z, mu, sd, b1, b2 = load(name)
    seg, n = windows(z, b1, b2, split, pred)
    acc = {k: dict(se=0.0, ae=0.0, sm=0.0, cnt=0) for k in ["mean", "last", "seasonal24", "seasonal168"]}
    for i in range(n):
        x = seg[i:i+SEQ]; y = seg[i+SEQ:i+SEQ+pred]
        preds = {"mean": np.zeros_like(y), "last": np.repeat(x[-1:], pred, 0),
                 "seasonal24": np.array([x[SEQ-24 + (t % 24)] for t in range(pred)]),
                 "seasonal168": np.array([x[SEQ-168 + (t % 168)] for t in range(pred)])}
        yo = y*sd + mu
        for k, p in preds.items():
            d = p - y; po = p*sd + mu
            den = np.abs(yo) + np.abs(po); num = 2*np.abs(po - yo)
            a = acc[k]; a["se"] += (d*d).sum(); a["ae"] += np.abs(d).sum(); a["cnt"] += d.size
            a["sm"] += np.where(den > 0, num/np.where(den > 0, den, 1), 0).sum()
    out = {k: dict(RMSE=np.sqrt(a["se"]/a["cnt"]), MSE=a["se"]/a["cnt"], MAE=a["ae"]/a["cnt"], SMAPE=100*a["sm"]/a["cnt"]) for k, a in acc.items()}
    return n, out
if __name__ == "__main__":
    res = {}
    for name in ["ETTh1", "ETTh2", "ETTm1"]:
        df, cols, raw, z, mu, sd, b1, b2 = load(name)
        print(f"\n===== {name}: rows={len(df)} cols={cols}")
        print("dates:", df.date.iloc[0], "->", df.date.iloc[-1], "| rows used by protocol:", b2[2], "of", len(df))
        for k, s in enumerate(["train", "val", "test"]):
            seg, n = windows(z, b1, b2, k)
            tgt0 = b1[k] + SEQ                                     # first target row of this split
            print(f"{s:5s} rows [{b1[k]},{b2[k]}) dates {df.date.iloc[b1[k]]} .. {df.date.iloc[b2[k]-1]}  windows={n}  "
                  f"first target row={tgt0} ({df.date.iloc[tgt0]})")
        # leakage checks
        assert b1[1] + SEQ == b2[0] and b1[2] + SEQ == b2[1], "targets must start exactly at split boundary"
        print("leakage: val/test TARGETS start exactly at the split boundary; inputs reuse the previous 512 rows (standard, past only). OK")
        print("normalization: mean/std from train rows only:", np.round(mu, 3), np.round(sd, 3))
        # distribution shift per split (z units = in train std)
        print(f"{'channel':6s} " + " ".join(f"{s:>16s}" for s in ["train mean/std", "val mean/std", "test mean/std"]) + "   KS(tr,va)  KS(tr,te)  KS(va,te)")
        segs = [z[b2[0]-b2[0]:b2[0]], z[b2[0]:b2[1]], z[b2[1]:b2[2]]]
        for j, c in enumerate(cols):
            ms = [f"{s[:, j].mean():7.2f}/{s[:, j].std():5.2f}" for s in segs]
            ks = [stats.ks_2samp(segs[a][:, j], segs[b][:, j]).statistic for a, b in [(0, 1), (0, 2), (1, 2)]]
            print(f"{c:6s} " + " ".join(f"{m:>16s}" for m in ms) + "   " + "  ".join(f"{k:8.2f}" for k in ks))
        n, out = baselines(name)
        print(f"sanity baselines on TEST ({n} windows, all channels, H=96):")
        for k, v in out.items(): print(f"   {k:12s} RMSE={v['RMSE']:.4f} MSE={v['MSE']:.4f} MAE={v['MAE']:.4f} SMAPE={v['SMAPE']:.2f}%")
        nv, outv = baselines(name, split=1)
        print(f"sanity baselines on VAL ({nv} windows):")
        for k, v in outv.items(): print(f"   {k:12s} RMSE={v['RMSE']:.4f} MAE={v['MAE']:.4f} SMAPE={v['SMAPE']:.2f}%")
        res[name] = dict(test=out, val=outv)
    json.dump(res, open("/home/claude/research/audit/baselines.json", "w"), indent=1, default=float)
