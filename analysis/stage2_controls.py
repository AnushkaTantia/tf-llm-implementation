"""
Stage 2 (S2) forecasting controls on the frozen ETTh1 protocol, computed with an INDEPENDENT pipeline:
the data loader of the data audit (audit/data_audit.py, written separately from src/data/loaders.py) and
metric code written here in numpy (separate from src/metrics.py).

  python s2/stage2_controls.py --out s2/out [--seeds 42 123 ...]

Controls
  deterministic: training mean, last value, window mean (last 512 h), seasonal naive 24 h, seasonal naive 168 h
  trained (10 seeds each): DLinear (Zeng et al., AAAI 2023) with its published training recipe
                           (Adam lr 1e-3 halved every epoch, batch 32, <=20 epochs, early stop 5 on validation MSE)
                           RevIN-Linear: per-window normalisation + one linear map 512->96 per variable, same recipe
Protocol (identical to the corrected forecaster): ETTh1, 7 variables, lookback 512, horizon 96, chronological
12/4/4-month split, z-score with training statistics only, best-validation checkpoint, test evaluated once.

Outputs: controls.csv (one row per predictor and seed), per_horizon.csv, per_variable.csv,
         preds_<name>_s42.npz (validation/test forecasts, for the temporal-learning analysis), and a check that the
         repository loader (src/data/loaders.py) produces exactly the same windows as the independent loader.
"""
import argparse, csv, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data_audit import load                                # independent loader (data audit)

SEQ, PRED = 512, 96
MAPE_EPS = 1e-3


# ------------------------------------------------------------------ independent metrics (numpy, float64)
def metrics(pred_z, true_z, mu, sd):
    p, t = pred_z.astype(np.float64), true_z.astype(np.float64)
    d = p - t
    mse = float((d ** 2).mean())
    po, to = p * sd + mu, t * sd + mu
    num, den = 2 * np.abs(po - to), np.abs(po) + np.abs(to)
    smape = 100 * float(np.where(den > 0, num / np.maximum(den, 1e-12), 0).mean())
    m = np.abs(to) > MAPE_EPS
    mape = 100 * float((np.abs(po - to)[m] / np.abs(to)[m]).mean())
    return dict(mse=mse, rmse=mse ** 0.5, mae=float(np.abs(d).mean()), smape=smape, mape=mape)


def windows(z, b1, b2, k):
    seg = z[b1[k]:b2[k]]
    n = len(seg) - SEQ - PRED + 1
    idx = np.arange(n)[:, None]
    return seg[idx + np.arange(SEQ)].astype(np.float32), seg[idx + SEQ + np.arange(PRED)].astype(np.float32)


# ------------------------------------------------------------------ models
class DLinear(nn.Module):
    def __init__(s, k=25):
        super().__init__(); s.k = k; s.avg = nn.AvgPool1d(k, 1, 0)
        s.ls = nn.Linear(SEQ, PRED); s.lt = nn.Linear(SEQ, PRED)

    def forward(s, x):
        p = (s.k - 1) // 2
        xp = torch.cat([x[:, :1].repeat(1, p, 1), x, x[:, -1:].repeat(1, s.k - 1 - p, 1)], 1)
        tr = s.avg(xp.permute(0, 2, 1)).permute(0, 2, 1); se = x - tr
        return (s.ls(se.permute(0, 2, 1)) + s.lt(tr.permute(0, 2, 1))).permute(0, 2, 1)


class RevINLinear(nn.Module):
    def __init__(s):
        super().__init__(); s.lin = nn.Linear(SEQ, PRED)

    def forward(s, x):
        m = x.mean(1, keepdim=True); sd = x.std(1, keepdim=True, unbiased=False) + 1e-5
        return s.lin(((x - m) / sd).permute(0, 2, 1)).permute(0, 2, 1) * sd + m


def predict(m, X):
    m.eval()
    with torch.no_grad():
        return torch.cat([m(torch.from_numpy(X[i:i + 512])) for i in range(0, len(X), 512)]).numpy()


def train_model(cls, seed, data):
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = data
    torch.manual_seed(seed); np.random.seed(seed)
    m = cls(); opt = torch.optim.Adam(m.parameters(), 1e-3)
    Xt, Yt = torch.from_numpy(Xtr), torch.from_numpy(Ytr)
    best, bad, state, bep, hist = 1e9, 0, None, 0, []
    for ep in range(20):
        m.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), 32):
            b = perm[i:i + 32]; loss = ((m(Xt[b]) - Yt[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        v = float(((predict(m, Xva) - Yva) ** 2).mean()); hist.append(v)
        if v < best:
            best, bad, bep, state = v, 0, ep + 1, {k: t.clone() for k, t in m.state_dict().items()}
        else:
            bad += 1
        for g in opt.param_groups:
            g["lr"] *= 0.5
        if bad >= 5:
            break
    m.load_state_dict(state)
    return m, best, bep, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 2024, 1, 7, 21, 100, 999])
    ap.add_argument("--src", default=os.path.join(HERE, "..", "src"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df, cols, raw, z, mu, sd, b1, b2 = load("ETTh1")
    mu, sd = np.asarray(mu, np.float64), np.asarray(sd, np.float64)
    data = [windows(z, b1, b2, k) for k in range(3)]
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = data
    log = {"variables": list(cols), "n_windows": [len(d[0]) for d in data]}

    # --- loader agreement: repository loader vs independent loader
    sys.path.insert(0, os.path.join(a.src, "data"))
    from loaders import ETTh1Dataset
    csvp = os.path.join(os.environ.get("ETT_DATA_DIR", os.path.join(HERE, "..", "data")), "ETTh1.csv")
    agree = {}
    for k, flag in enumerate(["train", "val", "test"]):
        ds = ETTh1Dataset(csvp, flag=flag, seq_len=SEQ, pred_len=PRED, patch_len=16)
        idx = [0, len(ds) // 2, len(ds) - 1]
        mx = max(max(float(np.abs(np.asarray(ds[i]["x"]) - data[k][0][i]).max()),
                     float(np.abs(np.asarray(ds[i]["y"]) - data[k][1][i]).max())) for i in idx)
        assert len(ds) == len(data[k][0]), (flag, len(ds), len(data[k][0]))
        agree[flag] = dict(n=len(ds), max_abs_diff=mx)
        assert mx < 1e-5, (flag, mx)
    log["loader_agreement"] = agree
    print("repository loader == independent loader:", agree)

    rows, per_h, per_v = [], [], []

    def record(name, seed, split, P, Y, extra=None):
        r = dict(predictor=name, seed=seed, split=split, **metrics(P, Y, mu, sd))
        if extra:
            r.update(extra)
        rows.append(r)
        if split == "test":
            e = ((P.astype(np.float64) - Y) ** 2)
            per_h.extend(dict(predictor=name, seed=seed, step=h + 1, mse=float(v)) for h, v in enumerate(e.mean((0, 2))))
            per_v.extend(dict(predictor=name, seed=seed, variable=cols[j], mse=float(v)) for j, v in enumerate(e.mean((0, 1))))

    # --- deterministic controls
    det = {
        "train_mean": lambda X: np.zeros((len(X), PRED, X.shape[2]), np.float32),
        "last_value": lambda X: np.repeat(X[:, -1:], PRED, 1),
        "window_mean": lambda X: np.repeat(X.mean(1, keepdims=True), PRED, 1),
        "seasonal_naive_24h": lambda X: np.concatenate([X[:, -24:]] * 4, 1),
        "seasonal_naive_168h": lambda X: X[:, -168:-168 + PRED],
    }
    for name, f in det.items():
        for split, (X, Y) in (("val", data[1]), ("test", data[2])):
            record(name, None, split, f(X), Y)
        print(f"{name:20s} test MSE {rows[-1]['mse']:.4f}  RMSE {rows[-1]['rmse']:.4f}  SMAPE {rows[-1]['smape']:.2f}")

    # --- trained controls
    for name, cls in (("DLinear", DLinear), ("RevIN_Linear", RevINLinear)):
        for s in a.seeds:
            t0 = time.time()
            m, bv, bep, hist = train_model(cls, s, data)
            Pv, Pt = predict(m, Xva), predict(m, Xte)
            ex = dict(best_epoch=bep, seconds=round(time.time() - t0, 1), val_curve=json.dumps([round(x, 4) for x in hist]))
            record(name, s, "val", Pv, Yva, ex); record(name, s, "test", Pt, Yte, ex)
            if s == 42:
                np.savez_compressed(os.path.join(a.out, f"preds_{name}_s42.npz"), val=Pv, test=Pt)
            print(f"{name:13s} seed {s:5d}  best epoch {bep:2d}  val {bv:.4f}  test MSE {rows[-1]['mse']:.4f} "
                  f"RMSE {rows[-1]['rmse']:.4f} SMAPE {rows[-1]['smape']:.2f}  ({time.time() - t0:.0f}s)", flush=True)

    for fn, rr in (("controls.csv", rows), ("per_horizon.csv", per_h), ("per_variable.csv", per_v)):
        with open(os.path.join(a.out, fn), "w", newline="") as fh:
            keys = list(dict.fromkeys(k for r in rr for k in r))
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rr)
    np.savez_compressed(os.path.join(a.out, "targets.npz"), val=Yva, test=Yte, val_x_mean=Xva.mean(1), test_x_mean=Xte.mean(1),
                        mu=mu, sd=sd)
    json.dump(log, open(os.path.join(a.out, "controls_log.json"), "w"), indent=1)

    # --- summary
    lines = ["S2 controls, ETTh1 H=96, test split (independent loader and metrics)", ""]
    for name in list(det) + ["DLinear", "RevIN_Linear"]:
        rs = [r for r in rows if r["predictor"] == name and r["split"] == "test"]
        f = lambda k: (np.mean([r[k] for r in rs]), np.std([r[k] for r in rs], ddof=1) if len(rs) > 1 else 0.0)
        (m1, s1), (m2, s2_), (m3, s3), (m4, s4) = f("mse"), f("rmse"), f("mae"), f("smape")
        lines.append(f"{name:20s} n={len(rs):2d}  MSE {m1:.4f}±{s1:.4f}  RMSE {m2:.4f}±{s2_:.4f}  MAE {m3:.4f}±{s3:.4f}  SMAPE {m4:.2f}±{s4:.2f}")
    open(os.path.join(a.out, "controls_summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
