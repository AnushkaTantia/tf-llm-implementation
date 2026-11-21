"""
DLinear and RevIN-Linear controls for the Stage J Electricity settings: the recipe of the Stage X controls
(published DLinear recipe: Adam 1e-3 halved every epoch, batch 32, <= 20 epochs, early stop 5 on validation MSE,
best-validation checkpoint, test evaluated once), with the splits, look-back, scaling and training-window stride of
configs/stage_j_protocol.json (src/data/datasets.py borders, z-score with training statistics).
  python journal/controls_j.py --setting ECL_H96 --csv electricity.csv --out OUT
"""
import argparse, csv, json, os, sys, time
import numpy as np, torch, torch.nn as nn
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src", "data"))
from datasets import SPECS, borders


def metrics(p, t, mu, sd):
    p, t = p.astype(np.float64), t.astype(np.float64); d = p - t
    mse = float((d ** 2).mean()); po, to = p * sd + mu, t * sd + mu
    num, den = 2 * np.abs(po - to), np.abs(po) + np.abs(to)
    smape = 100 * float(np.where(den > 0, num / np.maximum(den, 1e-12), 0).mean())
    return dict(mse=mse, rmse=mse ** 0.5, mae=float(np.abs(d).mean()), smape=smape)


class DLinear(nn.Module):
    def __init__(s, L, pred, k=25):
        super().__init__(); s.k = k; s.avg = nn.AvgPool1d(k, 1, 0); s.ls = nn.Linear(L, pred); s.lt = nn.Linear(L, pred)

    def forward(s, x):
        p = (s.k - 1) // 2
        xp = torch.cat([x[:, :1].repeat(1, p, 1), x, x[:, -1:].repeat(1, s.k - 1 - p, 1)], 1)
        tr = s.avg(xp.permute(0, 2, 1)).permute(0, 2, 1); se = x - tr
        return (s.ls(se.permute(0, 2, 1)) + s.lt(tr.permute(0, 2, 1))).permute(0, 2, 1)


class RevINLinear(nn.Module):
    def __init__(s, L, pred):
        super().__init__(); s.lin = nn.Linear(L, pred)

    def forward(s, x):
        m = x.mean(1, keepdim=True); sd = x.std(1, keepdim=True, unbiased=False) + 1e-5
        return s.lin(((x - m) / sd).permute(0, 2, 1)).permute(0, 2, 1) * sd + m


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--setting", required=True); ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--max_rows", type=int, default=0, help="self-test only")
    a = ap.parse_args()
    P = json.load(open(os.path.join(REPO, "configs", "stage_j_protocol.json")))
    ds, H = a.setting.split("_H"); H = int(H)
    L, stride = P["datasets"][ds]["lookback"], P["datasets"][ds].get("train_stride", 1)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    import pandas as pd
    df = pd.read_csv(a.csv, nrows=a.max_rows or None)
    data = df[[c for c in df.columns if c != "date"]].values.astype(np.float32)
    assert data.shape[1] == SPECS[ds][2]
    b1, b2 = borders(ds, len(data), L)
    tr = data[b1[0]:b2[0]]; mu, sd = tr.mean(0), tr.std(0); sd[sd == 0] = 1
    z = (data - mu) / sd

    def starts(k):
        n = (b2[k] - b1[k]) - L - H + 1
        st = np.arange(n)
        return st[::stride] if k == 0 else st

    def batch(k, st):
        seg = z[b1[k]:b2[k]]; idx = np.asarray(st)[:, None]
        return torch.from_numpy(seg[idx + np.arange(L)]).to(dev), torch.from_numpy(seg[idx + L + np.arange(H)]).to(dev)

    def evaluate(m, k):
        """Streaming metrics (float64 sums), so long horizons on many variables never hold all predictions."""
        m.eval(); se = ae = sm = 0.0; n = 0
        with torch.no_grad():
            for i in range(0, len(starts(k)), 256):
                X, Y = batch(k, starts(k)[i:i + 256])
                p, t = m(X).cpu().numpy().astype(np.float64), Y.cpu().numpy().astype(np.float64); d = p - t
                se += float((d ** 2).sum()); ae += float(np.abs(d).sum()); n += d.size
                po, to = p * sd + mu, t * sd + mu
                num, den = 2 * np.abs(po - to), np.abs(po) + np.abs(to)
                sm += float(np.where(den > 0, num / np.maximum(den, 1e-12), 0).sum())
        mse = se / n
        return dict(mse=mse, rmse=mse ** 0.5, mae=ae / n, smape=100 * sm / n)

    rows = []
    for name, cls in (("DLinear", DLinear), ("RevIN_Linear", RevINLinear)):
        for s in a.seeds:
            t0 = time.time(); torch.manual_seed(s); np.random.seed(s)
            m = cls(L, H).to(dev); opt = torch.optim.Adam(m.parameters(), 1e-3)
            st_tr = starts(0); best, bad, state, bep = 1e9, 0, None, 0
            for ep in range(20):
                m.train(); perm = torch.randperm(len(st_tr)).numpy()      # same shuffling as the Stage X controls
                for i in range(0, len(perm), 32):
                    X, Y = batch(0, st_tr[perm[i:i + 32]])
                    loss = ((m(X) - Y) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
                v = evaluate(m, 1)["mse"]
                if v < best:
                    best, bad, bep, state = v, 0, ep + 1, {k: t.clone() for k, t in m.state_dict().items()}
                else:
                    bad += 1
                for g in opt.param_groups:
                    g["lr"] *= 0.5
                if bad >= 5:
                    break
            m.load_state_dict(state)
            for split, k in (("val", 1), ("test", 2)):
                rows.append(dict(setting=a.setting, predictor=name, seed=s, split=split, best_epoch=bep, seq_len=L,
                                 train_stride=stride, **evaluate(m, k)))
            print(f"{a.setting} {name} s{s} ep{bep} test MSE {rows[-1]['mse']:.4f} RMSE {rows[-1]['rmse']:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    with open(os.path.join(a.out, f"controls_{a.setting}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
