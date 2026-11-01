"""
metrics.py -- test-set evaluation with the journal metrics: RMSE and SMAPE (plus MAPE, MSE, MAE).

* RMSE (and MSE/MAE) are computed in the z-scored space, over every window x 96 steps x 7 variables
  (same convention as the training loss and the published ETTh1 tables).
* SMAPE and MAPE are computed after de-normalizing forecasts and targets to the original units
  (z-scored values cluster around 0, where percentage errors are undefined).
    SMAPE = 100/n * sum 2|yhat - y| / (|y| + |yhat|)      (terms with |y|+|yhat| = 0 count as 0)
    MAPE  = 100/n * sum |yhat - y| / |y|                  (only points with |y| > MAPE_EPS)
* Sums are accumulated in float64 over every element, so RMSE is exact (no per-batch averaging).

Usage
  from metrics import evaluate_full, eval_checkpoint
  python src/metrics.py checkpoints/<run_dir> [<run_dir> ...]      # evaluate saved checkpoints
"""
import os, re, sys, math
import torch
from torch.utils.data import DataLoader

MAPE_EPS = 1e-3


@torch.no_grad()
def evaluate_full(forecaster, dataloader, device):
    ds = dataloader.dataset
    mean = torch.as_tensor(ds.mean, dtype=torch.float64, device=device)
    std = torch.as_tensor(ds.std, dtype=torch.float64, device=device)
    forecaster.eval()
    n = 0
    se = ae = smape = 0.0
    mape, n_mape = 0.0, 0
    for batch in dataloader:
        x_patched = batch["x_patched"].to(device)
        x_raw = batch["x"].to(device)
        y = batch["y"].to(device).double()
        yhat, _ = forecaster(x_patched, x_raw)
        yhat = yhat.double()
        d = yhat - y
        se += float((d * d).sum()); ae += float(d.abs().sum()); n += d.numel()
        y_o, yhat_o = y * std + mean, yhat * std + mean          # original units
        num = 2.0 * (yhat_o - y_o).abs()
        den = y_o.abs() + yhat_o.abs()
        smape += float(torch.where(den > 0, num / den.clamp_min(1e-12), torch.zeros_like(den)).sum())
        m = y_o.abs() > MAPE_EPS
        mape += float(((yhat_o - y_o).abs()[m] / y_o.abs()[m]).sum()); n_mape += int(m.sum())
    mse = se / n
    return {"mse": mse, "rmse": math.sqrt(mse), "mae": ae / n,
            "smape": 100.0 * smape / n, "mape": 100.0 * mape / max(n_mape, 1)}


def _loaders(csv_path, flag, pred_len, batch_size=8):
    from loaders import ETTh1Dataset
    ds = ETTh1Dataset(csv_path, flag=flag, seq_len=512, pred_len=pred_len, patch_len=16)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)


def eval_checkpoint(ckpt_path, csv_path, pred_len=96, use_revin=None, device="cuda"):
    """Evaluate a saved best_model.pt on the test split.
    Checkpoints written by the Stage 1 code store their full configuration ("config"); the model
    is rebuilt from it exactly. Older checkpoints (no config) were all trained with the original
    architecture (mask head, frozen GPT-2); for those, use_revin=None decides RevIN automatically by
    reproducing the stored validation MSE."""
    import train as T
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config")
    if cfg is not None:
        forecaster, _ = T.build_model(pred_len=cfg["pred_len"], use_revin=cfg["use_revin"],
                                      seq_len=cfg["seq_len"], head_type=cfg["head_type"],
                                      use_llm=cfg["use_llm"], llm_policy=cfg["llm_policy"])
        forecaster.load_state_dict(ckpt["forecaster"])          # strict: must match exactly
        forecaster.to(device)
        res = evaluate_full(forecaster, _loaders(csv_path, "test", cfg["pred_len"]), device)
        res.update(use_revin=cfg["use_revin"], best_epoch=ckpt.get("epoch"), loss_mode=ckpt.get("loss_mode"),
                   head_type=cfg["head_type"], use_llm=cfg["use_llm"], llm_policy=cfg["llm_policy"])
        return res
    options = [use_revin] if use_revin is not None else [True, False]
    best = None
    for rv in options:
        forecaster, tfb = T.build_model(pred_len=pred_len, use_revin=rv,
                                        head_type="mask", use_llm=True, llm_policy="frozen")
        try:
            forecaster.load_state_dict(ckpt["forecaster"])
        except Exception:
            continue
        forecaster.to(device)
        if len(options) == 1:
            best = (rv, forecaster); break
        v = evaluate_full(forecaster, _loaders(csv_path, "val", pred_len), device)["mse"]
        gap = abs(v - float(ckpt.get("val_mse", v)))
        if best is None or gap < best[2]:
            best = (rv, forecaster, gap)
    if best is None:
        raise RuntimeError(f"could not load {ckpt_path}")
    rv, forecaster = best[0], best[1]
    res = evaluate_full(forecaster, _loaders(csv_path, "test", pred_len), device)
    res.update(use_revin=rv, best_epoch=ckpt.get("epoch"), loss_mode=ckpt.get("loss_mode"))
    return res


if __name__ == "__main__":
    import glob
    REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(REPO, "src"))
    csv_path = sorted(glob.glob(os.path.join(REPO, "**", "ETTh1.csv"), recursive=True))[0]
    for d in sys.argv[1:]:
        p = d if d.endswith(".pt") else os.path.join(d, "best_model.pt")
        m = re.search(r"_H(\d+)", d)
        r = eval_checkpoint(p, csv_path, pred_len=int(m.group(1)) if m else 96)
        print(f"{d}: RMSE={r['rmse']:.4f} SMAPE={r['smape']:.2f}% MAPE={r['mape']:.2f}% "
              f"MSE={r['mse']:.4f} MAE={r['mae']:.4f} revin={r['use_revin']} epoch={r['best_epoch']} mode={r['loss_mode']}")
