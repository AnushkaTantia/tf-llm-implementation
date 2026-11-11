"""
Benchmark datasets for Stage X (RAFT), with the standard splits of the long-term forecasting
literature (Informer / Autoformer / Time-Series-Library):

  ETTh1, ETTh2 : 12/4/4 months of hourly data       (identical to src/data/loaders.py ETTh1Dataset)
  ETTm1, ETTm2 : 12/4/4 months of 15-minute data    (x4 rows)
  exchange_rate, weather, national_illness (ILI), electricity, traffic:
                 chronological 70% / 10% / 20%  (Dataset_Custom of Time-Series-Library)

Scaling: z-score with training-split statistics only. Validation and test inputs reuse the preceding
seq_len rows, so their TARGETS start exactly at the split boundary.

Few-shot (percent < 100): the EARLIEST `percent` % of the training windows, as in the few-shot protocol of
GPT4TS / Time-LLM:  border2_train = (border2_train - seq_len) * percent // 100 + seq_len.
Validation and test are unchanged.
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from loaders import patchify

SPECS = {
    # name: (file name, kind, number of variables)
    "ETTh1": ("ETTh1.csv", "ett_hour", 7),
    "ETTh2": ("ETTh2.csv", "ett_hour", 7),
    "ETTm1": ("ETTm1.csv", "ett_minute", 7),
    "ETTm2": ("ETTm2.csv", "ett_minute", 7),
    "Exchange": ("exchange_rate.csv", "custom", 8),
    "Weather": ("weather.csv", "custom", 21),
    "ILI": ("national_illness.csv", "custom", 7),
    "ECL": ("electricity.csv", "custom", 321),
    "Traffic": ("traffic.csv", "custom", 862),
}


def dataset_name_from_path(csv_path):
    base = os.path.basename(csv_path)
    for name, (fn, _, _) in SPECS.items():
        if base == fn:
            return name
    raise ValueError(f"unknown dataset file {base}")


def borders(name, n_rows, seq_len):
    kind = SPECS[name][1]
    if kind == "ett_hour":
        tr, va, te = 12 * 30 * 24, 4 * 30 * 24, 4 * 30 * 24
        b1 = [0, tr - seq_len, tr + va - seq_len]
        b2 = [tr, tr + va, tr + va + te]
    elif kind == "ett_minute":
        tr, va, te = 12 * 30 * 24 * 4, 4 * 30 * 24 * 4, 4 * 30 * 24 * 4
        b1 = [0, tr - seq_len, tr + va - seq_len]
        b2 = [tr, tr + va, tr + va + te]
    else:
        tr = int(n_rows * 0.7)
        te = int(n_rows * 0.2)
        va = n_rows - tr - te
        b1 = [0, tr - seq_len, n_rows - te - seq_len]
        b2 = [tr, tr + va, n_rows]
    return b1, b2


class BenchmarkDataset(Dataset):
    """Same interface as loaders.ETTh1Dataset (x, x_patched, y), for any dataset in SPECS."""

    def __init__(self, csv_path, flag="train", seq_len=512, pred_len=96, patch_len=16, percent=100, name=None):
        assert flag in ("train", "val", "test")
        self.name = name or dataset_name_from_path(csv_path)
        self.seq_len, self.pred_len, self.patch_len = seq_len, pred_len, patch_len
        df = pd.read_csv(csv_path)
        cols = [c for c in df.columns if c != "date"]
        assert len(cols) == SPECS[self.name][2], f"{self.name}: {len(cols)} variables, expected {SPECS[self.name][2]}"
        self.feature_cols = cols
        data = df[cols].values.astype(np.float32)
        b1, b2 = borders(self.name, len(data), seq_len)
        assert b1[1] >= 0 and b1[2] >= 0, f"{self.name}: look-back {seq_len} too long for this dataset"
        k = {"train": 0, "val": 1, "test": 2}[flag]
        lo, hi = b1[k], b2[k]
        if flag == "train" and percent < 100:
            hi = (b2[0] - seq_len) * percent // 100 + seq_len
        train = data[b1[0]:b2[0]]
        self.mean = train.mean(0)
        self.std = train.std(0)
        self.std[self.std == 0] = 1.0
        self.data = ((data - self.mean) / self.std)[lo:hi]
        self.borders = (b1, b2)
        assert len(self) > 0, f"{self.name} {flag}: no windows (seq_len {seq_len}, pred_len {pred_len})"

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, i):
        x = torch.from_numpy(self.data[i:i + self.seq_len]).float()
        y = torch.from_numpy(self.data[i + self.seq_len:i + self.seq_len + self.pred_len]).float()
        return {"x": x, "x_patched": patchify(x, self.patch_len), "y": y}
