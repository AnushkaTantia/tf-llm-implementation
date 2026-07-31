"""
Data pipeline for ETTh1 forecasting task.

Implements:
- Standard ETT train/val/test split (12/4/4 months)
- Z-score normalization using train statistics only
- Sliding window sampling for (input, forecast horizon) pairs
- Patching of input sequences (patch length P=16, per TF-LLM paper Section 3.1)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ETTh1Dataset(Dataset):
    """
    PyTorch Dataset for ETTh1 forecasting.

    Each sample is a sliding window over the time series:
      - input:  seq_len time steps (default 512, per paper)
      - target: pred_len time steps immediately following the input

    Returns both the raw input sequence and a patched version of it,
    since the TD/FD encoders operate on patches while some downstream
    logic may need the raw sequence.
    """

    def __init__(
        self,
        csv_path,
        flag="train",
        seq_len=512,
        pred_len=96,
        patch_len=16,
        target_col="OT",
        use_all_features=True,
    ):
        assert flag in ["train", "val", "test"]
        self.flag = flag
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.target_col = target_col

        df = pd.read_csv(csv_path)

        # feature columns: all numeric columns except date, or just target
        feature_cols = [c for c in df.columns if c != "date"]
        if not use_all_features:
            feature_cols = [target_col]
        self.feature_cols = feature_cols
        self.target_idx = feature_cols.index(target_col)

        data = df[feature_cols].values.astype(np.float32)

        # Standard ETT split: 12 months train, 4 months val, 4 months test (hourly data)
        # border indices follow the convention used in the original ETDataset / TSlib papers
        num_train = 12 * 30 * 24
        num_val = 4 * 30 * 24
        num_test = 4 * 30 * 24

        border1s = [0, num_train - seq_len, num_train + num_val - seq_len]
        border2s = [num_train, num_train + num_val, num_train + num_val + num_test]

        split_idx = {"train": 0, "val": 1, "test": 2}[flag]
        border1, border2 = border1s[split_idx], border2s[split_idx]

        # Normalize using train statistics only (prevents test leakage)
        train_data = data[border1s[0]:border2s[0]]
        self.mean = train_data.mean(axis=0)
        self.std = train_data.std(axis=0)
        self.std[self.std == 0] = 1.0  # avoid divide-by-zero on constant columns

        data_normalized = (data - self.mean) / self.std

        self.data = data_normalized[border1:border2]

    def __len__(self):
        # number of valid windows in this split
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        seq_start = idx
        seq_end = seq_start + self.seq_len
        pred_end = seq_end + self.pred_len

        x = self.data[seq_start:seq_end]          # (seq_len, num_features)
        y = self.data[seq_end:pred_end]            # (pred_len, num_features)

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        x_patched = patchify(x, self.patch_len)     # (num_patches, patch_len, num_features)

        return {
            "x": x,                    # raw input sequence
            "x_patched": x_patched,    # patched input sequence
            "y": y,                    # forecast target
        }

    def inverse_transform_target(self, values):
        """Convert normalized target-column values back to original scale."""
        mean = self.mean[self.target_idx]
        std = self.std[self.target_idx]
        return values * std + mean


def patchify(x, patch_len):
    """
    Split a (seq_len, num_features) tensor into non-overlapping patches
    along the time dimension.

    Args:
        x: tensor of shape (seq_len, num_features)
        patch_len: patch length P (paper uses P=16)

    Returns:
        tensor of shape (num_patches, patch_len, num_features)
    """
    seq_len, num_features = x.shape
    assert seq_len % patch_len == 0, (
        f"seq_len ({seq_len}) must be divisible by patch_len ({patch_len}); "
        f"adjust seq_len or patch_len."
    )
    num_patches = seq_len // patch_len
    return x.reshape(num_patches, patch_len, num_features)


if __name__ == "__main__":
    # quick smoke test — run this file directly to sanity check the pipeline
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "ETTh1.csv"

    for flag in ["train", "val", "test"]:
        ds = ETTh1Dataset(csv_path, flag=flag, seq_len=512, pred_len=96, patch_len=16)
        sample = ds[0]
        print(
            f"[{flag}] len={len(ds)}  "
            f"x={tuple(sample['x'].shape)}  "
            f"x_patched={tuple(sample['x_patched'].shape)}  "
            f"y={tuple(sample['y'].shape)}"
        )
