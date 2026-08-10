"""
Training loop for TF-LLM forecasting task (ETTh1, horizon 96 first).

Combines the forecasting task loss (MSE) with the TFB loss (Eq. 4:
L_TFB = lambda*(L_T + L_F) + (1-lambda)*L_B), per Stage 7 of the plan.

IMPLEMENTATION NOTES (documented deviations, due to paper ambiguity):

1. Task loss + L_TFB combination: the paper does not give an explicit
   formula for combining the forecasting task loss with L_TFB in the
   unified multi-task setting (Eq. 4 only combines L_T, L_F, L_B WITHIN
   the TFB module). We use simple addition: L_total = L_task + L_TFB.

2. Hyperparameters: Table 11 states learning_rate=1e-4, while Section
   4.2's prose states "an initial learning rate of 3e-4" -- a genuine
   contradiction in the paper itself. We follow Table 11 (1e-4) as the
   dedicated hyperparameter reference.

3. Training loss metric: the paper reports both MSE and MAE as
   evaluation metrics but does not explicitly state which is used as the
   training objective. We use MSE (standard for forecasting, and the
   metric Table 1 leads with), tracking MAE alongside for monitoring.
"""

import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), "data"))
sys.path.append(os.path.join(os.path.dirname(__file__), "losses"))
sys.path.append(os.path.join(os.path.dirname(__file__), "models"))

from loaders import ETTh1Dataset
from contrastive import nt_xent_loss
from td_encoder import TDEncoder, apply_td_augmentation
from fd_encoder import FDEncoder, apply_fd_augmentation
from tfb_module import TFBModule, balance_loss, tfb_total_loss
from tf_llm import TFLLMBackbone
from forecast_head import TFLLMForecaster


def build_model(d_emb=64, patch_len=16, num_features=7, pred_len=96, num_prompt_tokens=10):
    """Instantiates all trainable components and wires them together."""
    td_encoder = TDEncoder(patch_len=patch_len, num_features=num_features, d_emb=d_emb)
    fd_encoder = FDEncoder(seq_len=512, num_features=num_features, d_emb=d_emb)
    backbone = TFLLMBackbone(d_emb=d_emb, num_prompt_tokens=num_prompt_tokens)
    tfb = TFBModule(d_emb=d_emb, proj_dim=128)

    forecaster = TFLLMForecaster(
        td_encoder=td_encoder,
        fd_encoder=fd_encoder,
        backbone=backbone,
        pred_len=pred_len,
        patch_len=patch_len,
        num_features=num_features,
    )

    return forecaster, tfb


def get_trainable_parameters(forecaster, tfb):
    """
    Collects all trainable parameters across the forecaster (TD/FD
    encoders, backbone's fusion/projection/prompt/mask components -- GPT-2
    itself stays frozen) and the TFB module's projection heads.
    """
    params = list(forecaster.parameters()) + list(tfb.parameters())
    trainable = [p for p in params if p.requires_grad]
    return trainable


def compute_losses(forecaster, tfb, x_patched, x_raw, y, lam=0.5):
    """
    Runs one forward pass and computes all loss components.

    Returns:
        dict with 'task_loss', 'l_t', 'l_f', 'l_b', 'l_tfb', 'total_loss',
        and 'forecast' (for optional MAE tracking)
    """
    # forward pass on ORIGINAL (non-augmented) inputs -- gives the forecast
    # and the original z_T, z_F embeddings (via aux)
    forecast, aux = forecaster(x_patched, x_raw)
    z_T, z_F = aux["z_T"], aux["z_F"]

    task_loss = nn.functional.mse_loss(forecast, y)

    # augmented views, encoded independently for the contrastive objectives
    x_patched_aug = apply_td_augmentation(x_patched)
    x_raw_aug = apply_fd_augmentation(x_raw)

    z_T_aug, _ = forecaster.td_encoder(x_patched_aug)
    z_F_aug, _ = forecaster.fd_encoder(x_raw_aug)

    l_t = nt_xent_loss(z_T, z_T_aug)
    l_f = nt_xent_loss(z_F, z_F_aug)

    p_T, p_F = tfb(z_T, z_F)
    p_T_aug, p_F_aug = tfb(z_T_aug, z_F_aug)
    l_b = balance_loss(p_T, p_T_aug, p_F, p_F_aug)

    l_tfb = tfb_total_loss(l_t, l_f, l_b, lam=lam)

    total_loss = task_loss + l_tfb

    return {
        "task_loss": task_loss,
        "l_t": l_t,
        "l_f": l_f,
        "l_b": l_b,
        "l_tfb": l_tfb,
        "total_loss": total_loss,
        "forecast": forecast,
    }


def evaluate(forecaster, tfb, dataloader, device):
    """Computes MSE and MAE on a dataset split, no augmentation/TFB loss."""
    forecaster.eval()
    total_mse, total_mae, n_batches = 0.0, 0.0, 0

    with torch.no_grad():
        for batch in dataloader:
            x_patched = batch["x_patched"].to(device)
            x_raw = batch["x"].to(device)
            y = batch["y"].to(device)

            forecast, _ = forecaster(x_patched, x_raw)

            mse = nn.functional.mse_loss(forecast, y)
            mae = nn.functional.l1_loss(forecast, y)

            total_mse += mse.item()
            total_mae += mae.item()
            n_batches += 1

    forecaster.train()
    return total_mse / n_batches, total_mae / n_batches


def train(
    csv_path,
    checkpoint_dir,
    seq_len=512,
    pred_len=96,
    patch_len=16,
    num_features=7,
    batch_size=16,       # per paper Table 11
    learning_rate=1e-4,   # per paper Table 11 (see note re: 3e-4 discrepancy)
    epochs=50,             # per paper Table 11
    lam=0.5,                # TFB balance weight, paper doesn't specify exact value used
    device=None,
    log_every=50,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_ds = ETTh1Dataset(csv_path, flag="train", seq_len=seq_len, pred_len=pred_len, patch_len=patch_len)
    val_ds = ETTh1Dataset(csv_path, flag="val", seq_len=seq_len, pred_len=pred_len, patch_len=patch_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    forecaster, tfb = build_model(
        d_emb=64, patch_len=patch_len, num_features=num_features, pred_len=pred_len
    )
    forecaster.to(device)
    tfb.to(device)

    trainable_params = get_trainable_parameters(forecaster, tfb)
    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)

    print(f"Device: {device}")
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    print(f"Train batches/epoch: {len(train_loader)}  Val batches: {len(val_loader)}")
    print("-" * 70)

    best_val_mse = float("inf")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        running_total, running_task = 0.0, 0.0

        for step, batch in enumerate(train_loader):
            x_patched = batch["x_patched"].to(device)
            x_raw = batch["x"].to(device)
            y = batch["y"].to(device)

            losses = compute_losses(forecaster, tfb, x_patched, x_raw, y, lam=lam)

            optimizer.zero_grad()
            losses["total_loss"].backward()
            optimizer.step()

            running_total += losses["total_loss"].item()
            running_task += losses["task_loss"].item()

            if (step + 1) % log_every == 0:
                print(
                    f"  epoch {epoch} step {step+1}/{len(train_loader)}  "
                    f"total={losses['total_loss'].item():.4f}  "
                    f"task={losses['task_loss'].item():.4f}  "
                    f"L_TFB={losses['l_tfb'].item():.4f}"
                )

        val_mse, val_mae = evaluate(forecaster, tfb, val_loader, device)
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch}/{epochs}  "
            f"train_total={running_total/len(train_loader):.4f}  "
            f"train_task={running_task/len(train_loader):.4f}  "
            f"val_MSE={val_mse:.4f}  val_MAE={val_mae:.4f}  "
            f"({epoch_time:.1f}s)"
        )

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save(
                {
                    "forecaster": forecaster.state_dict(),
                    "tfb": tfb.state_dict(),
                    "epoch": epoch,
                    "val_mse": val_mse,
                    "val_mae": val_mae,
                },
                checkpoint_path,
            )
            print(f"  -> saved new best checkpoint (val_MSE={val_mse:.4f})")

        print("-" * 70)

    print(f"Training complete. Best val MSE: {best_val_mse:.4f}")
    return forecaster, tfb


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--pred_len", type=int, default=96)
    args = parser.parse_args()

    train(
        csv_path=args.csv_path,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        pred_len=args.pred_len,
    )
