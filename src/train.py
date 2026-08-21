"""
Training loop for TF-LLM forecasting task (ETTh1, horizon 96 first).
Combines the forecasting task loss (MSE) with the TFB loss (Eq. 4:
L_TFB = lambda*(L_T + L_F) + (1-lambda)*L_B), per Stage 7 of the plan.
UPDATE:
- Added test-set evaluation (previously only train/val existed; the
  paper's Table 1 numbers are TEST-set results, so val_MSE alone is not
  the reportable/comparable number).
- Added weight_decay=1e-4 to the optimizer, per paper Section 4.2.
- Added a switchable loss-combination mode for task_loss vs. L_TFB,
  since the paper's Eq. 4 only combines L_T/L_F/L_B WITHIN the TFB
  module and gives no explicit formula for combining L_TFB with the
  forecasting task loss. Previously we used plain addition; this was
  identified as a likely cause of poor validation generalization
  (L_TFB was consistently 2-3x larger than task_loss, likely dominating
  gradients). Now supports: sum, average, fixed_weighted,
  variable_weighted (learnable), time_varying (scheduled).
- Added a standalone sanity_check() function to print raw forecast vs.
  target values on one batch, to rule out scale/shape bugs before
  attributing poor validation performance to the loss-combination
  strategy alone.
IMPLEMENTATION NOTES (documented deviations, due to paper ambiguity):
1. Task loss + L_TFB combination: the paper does not give an explicit
   formula for combining the forecasting task loss with L_TFB in the
   unified multi-task setting (Eq. 4 only combines L_T, L_F, L_B WITHIN
   the TFB module). We now support multiple combination strategies
   (see loss_mode below) instead of a single hardcoded choice, and
   select the best-performing one empirically via short trials.
2. Hyperparameters: Table 11 states learning_rate=1e-4, while Section
   4.2's prose states "an initial learning rate of 3e-4" -- a genuine
   contradiction in the paper itself. We follow Table 11 (1e-4) as the
   dedicated hyperparameter reference. weight_decay=1e-4 and
   dropout=0.2 are both stated in Section 4.2 prose only; weight_decay
   is now applied (dropout is architecture-level and tracked separately
   in the model files).
3. Training loss metric: the paper reports both MSE and MAE as
   evaluation metrics but does not explicitly state which is used as the
   training objective. We use MSE (standard for forecasting, and the
   metric Table 1 leads with), tracking MAE alongside for monitoring.
"""
import os
import sys
import time
import math
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
def get_trainable_parameters(forecaster, tfb, loss_combiner=None):
    """
    Collects all trainable parameters across the forecaster (TD/FD
    encoders, backbone's fusion/projection/prompt/mask components -- GPT-2
    itself stays frozen), the TFB module's projection heads, and (if
    loss_mode='variable_weighted') the learnable loss-weighting parameters.
    """
    params = list(forecaster.parameters()) + list(tfb.parameters())
    if loss_combiner is not None:
        params += list(loss_combiner.parameters())
    trainable = [p for p in params if p.requires_grad]
    return trainable
class LearnableLossWeight(nn.Module):
    """
    Learnable weighting for combining task_loss and l_tfb, using the
    uncertainty-weighting approach (Kendall et al., 2018):
        total = task_loss / (2*sigma1^2) + l_tfb / (2*sigma2^2)
                + log(sigma1) + log(sigma2)
    We parameterize log(sigma^2) directly for numerical stability.
    Used only when loss_mode='variable_weighted'.
    """
    def __init__(self):
        super().__init__()
        self.log_var_task = nn.Parameter(torch.zeros(1))
        self.log_var_tfb = nn.Parameter(torch.zeros(1))
    def forward(self, task_loss, l_tfb):
        precision_task = torch.exp(-self.log_var_task)
        precision_tfb = torch.exp(-self.log_var_tfb)
        total = (
            precision_task * task_loss + self.log_var_task
            + precision_tfb * l_tfb + self.log_var_tfb
        )
        return total.squeeze()
def combine_losses(task_loss, l_tfb, loss_mode="sum", alpha=0.3,
                    loss_combiner=None, epoch=None, total_epochs=None,
                    warmup_epochs=15):
    """
    Combines task_loss and l_tfb according to the selected strategy.
    Args:
        task_loss, l_tfb: scalar tensors for this step
        loss_mode: one of "sum", "average", "fixed_weighted",
            "variable_weighted", "time_varying", "median", "min", "max"
        alpha: weight on l_tfb, used by "fixed_weighted" and as the
            target/max weight for "time_varying"
        loss_combiner: a LearnableLossWeight instance, required if
            loss_mode == "variable_weighted"
        epoch, total_epochs, warmup_epochs: used by "time_varying" to
            compute the current schedule position
    Returns:
        total_loss (scalar tensor), effective_alpha (float, for logging;
        None for modes where a single alpha isn't meaningful)
    """
    if loss_mode == "sum":
        return task_loss + l_tfb, None
    elif loss_mode == "average":
        return (task_loss + l_tfb) / 2.0, 0.5
    elif loss_mode == "fixed_weighted":
        return task_loss + alpha * l_tfb, alpha
    elif loss_mode == "variable_weighted":
        assert loss_combiner is not None, "variable_weighted requires a LearnableLossWeight instance"
        return loss_combiner(task_loss, l_tfb), None
    elif loss_mode == "time_varying":
        # Linear warmup of the L_TFB weight from 0 -> alpha over
        # `warmup_epochs`, then held constant at alpha. Rationale: let
        # the forecast head learn a reasonable signal first, then
        # gradually introduce the time-frequency balance structure
        # rather than imposing it from step 1.
        assert epoch is not None, "time_varying requires the current epoch"
        progress = min(epoch / max(warmup_epochs, 1), 1.0)
        current_alpha = alpha * progress
        return task_loss + current_alpha * l_tfb, current_alpha
    elif loss_mode == "median":
        # Included per request; treated as median of the two scalar
        # values themselves (not a per-sample median), which is a weak
        # combination strategy for two aggregate losses but provided
        # for completeness/comparison.
        stacked = torch.stack([task_loss, l_tfb])
        return torch.median(stacked), None
    elif loss_mode == "min":
        # Total loss is whichever of the two components is currently
        # smaller. Included for completeness per the ablation request;
        # like median, this is a weak combination strategy for two
        # aggregate scalar losses (no clean gradient interpretation as
        # a weighted objective), provided mainly as a baseline point of
        # comparison rather than a principled design choice.
        stacked = torch.stack([task_loss, l_tfb])
        return torch.min(stacked), None
    elif loss_mode == "max":
        # Total loss is whichever of the two components is currently
        # larger -- effectively forces optimization to address whichever
        # objective is lagging. Same caveats as "min" above.
        stacked = torch.stack([task_loss, l_tfb])
        return torch.max(stacked), None
    else:
        raise ValueError(f"Unknown loss_mode: {loss_mode}")
def compute_losses(forecaster, tfb, x_patched, x_raw, y, lam=0.5,
                    loss_mode="sum", alpha=0.3, loss_combiner=None,
                    epoch=None, total_epochs=None, warmup_epochs=15):
    """
    Runs one forward pass and computes all loss components.
    Returns:
        dict with 'task_loss', 'l_t', 'l_f', 'l_b', 'l_tfb', 'total_loss',
        'effective_alpha', and 'forecast' (for optional MAE tracking)
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
    total_loss, effective_alpha = combine_losses(
        task_loss, l_tfb, loss_mode=loss_mode, alpha=alpha,
        loss_combiner=loss_combiner, epoch=epoch,
        total_epochs=total_epochs, warmup_epochs=warmup_epochs,
    )
    return {
        "task_loss": task_loss,
        "l_t": l_t,
        "l_f": l_f,
        "l_b": l_b,
        "l_tfb": l_tfb,
        "total_loss": total_loss,
        "effective_alpha": effective_alpha,
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
def sanity_check(forecaster, tfb, dataloader, device, n_samples=5):
    """
    Step 1 of the Stage 8 debugging plan: prints raw forecast vs. target
    values on one batch to rule out a scale/shape mismatch bug before
    attributing poor validation performance purely to loss-weighting.
    Call this BEFORE running any loss-mode trials.
    """
    forecaster.eval()
    batch = next(iter(dataloader))
    x_patched = batch["x_patched"].to(device)
    x_raw = batch["x"].to(device)
    y = batch["y"].to(device)
    with torch.no_grad():
        forecast, _ = forecaster(x_patched, x_raw)
    print("=== Sanity check: forecast vs. target (normalized space) ===")
    print(f"forecast shape: {tuple(forecast.shape)}   target shape: {tuple(y.shape)}")
    print(f"forecast[0, :{n_samples}, 0]: {forecast[0, :n_samples, 0].tolist()}")
    print(f"target[0,   :{n_samples}, 0]: {y[0, :n_samples, 0].tolist()}")
    print(f"forecast mean/std: {forecast.mean().item():.4f} / {forecast.std().item():.4f}")
    print(f"target   mean/std: {y.mean().item():.4f} / {y.std().item():.4f}")
    mse = nn.functional.mse_loss(forecast, y).item()
    print(f"single-batch MSE: {mse:.4f}")
    print("=" * 60)
    forecaster.train()
    return mse
def train(
    csv_path,
    checkpoint_dir,
    seq_len=512,
    pred_len=96,
    patch_len=16,
    num_features=7,
    batch_size=16,          # per paper Table 11
    learning_rate=1e-4,     # per paper Table 11 (see note re: 3e-4 discrepancy)
    weight_decay=1e-4,      # per paper Section 4.2
    epochs=50,              # per paper Table 11
    lam=0.5,                # TFB internal balance weight (Eq. 4, L_T/L_F vs L_B)
    loss_mode="sum",        # task_loss vs l_tfb combination: sum/average/
                             # fixed_weighted/variable_weighted/time_varying/median/min/max
    alpha=0.3,               # weight on l_tfb for fixed_weighted / time_varying
    warmup_epochs=15,        # for time_varying: epochs to reach full alpha
    device=None,
    log_every=50,
    run_sanity_check=True,
    run_test_eval=True,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(checkpoint_dir, exist_ok=True)
    train_ds = ETTh1Dataset(csv_path, flag="train", seq_len=seq_len, pred_len=pred_len, patch_len=patch_len)
    val_ds = ETTh1Dataset(csv_path, flag="val", seq_len=seq_len, pred_len=pred_len, patch_len=patch_len)
    test_ds = ETTh1Dataset(csv_path, flag="test", seq_len=seq_len, pred_len=pred_len, patch_len=patch_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    forecaster, tfb = build_model(
        d_emb=64, patch_len=patch_len, num_features=num_features, pred_len=pred_len
    )
    forecaster.to(device)
    tfb.to(device)
    loss_combiner = None
    if loss_mode == "variable_weighted":
        loss_combiner = LearnableLossWeight().to(device)
    trainable_params = get_trainable_parameters(forecaster, tfb, loss_combiner=loss_combiner)
    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    print(f"Device: {device}")
    print(f"Loss mode: {loss_mode}" + (f" (alpha={alpha})" if loss_mode in ("fixed_weighted", "time_varying") else ""))
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    print(f"Train batches/epoch: {len(train_loader)}  Val batches: {len(val_loader)}  Test batches: {len(test_loader)}")
    print("-" * 70)
    if run_sanity_check:
        sanity_check(forecaster, tfb, val_loader, device)
    best_val_mse = float("inf")
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        running_total, running_task = 0.0, 0.0
        for step, batch in enumerate(train_loader):
            x_patched = batch["x_patched"].to(device)
            x_raw = batch["x"].to(device)
            y = batch["y"].to(device)
            losses = compute_losses(
                forecaster, tfb, x_patched, x_raw, y, lam=lam,
                loss_mode=loss_mode, alpha=alpha, loss_combiner=loss_combiner,
                epoch=epoch, total_epochs=epochs, warmup_epochs=warmup_epochs,
            )
            optimizer.zero_grad()
            losses["total_loss"].backward()
            optimizer.step()
            running_total += losses["total_loss"].item()
            running_task += losses["task_loss"].item()
            if (step + 1) % log_every == 0:
                alpha_str = f"  alpha={losses['effective_alpha']:.3f}" if losses["effective_alpha"] is not None else ""
                print(
                    f"  epoch {epoch} step {step+1}/{len(train_loader)}  "
                    f"total={losses['total_loss'].item():.4f}  "
                    f"task={losses['task_loss'].item():.4f}  "
                    f"L_TFB={losses['l_tfb'].item():.4f}{alpha_str}"
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
                    "loss_mode": loss_mode,
                },
                checkpoint_path,
            )
            print(f"  -> saved new best checkpoint (val_MSE={val_mse:.4f})")
        print("-" * 70)
    print(f"Training complete. Best val MSE: {best_val_mse:.4f}")
    test_mse, test_mae = None, None
    if run_test_eval:
        # Reload best checkpoint before final test evaluation, so the
        # reported number reflects the best model, not just whatever
        # state training happened to end on.
        checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=device)
            forecaster.load_state_dict(ckpt["forecaster"])
            tfb.load_state_dict(ckpt["tfb"])
        test_mse, test_mae = evaluate(forecaster, tfb, test_loader, device)
        print(f"TEST SET  ->  MSE: {test_mse:.4f}  MAE: {test_mae:.4f}")
        print(f"Paper's reported (Table 1, ETTh1, horizon 96)  ->  MSE: 0.339  MAE: 0.373")
    return forecaster, tfb, {"best_val_mse": best_val_mse, "test_mse": test_mse, "test_mae": test_mae}
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--loss_mode", type=str, default="sum",
                         choices=["sum", "average", "fixed_weighted",
                                  "variable_weighted", "time_varying", "median",
                                  "min", "max"])
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--warmup_epochs", type=int, default=15)
    args = parser.parse_args()
    train(
        csv_path=args.csv_path,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        pred_len=args.pred_len,
        loss_mode=args.loss_mode,
        alpha=args.alpha,
        warmup_epochs=args.warmup_epochs,
    )
