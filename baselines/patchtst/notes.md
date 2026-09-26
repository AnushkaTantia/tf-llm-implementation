# PatchTST Baseline

**Source:** Official implementation, [yuqinie98/PatchTST](https://github.com/yuqinie98/PatchTST) (PatchTST_supervised)
**Paper:** Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. ICLR 2023.

We did not modify the model architecture or training logic. Ran the repository's own official ETTh1 script configuration (`scripts/PatchTST/etth1.sh`, horizon-96 iteration only). Note this repo's data path convention differs from iTransformer/Autoformer (`./dataset/ETTh1.csv` directly, not `./dataset/ETT-small/`), and PatchTST's own recommended lookback is 336, not 96 — consistent with the paper's stated per-model lookback methodology.

## Environment fixes required

1. **`np.Inf` deprecation**, found in two locations: `utils/tools.py` and `Formers/FEDformer/utils/tools.py`. Fix applied to both:
sed -i 's/np.Inf/np.inf/g' utils/tools.py
sed -i 's/np.Inf/np.inf/g' Formers/FEDformer/utils/tools.py
2. **`torch==1.11.0` pin excluded** from install, same as prior baselines, to preserve Colab's CUDA build.

## Command run (horizon 96)

```bash
python -u run_longExp.py \
  --random_seed 2021 \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_336_96 \
  --model PatchTST \
  --data ETTh1 \
  --features M \
  --seq_len 336 \
  --pred_len 96 \
  --enc_in 7 \
  --e_layers 3 \
  --n_heads 4 \
  --d_model 16 \
  --d_ff 128 \
  --dropout 0.3 \
  --fc_dropout 0.3 \
  --head_dropout 0 \
  --patch_len 16 \
  --stride 8 \
  --des 'Exp' \
  --train_epochs 100 \
  --itr 1 \
  --batch_size 128 \
  --learning_rate 0.0001
```

## Result

Ran full 100 epochs (script's own `patience=100` means early stopping essentially never triggers). Best validation loss at epoch 12; never improved upon in the remaining 88 epochs. Best checkpoint retained for final testing.

**Test MSE: 0.3816**
**Test MAE: 0.4051**
**RSE: 0.5858** (repo's own additional metric)

Full log: see `run_log.txt` in this folder.
