# Autoformer Baseline

**Source:** Official implementation, [thuml/Autoformer](https://github.com/thuml/Autoformer)
**Paper:** Wu, H., Xu, J., Wang, J., & Long, M. (2021). Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting. NeurIPS 2021.

We did not modify the model architecture or training logic. The repository's own official scripts (`scripts/ETT_script/Autoformer_ETTh1.sh`) only cover horizons 24/48/168/336/720 (the older Informer-era convention); we constructed the horizon-96 command ourselves by taking the script's first block and changing only `--model_id` and `--pred_len`, keeping every other setting exactly as authored.

## Environment fixes required

1. **`np.Inf` deprecation** in `utils/tools.py` (same issue as iTransformer). Fix:
2. **`torchvision --no-deps` install initially broke `reformer_pytorch`** by blocking its own required sub-dependencies (`local_attention`, etc.). Fix: installed `reformer_pytorch` and the other requirements normally (with dependencies), and restricted `--no-deps` to `torchvision` only.

## Command run (horizon 96)

```bash
python run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model Autoformer \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1
```

## Result

Training stopped via early stopping (patience=3) after 4 epochs; best checkpoint from epoch 1 (validation loss increased every epoch thereafter).

**Test MSE: 0.4300**
**Test MAE: 0.4442**

Full log: see `run_log.txt` in this folder.
