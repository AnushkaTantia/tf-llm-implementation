# iTransformer Baseline

**Source:** Official implementation, [thuml/iTransformer](https://github.com/thuml/iTransformer)
**Paper:** Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L., & Long, M. (2023). iTransformer: Inverted Transformers Are Effective for Time Series Forecasting. arXiv:2310.06625 (ICLR 2024 Spotlight).

We did not modify the model architecture or training logic. We ran the repository's own official ETTh1 script (`scripts/multivariate_forecasting/ETT/iTransformer_ETTh1.sh`), horizon-96 configuration only, on Google Colab (T4 GPU).

## Environment fixes required

1. **`requirements.txt` version pins failed to build** (`pandas==1.5.3`, `scikit-learn==1.2.2` had no pre-built wheel for Colab's Python version). Fix: stripped exact version pins, let pip resolve compatible versions against Colab's pre-installed packages. `torch` was excluded entirely from the install to avoid overwriting Colab's existing CUDA-enabled build.

2. **`AttributeError: np.Inf was removed in NumPy 2.0`** in `utils/tools.py` (`EarlyStopping.__init__`). Fix:
   sed -i 's/np\.Inf/np.inf/g' utils/tools.py
   
## Command run (horizon 96)

```bash
python run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model iTransformer \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --itr 1
```

Configuration matches the repository's own default script exactly — no hyperparameters were changed. Per the TF-LLM paper's own stated methodology (Section 4.2: *"we strictly followed the recommended settings of each original model for key parameters such as the input window (seq_len) and prediction length (pred_len)"*), we did not force iTransformer to use TF-LLM's own lookback (512); we used iTransformer's own standard lookback (96), matching how the original TF-LLM paper obtained its own baseline numbers.

## Result

Training stopped via early stopping (patience=3) after 7 epochs; best checkpoint from epoch 4.

**Test MSE: 0.3866**
**Test MAE: 0.4046**

Full log: see `run_log.txt` in this folder.
