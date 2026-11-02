# Stage 2 report: credible baseline

Forecaster: `F_flatten_noLLM_v1` (frozen at the S1 exit), alpha = 0, ETTh1, lookback 512, horizon 96. Seeds (10): [1, 7, 21, 42, 100, 123, 456, 789, 999, 2024].

## B. Independent verification of the test metrics
Test metrics recomputed from the saved forecasts with the independent loader (audit/data_audit.py) and independent numpy metric code, compared with the values reported by `src/metrics.py`. Largest relative difference over all seeds (forecasts are stored as float16, which bounds the attainable agreement):

| metric | max relative difference |
|---|---|
| mse | 7.63e-07 |
| rmse | 3.81e-07 |
| mae | 4.53e-07 |
| smape | 8.78e-07 |
| mape | 3.34e-06 |

Verdict: agreement (tolerance 0.2 %).

## A. Corrected forecaster vs controls (test split)
| Predictor | runs | MSE mean ± sd [95% CI] | RMSE | MAE | SMAPE % |
|---|---|---|---|---|---|
| Corrected forecaster (F) | 10 seeds | 0.3761 ± 0.0028 [0.3745, 0.3778] | 0.6132 | 0.3998 | 37.98 |
| DLinear (same protocol) | 10 seeds | 0.3742 ± 0.0044 [0.3721, 0.3771] | 0.6117 | 0.3995 | 37.64 |
| RevIN_Linear (same protocol) | 10 seeds | 0.3737 ± 0.0016 [0.3730, 0.3747] | 0.6113 | 0.3983 | 37.71 |
| seasonal_naive_24h | deterministic | 0.5122 ± 0.0000 | 0.7157 | 0.4333 | 38.72 |
| seasonal_naive_168h | deterministic | 0.6570 ± 0.0000 | 0.8105 | 0.5086 | 44.61 |
| window_mean | deterministic | 0.7086 ± 0.0000 | 0.8418 | 0.5730 | 48.19 |
| last_value | deterministic | 1.2944 ± 0.0000 | 1.1377 | 0.7132 | 54.83 |
| train_mean | deterministic | 1.1099 ± 0.0000 | 1.0535 | 0.7960 | 61.96 |

Relative to seasonal naive (24 h) the corrected forecaster lowers test MSE by 26.6 %; relative to the window mean by 46.9 %; its difference to DLinear is +0.0018 MSE (+0.5 %).

## D. Does the forecaster learn temporal patterns rather than the average?
Level-removed skill: forecast and target minus each window's input mean, so a predictor that only matches the level scores 0 and a perfect forecaster scores 1.

| Predictor | level-removed R² | mean within-window correlation |
|---|---|---|
| Corrected forecaster (mean over seeds) | 0.469 (min 0.461) | 0.590 |
| DLinear (seed 42) | 0.476 | 0.575 |
| Seasonal naive 24 h | 0.277 | 0.489 |
| Window mean | 0.000 | (constant) |
| RevIN-Linear (seed 42) | 0.474 | 0.577 |

Similarity to the linear control: mean squared difference between the forecaster's (seed-averaged) forecast and the RevIN-Linear forecast is 0.0221, against a RevIN-Linear test MSE of 0.3729 (ratio 0.06; near 0 would mean the forecaster reproduces the linear map).

Input-order test: with the 32 input patches presented in a shuffled order, test MSE rises by a factor 2.12 (range 2.09-2.15) over the normal order.
Per-step test MSE (mean over seeds): step 1 0.128, step 24 0.336, step 48 0.370, step 96 0.421; seasonal naive 0.417/0.421/0.494/0.604. The forecaster is better than seasonal naive at 96 of 96 steps.

| Variable | forecaster MSE | DLinear | seasonal naive |
|---|---|---|---|
| HUFL | 0.7705 | 0.7370 | 0.9696 |
| HULL | 0.2159 | 0.2165 | 0.3080 |
| MUFL | 0.7962 | 0.7602 | 1.0085 |
| MULL | 0.1729 | 0.1767 | 0.2543 |
| LUFL | 0.5021 | 0.5205 | 0.7829 |
| LULL | 0.1195 | 0.1235 | 0.1909 |
| OT | 0.0553 | 0.0661 | 0.0715 |

Verdict: the forecaster predicts within-window structure, not only the level (criteria fixed in advance: level-removed R² > 0.2 for every seed and input-order factor > 1.1).

## C. Training and validation curves
Best epoch: median 5 (range 2-9); epochs run: median 10. Training task MSE falls from 0.386 (epoch 1) to 0.313 (last epoch); validation MSE is 0.766 after epoch 1 and 0.731 at its best, i.e. 4.5 % lower. Epoch-to-epoch validation change: median 0.0162, 90th percentile 0.0440. Generalization gap at the selected epoch: 0.403.

## E. Seed variability and reproducibility
Test MSE over 10 seeds: mean 0.3761, sd 0.0028 (CV 0.7 %), range 0.3726-0.3821. DLinear: sd 0.0044.

Same code commit, configuration and seed run twice (S1 and S2, separate Kaggle sessions):

| seed | S1 test MSE | S2 test MSE | difference | S1 best epoch | S2 best epoch |
|---|---|---|---|---|---|
| 42 | 0.3726 | 0.3726 | -0.0000 | 5 | 5 |
| 123 | 0.3782 | 0.3782 | +0.0000 | 6 | 6 |
| 456 | 0.3748 | 0.3748 | -0.0000 | 4 | 4 |

## F. Validation/test distribution shift
Validation MSE is higher than test MSE for every predictor (forecaster 0.735 vs 0.376; DLinear 0.652 vs 0.374; seasonal naive 0.827 vs 0.512), consistent with the documented shift (OT falls by 1.17 training SD between the validation and test periods). Across seeds, the rank correlation between validation and test MSE of the selected checkpoints is -0.21: a lower validation loss does not reliably indicate a lower test loss.

| Test month | forecaster MSE | seasonal naive | DLinear |
|---|---|---|---|
| 1 | 0.2689 | 0.4094 | 0.2727 |
| 2 | 0.3261 | 0.5012 | 0.3462 |
| 3 | 0.4349 | 0.6440 | 0.4496 |
| 4 | 0.4471 | 0.4945 | 0.4176 |
(forecaster column: mean forecast over seeds)

Figure: `fig_s2_baseline.png` (error by step, curves, example forecast).

## S2 exit assessment
- Better than seasonal naive and window mean on every seed: yes
- Comparable to DLinear reproduced under the same protocol: difference +0.0018 MSE
- Metrics independently verified: yes
- Temporal learning beyond the window average: yes
