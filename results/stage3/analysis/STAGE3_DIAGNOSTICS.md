# Stage 3 diagnostics: how the fixed TFB weight changes training

Descriptive analysis of the logged trajectories of all 60 runs (6 arms x 10 seeds). Nothing here selects a configuration; it explains the Stage 3 result.

## D1. Forecasting fit and generalisation per arm (mean over seeds)

| alpha | val MSE ep.1 | best epoch | train task loss ep.1 | train task loss ep.5 | train task at best | gap at best (val-train) | gap slope ep.1-5 | TFB loss ep.1 | TFB loss at best | alpha*TFB / task at best |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.7662 | 5.2 | 0.3859 | 0.3273 | 0.3288 | 0.4026 | +0.0108 | 4.181 | 4.226 | 0.000 |
| 0.01 | 0.7728 | 5.2 | 0.3863 | 0.3273 | 0.3287 | 0.4107 | +0.0114 | 4.146 | 4.129 | 0.126 |
| 0.03 | 0.8028 | 4.8 | 0.3892 | 0.3300 | 0.3355 | 0.4396 | +0.0120 | 3.817 | 3.507 | 0.314 |
| 0.1 | 0.8023 | 4.7 | 0.3905 | 0.3333 | 0.3356 | 0.4467 | +0.0125 | 3.616 | 3.391 | 1.011 |
| 0.3 | 0.7991 | 3.3 | 0.3921 | 0.3391 | 0.3586 | 0.4265 | +0.0134 | 3.511 | 3.204 | 2.683 |
| 1.0 | 0.7944 | 2.5 | 0.3938 | 0.3452 | 0.3695 | 0.4163 | +0.0161 | 3.402 | 3.122 | 8.443 |

## D2. Validation MSE by epoch, paired difference to alpha = 0 (mean over seeds; epochs all runs reached)

| alpha | ep 1 | ep 2 | ep 3 | ep 4 | ep 5 | ep 6 |
|---|---|---|---|---|---|---|
| 0.01 | +0.0066 | +0.0097 | +0.0108 | +0.0090 | +0.0091 | +0.0097 |
| 0.03 | +0.0366 | +0.0540 | +0.0465 | +0.0451 | +0.0450 | +0.0444 |
| 0.1 | +0.0360 | +0.0563 | +0.0520 | +0.0467 | +0.0500 | +0.0500 |
| 0.3 | +0.0329 | +0.0516 | +0.0570 | +0.0497 | +0.0525 | +0.0657 |
| 1.0 | +0.0282 | +0.0477 | +0.0582 | +0.0560 | +0.0617 | +0.0624 |

## D3. Where the error increases: test MSE, paired difference to alpha = 0 (mean over seeds)

| alpha | HUFL | HULL | MUFL | MULL | LUFL | LULL | OT | steps 1-24 | 25-48 | 49-72 | 73-96 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | +0.0090 | +0.0018 | +0.0096 | +0.0015 | +0.0020 | +0.0006 | +0.0002 | +0.0028 | +0.0033 | +0.0038 | +0.0042 |
| 0.03 | +0.0526 | +0.0076 | +0.0553 | +0.0057 | +0.0102 | +0.0029 | +0.0019 | +0.0187 | +0.0181 | +0.0197 | +0.0213 |
| 0.1 | +0.0513 | +0.0087 | +0.0528 | +0.0061 | +0.0099 | +0.0033 | +0.0039 | +0.0176 | +0.0178 | +0.0187 | +0.0236 |
| 0.3 | +0.0567 | +0.0111 | +0.0566 | +0.0093 | +0.0145 | +0.0046 | +0.0055 | +0.0239 | +0.0218 | +0.0202 | +0.0246 |
| 1.0 | +0.0586 | +0.0115 | +0.0572 | +0.0099 | +0.0153 | +0.0046 | +0.0065 | +0.0281 | +0.0239 | +0.0184 | +0.0230 |

## D4. Alignment vs accuracy

- Spearman correlation between the TFB loss at the selected epoch and the best validation MSE, all 60 runs: rho = -0.733 (p = 2.7e-11); alpha > 0 runs only: rho = -0.568 (p = 1.7e-05). Negative rho: runs whose time and frequency embeddings are more aligned (lower TFB loss) forecast worse.
- At alpha = 0 the TFB loss is computed but not optimised; it stays at 4.221 (range 4.148-4.230) throughout training: the forecasting objective alone does not align the views.

