# Stage 3 report: does the auxiliary TFB objective help? (H1)

Protocol: `configs/stage3_protocol.json`, declared before any run. Frozen forecaster F; the only thing that changes between arms is the fixed weight alpha of the TFB loss. Arms: alpha in [0.0, 0.01, 0.03, 0.1, 0.3, 1.0]; seeds with every arm complete: 10 of 10 ([1, 7, 21, 42, 100, 123, 456, 789, 999, 2024]). Only complete paired sets are analysed.

## Results by arm (mean ± sd over seeds)

| alpha | best val MSE | test MSE | test RMSE | test MAE | test SMAPE % | best epoch | final train TFB loss |
|---|---|---|---|---|---|---|---|
| 0.0 | 0.7314 ± 0.0112 | 0.3761 ± 0.0028 | 0.6132 | 0.3998 | 37.98 | 5.2 | 4.226 |
| 0.01 | 0.7394 ± 0.0121 | 0.3796 ± 0.0055 | 0.6161 | 0.4023 | 38.10 | 5.2 | 4.129 |
| 0.03 | 0.7750 ± 0.0159 | 0.3955 ± 0.0055 | 0.6289 | 0.4131 | 38.83 | 4.8 | 3.507 |
| 0.1 | 0.7823 ± 0.0109 | 0.3955 ± 0.0051 | 0.6289 | 0.4124 | 38.85 | 4.7 | 3.391 |
| 0.3 | 0.7851 ± 0.0149 | 0.3987 ± 0.0061 | 0.6314 | 0.4157 | 39.33 | 3.3 | 3.204 |
| 1.0 | 0.7858 ± 0.0178 | 0.3994 ± 0.0051 | 0.6320 | 0.4166 | 39.40 | 2.5 | 3.122 |

(final train TFB loss: the epoch-mean TFB loss at the selected epoch; at alpha = 0 it is computed but not optimised, so the difference shows whether training actually reduces the auxiliary objective.)

## Validation-selected alpha*

Rule (declared): the arm with the lowest mean best-validation MSE over seeds, alpha = 0 included. Test results are not used.

| alpha | mean best-validation MSE |
|---|---|
| 0.0 | 0.7314  ← alpha* |
| 0.01 | 0.7394 |
| 0.03 | 0.7750 |
| 0.1 | 0.7823 |
| 0.3 | 0.7851 |
| 1.0 | 0.7858 |

**alpha\* = 0.0**

## Primary test: alpha* vs alpha = 0 (paired by seed, test RMSE)

alpha* = 0: validation does not favour any positive TFB weight. **H1 is not supported** by the declared decision rule (it requires alpha* > 0).

## Secondary: every fixed alpha vs alpha = 0 (paired, test RMSE, Holm-corrected)

| alpha | mean diff RMSE | 95% CI | better on | Wilcoxon p | Holm p | permutation p |
|---|---|---|---|---|---|---|
| 0.01 | +0.00286 | [+0.00154, +0.00488] | 0/10 | 0.0020 | 0.0098 | 0.0019 |
| 0.03 | +0.01564 | [+0.01373, +0.01751] | 0/10 | 0.0020 | 0.0098 | 0.0021 |
| 0.1 | +0.01563 | [+0.01389, +0.01734] | 0/10 | 0.0020 | 0.0098 | 0.0016 |
| 0.3 | +0.01816 | [+0.01560, +0.02081] | 0/10 | 0.0020 | 0.0098 | 0.0023 |
| 1.0 | +0.01876 | [+0.01643, +0.02107] | 0/10 | 0.0020 | 0.0098 | 0.0025 |

## Decision (rule declared in configs/stage3_protocol.json)

- H1 (TFB improves test error over alpha = 0): **NOT SUPPORTED**
- Practically meaningful (improvement > seed sd 0.0028 MSE): **no**
- Calibrated fixed alpha for S4: **alpha* = 0.0**
- Per the research plan: the scheduler study (S4) is **not started**; the central premise must be reassessed with the supervisor (see the options in the Stage 3 summary).

Figure: `fig_s3_alpha.png` (validation and test MSE against alpha; dots = seeds, bars = sd).
