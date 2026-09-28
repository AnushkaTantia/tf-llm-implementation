# Baseline Reproductions

Independent reproductions of published baseline models on ETTh1 (horizon 96), each run using the model's own official public code, unmodified, with the model's own standard hyperparameters and lookback (per the TF-LLM paper's own stated per-model lookback methodology). No baseline model was reimplemented — see each baseline's own `notes.md` for the exact repository, environment fixes, and command used.

| Baseline | Official repo | Test MSE | Test MAE |
|---|---|---|---|
| iTransformer | [thuml/iTransformer](https://github.com/thuml/iTransformer) | 0.3866 | 0.4046 |
| Autoformer | [thuml/Autoformer](https://github.com/thuml/Autoformer) | 0.4300 | 0.4442 |
| PatchTST | [yuqinie98/PatchTST](https://github.com/yuqinie98/PatchTST) | 0.3816 | 0.4051 |
| TimesNet | [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library) | 0.3891 | 0.4120 |

See `<baseline>/notes.md` for methodology and environment fixes, and `<baseline>/run_log.txt` for the full training log, for each baseline.
