# TimesNet Baseline

**Source:** [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library) (official repository)
**Paper:** Wu et al., "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis," arXiv:2210.02186, 2022.

## Setup
- Task: `long_term_forecast`, dataset ETTh1, horizon 96, seq_len 96 (model's own standard lookback — no lookback override, per the paper's per-model methodology)
- Ran the script's own long-term-forecasting ETTh1 config unmodified

## Environment fixes required
1. `ModuleNotFoundError: No module named 'patoolib'` — the pip package installs as `patool` but imports as `patoolib`. This import only serves the (irrelevant to ETTh1) M4 dataset loader but runs at module load regardless. Fixed via `pip install patool`.
2. `ModuleNotFoundError: No module named 'sktime'` at `data_provider/data_loader.py` — only needed for the (irrelevant) UEA classification loader. Fixed via `pip install sktime`.

Neither fix touched any modeling logic.

## Result
Full authored 10-epoch budget, no early stopping triggered (validation loss kept decreasing through epoch 10).

**Test MSE: 0.3891  Test MAE: 0.4120**
