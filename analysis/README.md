# analysis/ -- offline analysis scripts (CPU)

| Script | Stage | Purpose |
|---|---|---|
| `data_audit.py` | S2 | independent ETT loader (separate from src/data/loaders.py) |
| `stage2_controls.py` | S2 | controls on the frozen protocol: deterministic predictors, DLinear and RevIN-Linear (10 seeds), independent metrics, loader agreement |
| `stage2_analysis.py` | S2 | corrected forecaster vs controls, independent metric verification, temporal-learning tests, curves, seeds, validation/test shift |
| `stage3_analysis.py` | S3 | alpha* selection (validation only), paired tests, decision rule of configs/stage3_protocol.json |
| `stage3_diagnostics.py` | S3 | descriptive diagnostics of how the TFB weight changes training (no selection) |

Data: set `ETT_DATA_DIR` to the folder containing ETTh1.csv (default: `../data`).
