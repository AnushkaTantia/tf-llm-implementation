# Stage 2, second session (reproducibility evidence only)

A second complete Kaggle run of the Stage 2 configuration (frozen forecaster F, alpha = 0, the 10 predeclared seeds),
returned together with the Stage 3 outputs. It is NOT used in any analysis: the Stage 3 alpha = 0 arm is the committed
`results/stage2`. Compared with `results/stage2`: same selected epoch for 10/10 seeds; test MSE within 2.2e-5
(relative < 6e-5), i.e. reproducible up to non-deterministic GPU arithmetic. Forecast files are not included.
