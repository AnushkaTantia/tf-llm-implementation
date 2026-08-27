# TF-LLM: Time-Frequency LLM for Time Series Forecasting

An independent implementation of **TF-LLM** ([Zhang et al., *Neural Networks*, 2026](https://doi.org/10.1016/j.neunet.2026.108687)), built directly from the paper's description with no reference code available. TF-LLM fuses time-domain and frequency-domain representations of a time series through a Time-Frequency Balance (TFB) module, then feeds the result into a frozen, pre-trained GPT-2 backbone via prompt learning.

This repository implements the **forecasting** task on the **ETTh1** benchmark (lookback 512, horizon 96), and investigates a specific ambiguity in the paper's loss formulation through an 8-strategy ablation study.

---

## Design Decisions and Documented Ambiguities

The paper does not provide reference code, and three aspects of its specification are genuinely ambiguous or internally inconsistent. Rather than silently picking an interpretation, each is documented and, where it materially affects results, resolved empirically:

1. **The paper's Eq. 4 never specifies how the forecasting task loss should combine with its own Time-Frequency Balance loss** ($\mathcal{L}_{TFB}$) in the unified training objective — only how $\mathcal{L}_{TFB}$'s own internal components combine. This repository implements and empirically compares **eight** candidate combination strategies (see [Results](#results) below).
2. **The paper's Table 11 and Section 4.2 prose disagree** on the learning rate (1e-4 vs. 3e-4). Table 11 is followed as the more authoritative source.
3. **The paper's Eq. 3 (balance loss) is not parseable into a computable form as printed.** A margin-based triplet loss is substituted, preserving the intended semantics.

## Architecture

```
                       Input time series
                    (7 features, seq_len=512)
                              │
                ┌─────────────┴─────────────┐
                │                           │
     ┌──────────▼──────────┐    ┌───────────▼───────────┐
     │      TD Encoder      │    │      FD Encoder        │
     │   patch (P=16) +     │    │   rFFT + RevIN +       │
     │   contrastive z_T    │    │   contrastive z_F      │
     └──────────┬──────────┘    └───────────┬───────────┘
                │                           │
                └─────────────┬─────────────┘
                              │
                ┌─────────────▼─────────────┐
                │   Time-Frequency Balance    │
                │        (TFB) module         │
                └─────────────┬─────────────┘
                              │
                ┌─────────────▼─────────────┐
                │  Fused representation +     │
                │  learnable prompt tokens    │
                └─────────────┬─────────────┘
                              │
                   ┌──────────▼──────────┐
                   │     Frozen GPT-2      │
                   │       backbone         │
                   └──────────┬──────────┘
                              │
                   ┌──────────▼──────────┐
                   │   Forecasting head     │
                   └──────────┬──────────┘
                              │
                      96-step forecast
```

Implemented channel-independently, per the paper's Section 3.3 tensor-shape convention.

## Results

**Primary result** — full 50-epoch training, evaluated on the held-out test set:

| Source | MSE | MAE |
|---|---|---|
| TF-LLM (paper, Table 1, ETTh1, horizon 96) | 0.339 | 0.373 |
| This implementation, initial (unweighted sum, 50 ep.) | 1.437\* | — |
| **This implementation, time-varying loss weighting (50 ep.)** | **1.143** | **0.827** |

\*Validation MSE — the initial run predated the addition of test-set evaluation to the training loop.

**Loss-combination ablation** (short trials, 5 epochs, test set):

| Loss mode | Test MSE | Test MAE |
|---|---|---|
| Sum (baseline) | 1.288 | 0.941 |
| Fixed weighted (α=0.3) | 1.345 | 0.971 |
| Average | 1.200 | 0.876 |
| Variable (learnable) weighted | 1.154 | 0.839 |
| Median ≡ Minimum\*\* | 1.126 | 0.773 |
| Time-varying (short trial, 3 ep.) | 1.139 | 0.840 |
| **Time-varying (full 50 ep., primary result)** | **1.143** | **0.827** |

\*\*Median and minimum are numerically identical for a two-term loss under PyTorch's median convention (returns the lower of two middle values, unlike NumPy's mean-based convention).

**Key finding:** an unweighted sum of the task loss and $\mathcal{L}_{TFB}$ lets the latter (which was consistently 2–3× larger than the task loss during training) dominate gradient updates, degrading forecasting accuracy. A time-varying schedule — linearly annealing the TFB loss weight from 0 to 0.3 over the first 15 epochs — improves test MSE by ~20% over the naive sum. A short-trial ablation across the remaining strategies complicates this picture: the median/minimum strategy outperforms time-varying's own short trial, motivating a fully epoch-matched comparison as the primary open question for future work.

## Repository structure

```
src/
├── data/loaders.py          # ETTh1 dataset, standard 12/4/4-month split, z-score normalization
├── models/
│   ├── td_encoder.py        # Time-domain encoder + contrastive augmentation
│   ├── fd_encoder.py        # Frequency-domain encoder (rFFT + RevIN) + contrastive augmentation
│   ├── tfb_module.py        # Time-Frequency Balance module + margin-based balance loss
│   ├── prompt_builder.py    # Static/dynamic prompt templates, digit-tokenization workaround
│   ├── tf_llm.py             # Channel-independent GPT-2 backbone integration
│   └── forecast_head.py     # Forecasting head
├── losses/contrastive.py    # NT-Xent contrastive loss
└── train.py                  # Training loop, evaluation, 8-mode loss-combination switch
```

## Setup

```bash
pip install -r requirements.txt
```

Requires `torch>=2.0.0`, `transformers>=4.35.0`. GPU strongly recommended (GPT-2 forward/backward on CPU is impractically slow for this pipeline).

## Usage

```python
from train import train

forecaster, tfb, results = train(
    csv_path='data/ETTh1.csv',
    checkpoint_dir='checkpoints',
    device='cuda',
    epochs=50,
    batch_size=8,
    loss_mode='time_varying',   # sum | average | fixed_weighted | variable_weighted
                                 # | time_varying | median | min | max
    alpha=0.3,
    warmup_epochs=15,
)
```

`train()` builds the model, runs training with the selected loss-combination strategy, evaluates on the held-out test set using the best validation checkpoint, and prints a direct comparison against the paper's reported numbers.

## Reference

This repository implements the method described in the following paper:
```bibtex
@article{zhang2026tfllm,
  title={TF-LLM: Enhanced time series analysis with time-frequency large language models},
  author={Zhang, Yuhang and Yu, Zitong and Dai, Mingtong and Sun, Yue and Tan, Tao},
  journal={Neural Networks},
  volume={199},
  pages={108687},
  year={2026}
}
```

## Acknowledgments

Implemented as part of a Summer Industrial Research Experience (SIRE), National Institute of Technology Rourkela.
