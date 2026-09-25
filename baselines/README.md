# Baseline Comparisons

This folder tracks external baseline models compared against our TF-LLM implementation. Each baseline uses the original authors' official code, run by us on ETTh1 (horizon 96) for direct comparison — we do not reimplement these models ourselves. Each subfolder documents exactly what was run, any environment fixes required, and the resulting metrics.

| Baseline | Official repo | Test MSE | Test MAE |
|---|---|---|---|
| iTransformer | [thuml/iTransformer](https://github.com/thuml/iTransformer) | 0.3866 | 0.4046 |

See each subfolder's `notes.md` for the exact commands, fixes, and full logs.
