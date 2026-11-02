# Stage 3 report and reassessment of the central premise (Nov 2, 2026)

**Scope.** This update covers checklist stage **S3 (does TFB help?)** of the final research plan. The protocol and decision rule were declared in `configs/stage3_protocol.json` before any run. The plan says: *"If TFB does not help, stop and reassess the central premise before the scheduler comparisons."* TFB does not help, so **S4 has not been started**. This document gives the evidence, a diagnosis of how TFB hurts, and the options for the reassessment that needs the supervisor's decision.

---

## 1. What the evidence establishes

### 1.1 Run completeness and integrity
- **Runs:** all 50 Stage 3 runs completed (5 fixed α × 10 predeclared seeds). The α = 0 arm is the committed Stage 2 result.
- **Code identity:** the run script checked that `src/` is identical to the Stage 2 commit `b32fe77`. It would have stopped otherwise.
- **Configuration:** run configurations differ from Stage 2 only in α.
- **Tests:** the preflight passed 19/19 implementation checks on the Kaggle GPU with real GPT-2.
- **Analysis:** `analysis/stage3_analysis.py` was run unchanged, as committed at S2 exit, and is byte-identical to the declared version.

### 1.2 Results (ETTh1, H = 96, frozen forecaster F, 10 seeds)

| α | best val MSE | test MSE | test RMSE | SMAPE % | Δ test RMSE vs α=0 [95% CI] | seeds better | Holm p |
|---|---|---|---|---|---|---|---|
| **0** | **0.7314 ± 0.0112** | **0.3761 ± 0.0028** | **0.6132** | **37.98** | – | – | – |
| 0.01 | 0.7394 ± 0.0121 | 0.3796 ± 0.0055 | 0.6161 | 38.10 | +0.0029 [+0.0015, +0.0049] | 0/10 | 0.010 |
| 0.03 | 0.7750 ± 0.0159 | 0.3955 ± 0.0055 | 0.6289 | 38.83 | +0.0156 [+0.0137, +0.0175] | 0/10 | 0.010 |
| 0.1 | 0.7823 ± 0.0109 | 0.3955 ± 0.0051 | 0.6289 | 38.85 | +0.0156 [+0.0139, +0.0173] | 0/10 | 0.010 |
| 0.3 | 0.7851 ± 0.0149 | 0.3987 ± 0.0061 | 0.6314 | 39.33 | +0.0182 [+0.0156, +0.0208] | 0/10 | 0.010 |
| 1.0 | 0.7858 ± 0.0178 | 0.3994 ± 0.0051 | 0.6320 | 39.40 | +0.0188 [+0.0164, +0.0211] | 0/10 | 0.010 |

(Wilcoxon two-sided; Holm p = 0.0098 is the smallest value attainable with 10 pairs and 5 comparisons; paired permutation p = 0.002 for every arm.)

### 1.3 Decision under the declared rule
- **α\* (validation only) = 0.** Every positive weight has a higher mean validation error.
- **H1 is NOT SUPPORTED.** The rule requires α\* > 0.
- The opposite is established: **every fixed α > 0 is worse than α = 0 on all 10 seeds**, significantly, with a monotone dose-response. The harm is about 1% in test MSE at α = 0.01 and saturates at 5–6% for α ≥ 0.03.
- **Fixed α as a meaningful baseline:** the calibrated fixed weight is α = 0 (no TFB).

### 1.4 How TFB hurts (`STAGE3_DIAGNOSTICS.md`, `fig_s3_diagnostics.png`; descriptive, no selection)

1. **The harm is present from the first epoch.** Validation MSE after epoch 1 is +0.007 at α = 0.01 and +0.03 to +0.04 at α ≥ 0.03, and it stays higher at every later epoch. It is not a late-training or overfitting effect.
2. **It is a representation effect, not underfitting.** At α = 0.01 the training forecasting loss is identical to α = 0 (0.3273 at epoch 5), yet validation error is higher. TFB changes what the encoders represent in a way that generalises worse.
3. **TFB does not regularise here.** The generalisation gap at the selected epoch grows from 0.403 (α = 0) to 0.447 (α = 0.1), and best epochs move earlier (5.2 → 2.5 at α = 1).
   - This contradicts the design assumption of OA-HMM, which raises α to α_over = 0.5 when it detects overfitting, assuming TFB acts as a regulariser.
4. **Alignment and accuracy are opposed.**
   - Across all 60 runs, a lower TFB loss (better time–frequency alignment) goes with a higher validation error: Spearman ρ = −0.73, p = 3e−11 (ρ = −0.57 within α > 0).
   - Without the objective the views stay unaligned: the TFB loss is 4.15–4.23 throughout training at α = 0.
5. **Where the error goes.**
   - Mostly to HUFL and MUFL, the high-variance useful-load variables, and slightly more at later forecast steps.
   - Oil temperature is almost unaffected.
6. **Candidate mechanism (hypothesis, not yet tested).** The frequency encoder sees the FFT magnitude, which is invariant to time shifts. Pulling the time-domain embedding towards it may remove the temporal-position information the forecaster needs. Point 1 (harm from epoch 1) and point 4 (alignment opposed to accuracy) are consistent with this, but they do not prove it.

### 1.5 Consistency with earlier evidence
- Pre-freeze 4-seed experiment on the RevIN-corrected model (paper Appendix): α = 0 had the best rank of eight weighting strategies.
- Stage 1: GPT-2 does not improve F (3/3 seeds).
- Together: neither distinctive component of the time-frequency LLM design of Zhang et al. (Y. Zhang, Z. Yu, M. Dai, Y. Sun, T. Tan, "TF-LLM: Enhanced time series analysis with time-frequency large language models," *Neural Networks*, vol. 199, 108687, 2026) improves forecasting in our controlled reproduction: neither the TFB alignment nor the LLM backbone.

### 1.6 Additional reproducibility evidence (correction to an earlier statement)
- The results zip contains a second, complete Kaggle run of Stage 2 (all 10 seeds, α = 0). Compared with the committed Stage 2 results:
  - the same selected epoch for **10/10** seeds
  - test MSE within **2.2 × 10⁻⁵** (relative < 6 × 10⁻⁵)
- Seeds 42/123/456 of Stage 1 vs Stage 2 agree to within 7 × 10⁻⁶, with the same epoch.
- **Correction:** earlier documents said "bit-identical". That was not exactly right: GPU arithmetic is not bit-deterministic. The correct statement is "same selected epoch, test MSE agreeing to within 2.2 × 10⁻⁵". The paper draft, the mid-term report and the slides have been corrected. The conclusions are unchanged.
- The S3 analysis uses the **committed** Stage 2 results, as declared. Using the second run instead changes no number in the decision table beyond the 4th decimal.

---

## 2. What remains unresolved
1. **Scope of the negative result.** It is established for F, ETTh1, H = 96 and fixed α. It is **not** established for:
   - the GPT-2 model E, the setting in which TFB was proposed
   - other horizons
   - other datasets
   - our substitute balance loss vs the paper's unparseable Eq. 3 (documented deviation)
2. **Mechanism.** The shift-invariance hypothesis (§1.4, point 6) is untested.
3. **Whether any schedule can extract value from TFB.**
   - In principle, a time-varying α could help where every fixed α fails, for example TFB only in the first epochs.
   - The evidence makes this unlikely: the harm starts at epoch 1, appears even at α = 0.01, and TFB widens the gap.
   - Not tested.
4. **The paper's central contribution** (the adaptive TFB scheduler) depends on H1. As planned, it cannot be supported as stated.

## 3. Checklist changes
- **S3 → COMPLETE (verified).** H1 not supported; α\* = 0. Results, analysis, diagnostics and decision record are committed by `Colab_close_S3.ipynb` (tag `s3-tfb-validation`).
- **S4 → BLOCKED by the plan's rule,** pending the supervisor's reassessment. Not started.
- **S5–S8:** unchanged, and dependent on the reassessment.
- **New item R1 (reassessment of the central premise):** OPEN, owner: supervisor.

## 4. Is the stage complete?
**Yes.** S3's exit criterion is met: "we know whether TFB has a measurable benefit and whether fixed α is a meaningful baseline". TFB has a measurable **cost**, and the calibrated fixed baseline is α = 0.

## 5. Reassessment: options for the supervisor's decision

GPU estimates for F: about 13.5 min per run on one T4, two in parallel; Kaggle quota about 30 GPU-h per week.

**Option R: reframe around the evidence (recommended).**
- **Paper:** a rigorous controlled study of when time–frequency alignment helps LLM-style forecasters.
- **Contributions:**
  1. a significant, dose-dependent negative result for TFB under a predeclared protocol
  2. its mechanism: harm from epoch 1, a representation effect rather than underfitting, alignment opposed to accuracy, and no regularisation
  3. the LLM backbone is not needed (S1)
  4. the reproducible protocol and controls
- **Required before writing (H4-style scope checks, each predeclared like S3):**
  - (a) ETTh2, F, α ∈ {0, 0.01, 0.1}, 10 seeds: ≈ 7 GPU-h
  - (b) ETTh1 H = 336, α ∈ {0, 0.1}, 10 seeds: ≈ 5 GPU-h
  - (c) E (GPT-2), α ∈ {0, 0.1}, 5 paired seeds: ≈ 8 GPU-h. This is the original TFB setting and matches Option 3, step 4.
  - (d) mechanism probe: time-shift sensitivity of the TD and FD embeddings at α = 0 vs α > 0. This needs two short runs with checkpoints plus CPU analysis, ≈ 1 GPU-h.
- **Cost:** ≈ 21 GPU-h, about one week.
- **Role of OA-HMM:** retained as an analysis instrument (its regime inference on real trajectories) or dropped. That is the supervisor's choice.

**Option S: continue to S4 as planned, with α\* = 0 as the comparator.**
- Tests whether any adaptive schedule beats no TFB.
- **Cost:** ≈ 10 GPU-h.
- **Expected value: low.**
  - The harm appears at the smallest weight and from the first epoch, so a scheduler can at best match α = 0 by learning to switch TFB off.
  - OA-HMM's overfitting response (raise α) is contradicted by §1.4, point 3, so its mapping would have to be redesigned and re-declared first.

**Option M: revise the auxiliary objective (a new method), then repeat S3.**
- Design an alignment that keeps temporal-position information, for example:
  - a phase-aware frequency input instead of the magnitude only, or
  - a stop-gradient into the time-domain encoder
- Re-freeze it and re-run S3 with the new objective.
- **Upside:** the highest, a positive method paper.
- **Cost:** a new implementation/verification cycle (S1-style) plus ≈ 11 GPU-h per S3 grid; about 2 weeks.
- **Risk:** it may fail again.
- **Condition:** it must be declared in advance to avoid tuning on results. It is best decided after probe R(d) confirms or refutes the mechanism.

**Recommendation.**
1. Option R(a)–(d) first: one week, and it decides the scope and mechanism.
2. Then choose between writing Option R as the paper, or Option M if the mechanism probe shows a clear, fixable cause.
3. Option S is not recommended as the main line.

The title ("LLM-Based", "Adaptive … Scheduling") must follow the outcome (S8).

**Decisions needed from the supervisor:**
- **D-S3-1:** Option R, S or M (or a combination).
- **D-S3-2:** the role of OA-HMM in the paper.
- **D-S3-3:** confirm that the S3 negative result is reported as a primary finding (the plan says "preserve negative results").

## 6. Next assignment (after the decision)
- Write the predeclared protocol for the chosen option (as for S3).
- Implement and test it, deliver the Kaggle cell with its push/close notebook, then analyse with the declared rule.
- No experiment starts before the supervisor's decision.

## Files
- `results/stage3/`: all run outputs (`runs/*/result.json`, `epoch_log.json`, `run_config.json`, logs), `summary.txt`, `results.csv`, `protocol.json`, preflight log.
- `results/stage3/analysis/`:
  - `STAGE3_REPORT.md` (declared analysis, verbatim output)
  - `STAGE3_DIAGNOSTICS.md`
  - `fig_s3_alpha.png`, `fig_s3_diagnostics.png`
  - `s3_decision.json`, `s3_diagnostics.json`
  - this file
- `results/stage2_rerun/`: the second complete Stage 2 session (reproducibility evidence).
- `analysis/stage3_diagnostics.py`: the diagnostics script.
- Paper draft updated: Section V-D (Stage 3 results, Table VIII, Figs. 4–5), plus the reproducibility correction.
