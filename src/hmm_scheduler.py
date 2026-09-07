"""
hmm_scheduler.py

Hidden Markov Model-based scheduler for the TFB loss weight (alpha), as an
alternative to the fixed linear-warmup schedule used by loss_mode="time_varying".

DESIGN RATIONALE:

Three candidate emission signals were tested against real data extracted from
two completed 50-epoch training logs (training_log.txt = "sum" baseline,
time_varying_50ep_log.txt = "time_varying" run) before settling on this design:

1. l_tfb/task_loss ratio -- REJECTED. Contrary to the initial hypothesis, this
   ratio does NOT decrease as training stabilizes; in the time_varying run it
   actually *increases* late in training (2.8 -> 4.2) because task_loss keeps
   falling while l_tfb stays flat. Using this as a "stability" signal would
   mislabel the good late-training phase as "unstable".

2. task_loss epoch-to-epoch slope (raw and 5-epoch-windowed) -- REJECTED.
   Separation between "declining" and "plateaued" phases was weak even after
   smoothing (separation ratio ~0.59, below the ~1.0 threshold for reliable
   two-state discrimination via Gaussian emissions). Per-batch training loss
   is simply too noisy at this model/dataset scale.

3. Volatility (variance) of epoch-to-epoch VALIDATION MSE change -- ADOPTED.
   This gave by far the cleanest, most consistent separation across BOTH
   existing logs:
     - sum baseline:    early-epoch delta std=0.096  ->  late-epoch std=0.0075
                         (~13x drop -- settles into a stable regime)
     - time_varying run: early-epoch delta std=0.0079 -> late-epoch std=0.0436
                         (~5.5x rise -- destabilizes as alpha ramps to full
                         weight; consistent with that run's best checkpoint
                         occurring at epoch 8, well before alpha reaches 0.3
                         at epoch 15, and val_MSE degrading thereafter)

   Both logs independently support a genuine "stable" vs "unstable" validation
   regime, distinguished by the SPREAD (not the mean) of val_MSE fluctuation.
   This is therefore modeled as a two-state HMM with zero-mean Gaussian
   emissions of different variance per state.

PARAMETERS ARE FIXED (hand-set from the above data), NOT learned via EM/
Baum-Welch. This is a deliberate scope decision -- see project report,
Section III (Design Decisions), for the full justification. State inference
itself (which epoch is in which state) IS genuine Viterbi decoding on live
data; only the emission/transition parameters are fixed in advance.
"""

import math


# ---------------------------------------------------------------------------
# Fixed HMM parameters, calibrated from real training logs (see module
# docstring above). Update these constants if recalibrating against new data.
# ---------------------------------------------------------------------------

# State indices
STATE_UNSTABLE = 0   # validation performance still fluctuating meaningfully
STATE_STABLE = 1     # validation performance has settled

STATE_NAMES = {STATE_UNSTABLE: "unstable", STATE_STABLE: "stable"}

# Emission model: val_MSE epoch-to-epoch delta ~ N(0, sigma_state^2)
# Zero-mean for both states -- it is the SPREAD of the delta that
# distinguishes the states, not its average direction (see rationale above).
EMISSION_STD = {
    STATE_UNSTABLE: 0.075,   # informed by sum-baseline early phase (0.096)
                             # and time_varying late phase (0.044), averaged
                             # and rounded conservatively
    STATE_STABLE: 0.010,    # informed by sum-baseline late phase (0.0075)
                             # and time_varying early phase (0.0079)
}

# Transition matrix. Unlike the originally-planned one-directional
# ("absorbing") design, real data from both logs shows regimes can move
# in EITHER direction (sum baseline settles over time; time_varying
# destabilizes over time as alpha ramps up) -- so both directions are
# permitted, with a mild bias toward eventually stabilizing.
TRANSITION = {
    STATE_UNSTABLE: {STATE_UNSTABLE: 0.70, STATE_STABLE: 0.30},
    STATE_STABLE:   {STATE_UNSTABLE: 0.15, STATE_STABLE: 0.85},
}

# Initial state distribution: training always starts in the unstable regime.
INITIAL = {STATE_UNSTABLE: 1.0, STATE_STABLE: 0.0}

# State -> alpha mapping. STATE_STABLE maps to the same target alpha used by
# loss_mode="time_varying" (0.3), for a fair, apples-to-apples comparison of
# *how* alpha reaches its target value, not *what* the target value is.
ALPHA_BY_STATE = {
    STATE_UNSTABLE: 0.05,
    STATE_STABLE: 0.3,
}


def _log_gaussian_pdf(x, std):
    """Log-density of N(0, std^2) at x. Avoids underflow vs. raw pdf."""
    var = std * std
    return -0.5 * math.log(2 * math.pi * var) - (x * x) / (2 * var)


class HMMScheduler:
    """
    Tracks validation MSE across epochs and infers, via Viterbi decoding,
    whether training is currently in the "unstable" or "stable" regime,
    mapping that inferred state to a TFB loss weight (alpha).

    Usage (mirrors the interface needed by combine_losses() / train()):

        scheduler = HMMScheduler()
        ...
        # once per epoch, after computing that epoch's val_MSE:
        scheduler.observe(val_mse)
        alpha = scheduler.get_alpha()
    """

    def __init__(
        self,
        emission_std=None,
        transition=None,
        initial=None,
        alpha_by_state=None,
    ):
        self.emission_std = emission_std or EMISSION_STD
        self.transition = transition or TRANSITION
        self.initial = initial or INITIAL
        self.alpha_by_state = alpha_by_state or ALPHA_BY_STATE

        self.val_mse_history = []   # raw val_MSE per epoch, in order observed
        self.deltas = []            # epoch-to-epoch val_MSE deltas
        self.state_history = []     # inferred state per epoch (after Viterbi)

    def observe(self, val_mse):
        """
        Record a new epoch's validation MSE. Call once per epoch, after
        that epoch's validation evaluation completes.
        """
        self.val_mse_history.append(val_mse)
        if len(self.val_mse_history) >= 2:
            delta = self.val_mse_history[-1] - self.val_mse_history[-2]
            self.deltas.append(delta)
            self._run_viterbi()
        else:
            # first epoch: no delta yet, stay in initial state
            self.state_history.append(STATE_UNSTABLE)

    def _run_viterbi(self):
        """
        Runs Viterbi decoding over self.deltas (all observations so far) to
        find the most likely state sequence, and appends the most likely
        CURRENT (final) state to self.state_history.

        Re-running Viterbi over the full history each call is computationally
        trivial at this scale (at most 50 epochs) -- no need for an online/
        incremental variant.
        """
        states = [STATE_UNSTABLE, STATE_STABLE]
        T = len(self.deltas)

        # log-probabilities to avoid underflow over many epochs
        log_delta = [{} for _ in range(T)]   # log_delta[t][s] = best log-prob
        backpointer = [{} for _ in range(T)]

        # initialization (t=0)
        for s in states:
            emission_lp = _log_gaussian_pdf(self.deltas[0], self.emission_std[s])
            init_lp = math.log(self.initial[s] + 1e-12)
            log_delta[0][s] = init_lp + emission_lp
            backpointer[0][s] = None

        # recursion
        for t in range(1, T):
            for s in states:
                best_prev, best_lp = None, -math.inf
                for sp in states:
                    trans_lp = math.log(self.transition[sp][s] + 1e-12)
                    lp = log_delta[t - 1][sp] + trans_lp
                    if lp > best_lp:
                        best_lp, best_prev = lp, sp
                emission_lp = _log_gaussian_pdf(self.deltas[t], self.emission_std[s])
                log_delta[t][s] = best_lp + emission_lp
                backpointer[t][s] = best_prev

        # termination: best final state
        best_final_state = max(states, key=lambda s: log_delta[T - 1][s])

        # backtrack to get full path (only the current/final state is
        # actually needed by get_alpha(), but the full path is reconstructed
        # for logging/diagnostic purposes)
        path = [None] * T
        path[T - 1] = best_final_state
        for t in range(T - 2, -1, -1):
            path[t] = backpointer[t + 1][path[t + 1]]

        # state_history[0] is the epoch-1 placeholder already appended in
        # observe(); path corresponds to epochs 2..T+1, so extend accordingly
        self.state_history = self.state_history[:1] + path

    def get_state(self):
        """Returns the most recently inferred state (int)."""
        return self.state_history[-1] if self.state_history else STATE_UNSTABLE

    def get_state_name(self):
        return STATE_NAMES[self.get_state()]

    def get_alpha(self):
        """Returns the alpha value corresponding to the current inferred state."""
        return self.alpha_by_state[self.get_state()]

    def history_summary(self):
        """Returns a list of (epoch, val_mse, state_name) for logging/plots."""
        return [
            (i + 1, self.val_mse_history[i], STATE_NAMES[self.state_history[i]])
            for i in range(len(self.val_mse_history))
        ]


if __name__ == "__main__":
    # Quick standalone sanity check (per implementation plan Step 3):
    # feed a synthetic sequence that starts volatile and settles down,
    # confirm the inferred state sequence transitions sensibly.
    print("Sanity check 1: volatile -> stable synthetic sequence")
    synthetic_val_mse = [1.60, 1.50, 1.65, 1.48, 1.44, 1.445, 1.442, 1.444, 1.441, 1.443]
    sched = HMMScheduler()
    for v in synthetic_val_mse:
        sched.observe(v)
    for epoch, val_mse, state in sched.history_summary():
        print(f"  epoch {epoch}: val_MSE={val_mse:.4f}  state={state}  alpha={sched.alpha_by_state[STATE_STABLE if state=='stable' else STATE_UNSTABLE]}")

    print("\nSanity check 2: stable -> volatile synthetic sequence (e.g. destabilizing late)")
    synthetic_val_mse_2 = [1.40, 1.402, 1.399, 1.401, 1.40, 1.45, 1.55, 1.50, 1.62, 1.58]
    sched2 = HMMScheduler()
    for v in synthetic_val_mse_2:
        sched2.observe(v)
    for epoch, val_mse, state in sched2.history_summary():
        print(f"  epoch {epoch}: val_MSE={val_mse:.4f}  state={state}  alpha={sched2.alpha_by_state[STATE_STABLE if state=='stable' else STATE_UNSTABLE]}")
