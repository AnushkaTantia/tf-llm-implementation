"""
kalman_scheduler.py

Kalman filter-based scheduler for the TFB loss weight (alpha), producing a
CONTINUOUS alpha value each epoch, in contrast to hmm_scheduler.py's discrete
two-state (unstable/stable) switch.

DESIGN RATIONALE:

The HMM scheduler (hmm_scheduler.py) answers: "is training currently stable
or unstable?" and picks one of two fixed alpha values accordingly. This is a
different, complementary question from what a Kalman filter is naturally
suited to answer: "how surprising is this epoch's validation result, given
everything observed so far -- and can alpha scale smoothly with that degree
of surprise, rather than jumping between two preset levels?"

This also gives a principled second attempt at an idea rejected during HMM
design: using the trend of validation performance as a signal. That attempt
failed for the HMM because a simple windowed average of task_loss slope was
too noisy to cleanly separate two discrete states. A Kalman filter is
specifically built to denoise a noisy signal while tracking how it evolves,
which is the right tool for that job -- but instead of collapsing the result
into two labels, we let it drive alpha continuously.

THE FILTER

A standard scalar Kalman filter tracks a "true" underlying value (here: the
underlying validation MSE trend) from noisy per-epoch observations, via:

    Predict:  x_pred = x_hat            (random-walk model: no assumed drift
                                          direction, F=1, no control input)
              P_pred = P + Q

    Update:   innovation y = observation - x_pred
              S (innovation variance) = P_pred + R
              K (Kalman gain) = P_pred / S
              x_hat_new = x_pred + K * y
              P_new = (1 - K) * P_pred

The INNOVATION (y) is the gap between what the filter expected and what was
actually observed. Normalizing it by the filter's own expected uncertainty
(sqrt(S)) gives a standard, well-established quantity in Kalman filtering
--the normalized innovation -- used exactly for judging "is this surprising
or not", independent of the raw scale of the data.

PARAMETER CALIBRATION (fixed):

  R (measurement noise variance) = 0.0001  (std = 0.01)
      Set from the "stable" regime's observed epoch-to-epoch val_MSE
      spread in existing training logs (training_log.txt,
      time_varying_50ep_log.txt) -- the same figure that calibrated
      hmm_scheduler.py's STATE_STABLE emission std. This represents the
      filter's assumed noise floor: the smallest fluctuation level seen
      when training is behaving well.

  Q (process noise variance) = 0.0004  (std = 0.02)
      Set slightly larger than R, deliberately: this allows the filter to
      track genuine epoch-to-epoch improvement (observed early-training
      task_loss changes were on the order of 0.01-0.02 per epoch in prior
      logs) as real signal, while still flagging the much larger swings
      seen in "unstable" epochs (observed std ~0.075-0.096 in prior logs)
      as surprising, since those far exceed what Q+R alone would predict.

ALPHA MAPPING

The normalized innovation z = |y| / sqrt(S) is mapped through a smooth,
monotonically decreasing Gaussian-shaped function into alpha, bounded
between the same two endpoint values used by hmm_scheduler.py (0.05, 0.3)
for direct comparability:

    alpha = alpha_min + (alpha_max - alpha_min) * exp(-z^2 / (2*tau^2))

z near 0 (observation matched prediction closely) -> alpha near alpha_max.
z large (observation was a big surprise)          -> alpha near alpha_min.

tau controls how quickly alpha decays with surprise; calibrated via the
standalone sanity check below (see __main__ block) rather than guessed.
"""

import math


# ---------------------------------------------------------------------------
# Fixed parameters. See module docstring for full derivation.
# ---------------------------------------------------------------------------
R_MEASUREMENT_NOISE = 0.0001   # std 0.01, from "stable"-regime val_MSE spread
Q_PROCESS_NOISE = 0.0004       # std 0.02, allows tracking genuine drift

ALPHA_MIN = 0.05
ALPHA_MAX = 0.3
TAU = 1.5   # surprise scale; calibrated against synthetic sanity checks below


class KalmanScheduler:
    """
    Tracks validation MSE across epochs with a scalar Kalman filter and
    derives a continuously-varying alpha from the normalized innovation
    (how surprising each new observation is relative to the filter's own
    prediction and uncertainty).

    Usage (mirrors HMMScheduler's interface):

        scheduler = KalmanScheduler()
        ...
        # once per epoch, after computing that epoch's val_MSE:
        scheduler.observe(val_mse)
        alpha = scheduler.get_alpha()
    """

    def __init__(self, R=R_MEASUREMENT_NOISE, Q=Q_PROCESS_NOISE,
                 alpha_min=ALPHA_MIN, alpha_max=ALPHA_MAX, tau=TAU):
        self.R = R
        self.Q = Q
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.tau = tau

        self.x_hat = None   # current filtered estimate of "true" val_MSE
        self.P = 1.0         # current estimate variance (uncertainty)

        self.val_mse_history = []
        self.innovation_history = []
        self.normalized_innovation_history = []
        self.alpha_history = []

    def observe(self, val_mse):
        """
        Record a new epoch's validation MSE, run one Kalman predict+update
        step, and compute the resulting alpha. Call once per epoch, after
        that epoch's validation evaluation completes.
        """
        self.val_mse_history.append(val_mse)

        if self.x_hat is None:
            # First observation: initialize the filter's estimate directly
            # from the first reading, with a wide initial uncertainty since
            # we have no prior information yet.
            self.x_hat = val_mse
            self.P = 1.0
            innovation = 0.0
            normalized_innovation = 0.0
        else:
            # Predict
            x_pred = self.x_hat
            P_pred = self.P + self.Q

            # Update
            innovation = val_mse - x_pred
            S = P_pred + self.R
            K = P_pred / S
            self.x_hat = x_pred + K * innovation
            self.P = (1 - K) * P_pred

            normalized_innovation = abs(innovation) / math.sqrt(S)

        self.innovation_history.append(innovation)
        self.normalized_innovation_history.append(normalized_innovation)

        alpha = self._alpha_from_surprise(normalized_innovation)
        self.alpha_history.append(alpha)

    def _alpha_from_surprise(self, z):
        """Gaussian-shaped mapping from normalized innovation to alpha."""
        decay = math.exp(-(z ** 2) / (2 * self.tau ** 2))
        return self.alpha_min + (self.alpha_max - self.alpha_min) * decay

    def get_alpha(self):
        """Returns the most recently computed alpha value."""
        return self.alpha_history[-1] if self.alpha_history else self.alpha_max

    def get_normalized_innovation(self):
        return self.normalized_innovation_history[-1] if self.normalized_innovation_history else 0.0

    def history_summary(self):
        """Returns a list of (epoch, val_mse, x_hat, normalized_innovation, alpha) for logging/plots."""
        return [
            (i + 1, self.val_mse_history[i], self.alpha_history[i])
            for i in range(len(self.val_mse_history))
        ]


if __name__ == "__main__":
    # Standalone sanity check, same spirit as hmm_scheduler.py's Step 3:
    # confirm the filter behaves sensibly on synthetic data before ever
    # wiring it into train.py.
    print("Sanity check 1: smooth, slowly-improving sequence (expect alpha near max throughout)")
    smooth_seq = [1.45, 1.448, 1.446, 1.445, 1.443, 1.442, 1.441, 1.440, 1.439, 1.438]
    sched = KalmanScheduler()
    for v in smooth_seq:
        sched.observe(v)
    for epoch, val_mse, alpha in sched.history_summary():
        z = sched.normalized_innovation_history[epoch - 1]
        print(f"  epoch {epoch}: val_MSE={val_mse:.4f}  |z|={z:.3f}  alpha={alpha:.3f}")

    print("\nSanity check 2: smooth sequence with one sudden large jump (expect alpha to drop sharply at the jump, then recover)")
    jump_seq = [1.45, 1.448, 1.446, 1.445, 1.70, 1.44, 1.439, 1.438, 1.437, 1.436]
    sched2 = KalmanScheduler()
    for v in jump_seq:
        sched2.observe(v)
    for epoch, val_mse, alpha in sched2.history_summary():
        z = sched2.normalized_innovation_history[epoch - 1]
        print(f"  epoch {epoch}: val_MSE={val_mse:.4f}  |z|={z:.3f}  alpha={alpha:.3f}")

    print("\nSanity check 3: genuinely volatile sequence throughout (expect alpha to stay low)")
    volatile_seq = [1.45, 1.60, 1.40, 1.65, 1.38, 1.58, 1.42, 1.63, 1.39, 1.55]
    sched3 = KalmanScheduler()
    for v in volatile_seq:
        sched3.observe(v)
    for epoch, val_mse, alpha in sched3.history_summary():
        z = sched3.normalized_innovation_history[epoch - 1]
        print(f"  epoch {epoch}: val_MSE={val_mse:.4f}  |z|={z:.3f}  alpha={alpha:.3f}")
