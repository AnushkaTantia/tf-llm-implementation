"""
oahmm_scheduler.py  --  Overfitting-Aware HMM (OA-HMM) scheduler for the TFB loss weight.

Paper: Section III-G, Algorithm 3, Table (OA-HMM parameters).

Same interface as HMMScheduler, except observe() also takes the epoch's training task loss:

    sched = OAHMMScheduler(alpha_over=0.5)
    alpha = sched.get_alpha()          # before each epoch
    ...train one epoch...
    sched.observe(val_mse, train_task) # after validation

Observation per epoch e (from epoch 2 on):
    delta_e = val_MSE(e) - val_MSE(e-1)                         (same as standard HMM)
    s_e     = least-squares slope of G(k) = val_MSE(k) - train_task(k)
              over the last W=5 epochs; 0 until W epochs exist  ("overfitting velocity")
              (with fewer points the slope degenerates to the raw gap change,
               which is ~0.95 correlated with delta_e and was rejected)

Three states: 0 unstable, 1 stable, 2 overfitting.
Emission: N(delta; mu_delta[j], sd_delta[j]) * N(s; mu_s[j], sd_s[j])   (factorized)
          mu_delta[overfitting] > 0: in the overfitting regime validation loss drifts up.

v2 (2026-10-14): recalibrated for the RevIN model at lr=1e-5 (mu_delta, full-window slope).
Decoding: online Viterbi (identical decisions to re-decoding the full history).
"""
import math

STATE_UNSTABLE, STATE_STABLE, STATE_OVERFIT = 0, 1, 2
STATE_NAMES = {0: "unstable", 1: "stable", 2: "overfitting"}
STATES = (0, 1, 2)

MU_DELTA = {0: 0.0, 1: 0.0, 2: 0.010}
SD_DELTA = {0: 0.075, 1: 0.010, 2: 0.030}
MU_S = {0: 0.0, 1: 0.0, 2: 0.012}
SD_S = {0: 0.030, 1: 0.006, 2: 0.015}
TRANSITION = {
    0: {0: 0.70, 1: 0.20, 2: 0.10},
    1: {0: 0.10, 1: 0.80, 2: 0.10},
    2: {0: 0.05, 1: 0.10, 2: 0.85},
}
INITIAL = {0: 1.0, 1: 0.0, 2: 0.0}
WINDOW = 5


def _lg(x, mu, sd):
    return -0.5 * math.log(2 * math.pi * sd * sd) - (x - mu) ** 2 / (2 * sd * sd)


def _slope(ys):
    n = len(ys)
    if n < 2:
        return 0.0
    xm = (n - 1) / 2.0
    ym = sum(ys) / n
    num = sum((i - xm) * (y - ym) for i, y in enumerate(ys))
    den = sum((i - xm) ** 2 for i in range(n))
    return num / den


class OAHMMScheduler:
    def __init__(self, alpha_low=0.05, alpha_high=0.30, alpha_over=0.5, window=WINDOW):
        self.alpha_by_state = {0: alpha_low, 1: alpha_high, 2: alpha_over}
        self.window = window
        self.val, self.gap = [], []
        self.phi = None                 # online Viterbi log-scores per state
        self.state_history = []
        self.obs_history = []           # (delta, s) per epoch from epoch 2

    def observe(self, val_mse, train_task):
        self.val.append(float(val_mse))
        self.gap.append(float(val_mse) - float(train_task))
        if len(self.val) < 2:
            self.state_history.append(STATE_UNSTABLE)
            return
        d = self.val[-1] - self.val[-2]
        s = _slope(self.gap[-self.window:]) if len(self.gap) >= self.window else 0.0
        self.obs_history.append((d, s))
        e = {j: _lg(d, MU_DELTA[j], SD_DELTA[j]) + _lg(s, MU_S[j], SD_S[j]) for j in STATES}
        if self.phi is None:
            self.phi = {j: math.log(INITIAL[j] + 1e-12) + e[j] for j in STATES}
        else:
            self.phi = {
                j: max(self.phi[i] + math.log(TRANSITION[i][j] + 1e-12) for i in STATES) + e[j]
                for j in STATES
            }
        self.state_history.append(max(STATES, key=lambda j: self.phi[j]))

    def get_state(self):
        return self.state_history[-1] if self.state_history else STATE_UNSTABLE

    def get_state_name(self):
        return STATE_NAMES[self.get_state()]

    def get_alpha(self):
        return self.alpha_by_state[self.get_state()]


if __name__ == "__main__":
    # Sanity check: steady val, gap widening steadily -> should reach "overfitting".
    import random
    random.seed(0)
    sch = OAHMMScheduler()
    val, tr = 1.45, 0.80
    for ep in range(1, 21):
        val += random.gauss(0.004, 0.01)       # slow upward drift, low volatility
        tr -= 0.008                            # training loss keeps falling
        sch.observe(val, tr)
        print(f"epoch {ep:2d}  val={val:.4f}  train={tr:.4f}  state={sch.get_state_name():11s}  next alpha={sch.get_alpha():.2f}")
