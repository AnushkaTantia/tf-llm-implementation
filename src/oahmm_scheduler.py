"""
oahmm_scheduler.py -- Overfitting-Aware HMM (OA-HMM) scheduler for the TFB loss weight.
Paper: Section III-G, Algorithm 3. Same interface as HMMScheduler, but observe() also
takes the epoch's training task loss: sched.observe(val_mse, train_task).
States: 0 unstable, 1 stable, 2 overfitting.
Observation: delta = val_MSE(e)-val_MSE(e-1);  s = slope of (val_MSE - train_task) over last 5 epochs.
"""
import math

STATE_UNSTABLE, STATE_STABLE, STATE_OVERFIT = 0, 1, 2
STATE_NAMES = {0: "unstable", 1: "stable", 2: "overfitting"}
STATES = (0, 1, 2)

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
        self.phi = None
        self.state_history = []
        self.obs_history = []

    def observe(self, val_mse, train_task):
        self.val.append(float(val_mse))
        self.gap.append(float(val_mse) - float(train_task))
        if len(self.val) < 2:
            self.state_history.append(STATE_UNSTABLE)
            return
        d = self.val[-1] - self.val[-2]
        s = _slope(self.gap[-self.window:])
        self.obs_history.append((d, s))
        e = {j: _lg(d, 0.0, SD_DELTA[j]) + _lg(s, MU_S[j], SD_S[j]) for j in STATES}
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
    import random
    random.seed(0)
    sch = OAHMMScheduler()
    val, tr = 1.45, 0.80
    for ep in range(1, 21):
        val += random.gauss(0.004, 0.01)
        tr -= 0.008
        sch.observe(val, tr)
        print(f"epoch {ep:2d}  val={val:.4f}  train={tr:.4f}  state={sch.get_state_name():11s}  next alpha={sch.get_alpha():.2f}")
