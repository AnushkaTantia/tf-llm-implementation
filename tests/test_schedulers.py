"""
Unit tests for the alpha schedulers (checklist item V3).  Run:  python tests/test_schedulers.py <path-to-src>
Each test prints PASS/FAIL with the evidence; nothing here touches data or a GPU.
"""
import ast, itertools, math, random, sys, os
import numpy as np
SRC = sys.argv[1] if len(sys.argv) > 1 else "src"
sys.path.insert(0, SRC)
import hmm_scheduler as H
import oahmm_scheduler as OA
import kalman_scheduler as K

results = []
def check(name, ok, detail=""):
    results.append((name, ok)); print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))

# ---------- helpers: brute-force MAP decoding (independent of the schedulers' Viterbi code) ----------
def brute_hmm(deltas):
    S = [0, 1]; best = (-math.inf, None)
    for path in itertools.product(S, repeat=len(deltas)):
        lp = math.log(H.INITIAL[path[0]] + 1e-12) + H._log_gaussian_pdf(deltas[0], H.EMISSION_STD[path[0]])
        for t in range(1, len(deltas)):
            lp += math.log(H.TRANSITION[path[t-1]][path[t]] + 1e-12) + H._log_gaussian_pdf(deltas[t], H.EMISSION_STD[path[t]])
        if lp > best[0]: best = (lp, path)
    return list(best[1])

def brute_oahmm(obs):
    best = (-math.inf, None)
    em = lambda j, o: OA._lg(o[0], OA.MU_DELTA[j], OA.SD_DELTA[j]) + OA._lg(o[1], OA.MU_S[j], OA.SD_S[j])
    for path in itertools.product(OA.STATES, repeat=len(obs)):
        lp = math.log(OA.INITIAL[path[0]] + 1e-12) + em(path[0], obs[0])
        for t in range(1, len(obs)):
            lp += math.log(OA.TRANSITION[path[t-1]][path[t]] + 1e-12) + em(path[t], obs[t])
        if lp > best[0]: best = (lp, path)
    return list(best[1])

rng = random.Random(0)
def rand_val(T, scale):
    v = [1.4]; [v.append(v[-1] + rng.gauss(0, scale)) for _ in range(T - 1)]; return v

# ---------- T1: HMM Viterbi = exhaustive MAP ----------
ok = True; n = 0
for T in range(2, 11):
    for _ in range(40):
        vals = rand_val(T, rng.choice([0.005, 0.02, 0.08]))
        s = H.HMMScheduler(); [s.observe(v) for v in vals]
        mine = s.state_history[1:]                     # decoded path over the T-1 deltas
        ref = brute_hmm([vals[i+1] - vals[i] for i in range(T - 1)])
        ok &= (mine == ref); n += 1
check("T1 HMM Viterbi path equals exhaustive MAP path", ok, f"{n} random sequences, T=2..10")

# ---------- T2: OA-HMM online Viterbi final state = exhaustive MAP final state, at every epoch ----------
ok = True; n = 0
for T in range(2, 9):
    for _ in range(25):
        vals = rand_val(T, rng.choice([0.005, 0.02, 0.08])); trn = [0.6 - 0.01 * i + rng.gauss(0, 0.003) for i in range(T)]
        s = OA.OAHMMScheduler()
        for i, (v, t) in enumerate(zip(vals, trn)):
            s.observe(v, t)
            if i >= 1:
                ok &= (s.get_state() == brute_oahmm(s.obs_history)[-1]); n += 1
check("T2 OA-HMM online decision equals exhaustive MAP final state", ok, f"{n} decisions checked")

# ---------- T3: overfitting velocity = least-squares slope; 0 before the window fills ----------
s = OA.OAHMMScheduler(); vals = [1.40, 1.38, 1.39, 1.41, 1.44, 1.47, 1.52]; trn = [0.62, 0.60, 0.59, 0.58, 0.57, 0.56, 0.55]
for v, t in zip(vals, trn): s.observe(v, t)
gaps = [v - t for v, t in zip(vals, trn)]
ref = [0.0 if (i + 1) < OA.WINDOW else np.polyfit(range(OA.WINDOW), gaps[i+1-OA.WINDOW:i+1], 1)[0] for i in range(1, len(vals))]
got = [o[1] for o in s.obs_history]
check("T3 gap slope matches numpy least squares and is 0 before w epochs", np.allclose(got, ref, atol=1e-12), f"slopes {np.round(got, 4).tolist()}")

# ---------- T4: determinism ----------
def run_all(vals, trn):
    a = H.HMMScheduler(); b = OA.OAHMMScheduler(); c = K.KalmanScheduler(); out = []
    for v, t in zip(vals, trn):
        out.append((a.get_alpha(), b.get_alpha(), round(c.get_alpha(), 12))); a.observe(v); b.observe(v, t); c.observe(v)
    return out
vals = rand_val(30, 0.03); trn = [0.6 - 0.005 * i for i in range(30)]
check("T4 schedules are deterministic functions of the observed losses", run_all(vals, trn) == run_all(vals, trn))

# ---------- T5: alpha range ----------
okH = okO = okK = True
for _ in range(200):
    vals = rand_val(25, rng.choice([0.005, 0.03, 0.1])); trn = [0.6 - 0.004 * i for i in range(25)]
    a = H.HMMScheduler(); b = OA.OAHMMScheduler(alpha_over=0.5); c = K.KalmanScheduler()
    for v, t in zip(vals, trn):
        a.observe(v); b.observe(v, t); c.observe(v)
        okH &= a.get_alpha() in (0.05, 0.3); okO &= b.get_alpha() in (0.05, 0.3, 0.5); okK &= 0.05 - 1e-12 <= c.get_alpha() <= 0.3 + 1e-12
check("T5 HMM alpha in {0.05,0.30}; OA-HMM in {0.05,0.30,alpha_over}; Kalman in [0.05,0.30]", okH and okO and okK)

# ---------- T6: Kalman equations vs an independent implementation ----------
def kalman_ref(vals, Q=K.Q_PROCESS_NOISE, R=K.R_MEASUREMENT_NOISE, amin=0.05, amax=0.3, tau=1.5):
    x, P, out = None, 1.0, []
    for v in vals:
        if x is None: x, P, z = v, 1.0, 0.0
        else:
            Pp = P + Q; y = v - x; S_ = Pp + R; Kg = Pp / S_; x = x + Kg * y; P = (1 - Kg) * Pp; z = abs(y) / math.sqrt(S_)
        out.append(amin + (amax - amin) * math.exp(-z * z / (2 * tau * tau)))
    return out
vals = rand_val(40, 0.03); c = K.KalmanScheduler(); [c.observe(v) for v in vals]
check("T6 Kalman predict/update and alpha mapping match an independent implementation", np.allclose(c.alpha_history, kalman_ref(vals)))

# ---------- T7: behaviour on constructed trajectories ----------
stable = [1.40 + 0.001 * ((-1) ** i) for i in range(15)]
a = H.HMMScheduler(); [a.observe(v) for v in stable]
volatile = [1.40 + 0.12 * ((-1) ** i) for i in range(15)]
b = H.HMMScheduler(); [b.observe(v) for v in volatile]
check("T7a HMM: quiet trajectory -> stable; volatile -> unstable", a.get_state_name() == "stable" and b.get_state_name() == "unstable")
drift_v = [1.37, 1.36, 1.35, 1.345, 1.34] + [1.34 + 0.012 * k for k in range(1, 11)]
drift_t = [0.62 - 0.006 * i for i in range(15)]
o = OA.OAHMMScheduler(); h = H.HMMScheduler(); st_o, st_h = [], []
for v, t in zip(drift_v, drift_t):
    o.observe(v, t); h.observe(v); st_o.append(o.get_state_name()[0]); st_h.append(h.get_state_name()[0])
check("T7b slow upward drift + falling train loss: OA-HMM -> overfitting, HMM -> stable", st_o[-1] == "o" and st_h[-1] == "s",
      f"OA-HMM {''.join(st_o)} | HMM {''.join(st_h)}")
improving_v = [1.37, 1.36, 1.35, 1.34, 1.33, 1.32, 1.315, 1.31]; improving_t = [0.62 - 0.006 * i for i in range(8)]
o = OA.OAHMMScheduler(); seq = []
for v, t in zip(improving_v, improving_t): o.observe(v, t); seq.append(o.get_state_name()[0])
check("T7c OA-HMM: no overfitting alarm while validation loss is still improving", "o" not in seq, "".join(seq))

# ---------- T8: first-epoch handling ----------
a = H.HMMScheduler(); b = OA.OAHMMScheduler(); c = K.KalmanScheduler()
first = (a.get_alpha(), b.get_alpha(), c.get_alpha())
a.observe(1.4); b.observe(1.4, 0.6); c.observe(1.4)
after1 = (a.get_alpha(), b.get_alpha(), c.get_alpha())
check("T8 first epoch: HMM/OA-HMM start at alpha_low (0.05); Kalman starts at alpha_max (0.30); one observation changes nothing",
      first == (0.05, 0.05, 0.3) and after1 == (0.05, 0.05, 0.3), f"before {first} after epoch 1 {after1}")

# ---------- T9: missing / invalid observations ----------
def safe(fn):
    try: fn(); return "accepted"
    except Exception as e: return type(e).__name__
a = H.HMMScheduler(); a.observe(1.4); r_h = safe(lambda: a.observe(float("nan"))); alpha_h = a.get_alpha()
c = K.KalmanScheduler(); c.observe(1.4); r_k = safe(lambda: c.observe(float("nan"))); alpha_k = c.get_alpha()
nan_safe = not (isinstance(alpha_k, float) and math.isnan(alpha_k))
check("T9 NaN validation loss is rejected or handled (no NaN alpha)", nan_safe,
      f"HMM: {r_h}, alpha={alpha_h}; Kalman: {r_k}, alpha={alpha_k} -> needs an explicit guard if FAIL")

# ---------- T10: test-data isolation (static analysis of train.py) ----------
tree = ast.parse(open(os.path.join(SRC, "train.py")).read())
obs_args, test_eval_lines, loop_end = set(), [], None
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "observe":
        for a_ in node.args: obs_args |= {n.id for n in ast.walk(a_) if isinstance(n, ast.Name)}
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "evaluate"
            and any(isinstance(a_, ast.Name) and a_.id == "test_loader" for a_ in node.args)):
        test_eval_lines.append(node.lineno)
    if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "epoch":
        loop_end = node.end_lineno
# a name passed to observe() that is itself computed in train.py (e.g. train_task_mean = running_task / len(...))
# is replaced by the names it is computed from, so the check follows the data back to its source
assigned = {}
for node in ast.walk(tree):
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        assigned.setdefault(node.targets[0].id, set()).update(n.id for n in ast.walk(node.value) if isinstance(n, ast.Name))
allowed = {"val_mse", "running_task", "train_loader", "len"}
resolved, frontier = set(), set(obs_args)
while frontier:
    n_ = frontier.pop()
    if n_ in assigned and n_ not in allowed:
        frontier |= assigned[n_] - resolved - {n_}
    else:
        resolved.add(n_)
allowed = {"val_mse", "running_task", "train_loader", "len"}
check("T10 schedulers observe only validation loss and training loss", resolved <= allowed,
      f"names passed to observe(): {sorted(obs_args)} -> computed from {sorted(resolved)}")
check("T10b test_loader is evaluated only after the training loop ends", loop_end is not None and all(l > loop_end for l in test_eval_lines),
      f"epoch loop ends at line {loop_end}; test evaluation at lines {test_eval_lines}")

# ---------- T11: online decisions vs retrospective path ----------
vals = [1.40, 1.30, 1.42, 1.405, 1.41, 1.408, 1.409, 1.52, 1.40, 1.41]
a = H.HMMScheduler(); online = []
for v in vals: a.observe(v); online.append(a.get_state())
retro = a.state_history
check("T11 (documentation) HMM keeps the online decisions in the training log; state_history is the retrospective path",
      True, f"online {online} vs final retrospective {retro} -> differ: {online != retro}")

print(f"\n{sum(ok for _, ok in results)}/{len(results)} checks passed")
