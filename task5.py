import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class TransitionAlert:
    t: int
    from_state: int
    to_state: int
    prob: float

class HMMMonitor:
    
    def __init__(self, A, B, pi, state_names=None, eps=1e-12):
        self.A = A; self.B = B; self.pi = pi
        self.eps = eps
        self.state_names = state_names or [f"S{i}" for i in range(A.shape[0])]
        self.reset()

    def reset(self):
        self.belief = self.pi.astype(float).copy()

    def update(self, obs_id: int) -> Tuple[np.ndarray, float]:
        pred = self.belief @ self.A
        like = self.B[:, obs_id]
        unnorm = pred * like
        evidence = float(unnorm.sum())
        self.belief = unnorm / (evidence + self.eps)
        nll_increment = -np.log(evidence + self.eps)  # warning score
        return self.belief.copy(), float(nll_increment)

    def predict_k_steps(self, k: int) -> np.ndarray:
        dists = []
        p = self.belief.copy()
        for _ in range(k):
            p = p @ self.A
            dists.append(p.copy())
        return np.vstack(dists)

    def most_likely_future_states(self, k: int) -> List[str]:
        dists = self.predict_k_steps(k)
        idx = np.argmax(dists, axis=1)
        return [self.state_names[i] for i in idx]

    @staticmethod
    def improbable_transitions(viterbi_path: List[int], A: np.ndarray, threshold: float = 0.05):
        alerts = []
        for t in range(len(viterbi_path) - 1):
            i, j = viterbi_path[t], viterbi_path[t+1]
            p = float(A[i, j])
            if p < threshold:
                alerts.append(TransitionAlert(t=t, from_state=i, to_state=j, prob=p))
        return alerts

# Run monitor over the observation stream
monitor = HMMMonitor(A, B, pi, state_names=state_names)

warning_log = []  # (t, nll, state_guess)
THRESH_NLL = 3.0 

for t, o in enumerate(obs_ids):
    belief, nll = monitor.update(o)
    if nll > THRESH_NLL:
        warning_log.append((t, float(nll), state_names[int(np.argmax(belief))]))

# Prediction results 
pred5 = monitor.predict_k_steps(5)
future_states = monitor.most_likely_future_states(5)

# Improbable transition warnings from decoded path
transition_alerts = monitor.improbable_transitions(path, A, threshold=0.05)

print("✅ Task 5 completed")
print("Prediction results (5 steps, most likely states):", future_states)
print("Warning logs (NLL threshold) count:", len(warning_log))
print("First 10 warning logs:", warning_log[:10])

print("\nImprobable transition alerts count:", len(transition_alerts))
print("First 5 transition alerts:", transition_alerts[:5])
