import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class TransitionAlert:
    t: int
    from_state: int
    to_state: int
    prob: float


class HMMMonitor:

    def __init__(self, A: np.ndarray, B: np.ndarray, pi: np.ndarray,
                 state_names: Optional[List[str]] = None, eps: float = 1e-12):
        self.A = A
        self.B = B
        self.pi = pi
        self.eps = eps
        self.state_names = state_names or [f"S{i}" for i in range(A.shape[0])]
        self.reset()

    def reset(self):
        self.belief = self.pi.astype(float).copy()  # P(S_t | O_1..O_t) for online mode

    def update(self, obs_id: int) -> Tuple[np.ndarray, float]:
        """
        Online filtering update:
          pred = belief @ A
          belief' ∝ pred * B[:, obs_id]
        Returns:
          belief (N,), nll_increment = -log P(O_t | O_1..O_{t-1})
        """
        pred = self.belief @ self.A                      # predictive state distribution
        like = self.B[:, obs_id]                         # emission likelihoods
        unnorm = pred * like
        evidence = float(unnorm.sum())                   # P(O_t | past)
        self.belief = unnorm / (evidence + self.eps)

        nll_increment = -np.log(evidence + self.eps)     # anomaly score (bigger = more surprising)
        return self.belief.copy(), float(nll_increment)

    def predict_k_steps(self, k: int) -> np.ndarray:
        """
        Predict future hidden-state distributions:
          P(S_{t+1}) = belief @ A
          P(S_{t+2}) = belief @ A^2, etc.
        Returns:
          dists: (k, N)
        """
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
    def improbable_transitions(viterbi_path: List[int], A: np.ndarray,
                               threshold: float = 0.05) -> List[TransitionAlert]:
        """
        Early warning based on improbable transitions in the decoded path.
        Flags transitions whose probability A[s_t, s_{t+1}] is below threshold.
        """
        alerts: List[TransitionAlert] = []
        for t in range(len(viterbi_path) - 1):
            i, j = viterbi_path[t], viterbi_path[t + 1]
            p = float(A[i, j])
            if p < threshold:
                alerts.append(TransitionAlert(t=t, from_state=i, to_state=j, prob=p))
        return alerts
