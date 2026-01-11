import numpy as np
from typing import List

class HMMModel:
    
    def __init__(self, states: List[str], n_observations: int):
        self.states = states
        self.n_states = len(states)
        self.n_obs = n_observations

        self.A = self._init_transition_matrix()
        self.B = self._init_emission_matrix()
        self.pi = self._init_initial_distribution()

    def _init_transition_matrix(self) -> np.ndarray:
        A = np.full((self.n_states, self.n_states), 0.05)
        np.fill_diagonal(A, 0.85)
        A = A / A.sum(axis=1, keepdims=True)
        return A

    def _init_emission_matrix(self) -> np.ndarray:
        return np.full((self.n_states, self.n_obs), 1.0 / self.n_obs)

    def _init_initial_distribution(self) -> np.ndarray:
        pi = np.full(self.n_states, 0.05)
        pi[0] = 0.85  # start in Normal most of the time
        return pi / pi.sum()

# Create model
state_names = ["Normal", "Degraded", "ResourceContention", "SecurityThreat"]
hmm = HMMModel(states=state_names, n_observations=len(vocab))

print("✅ Task 2 completed")
print("States:", state_names)
print("Initial pi:", np.round(hmm.pi, 3))
print("Initial A (rounded):\n", np.round(hmm.A, 3))
print("Initial B shape:", hmm.B.shape)
