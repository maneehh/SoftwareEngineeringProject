import numpy as np
from typing import List


class HMMModel:
    """
    Task 2:
    Defines the Hidden Markov Model structure:
    - hidden states
    - observation symbols
    - initial parameters A, B, pi
    """

    def __init__(self, states: List[str], n_observations: int):
        self.states = states
        self.n_states = len(states)
        self.n_obs = n_observations

        self.A = self._init_transition_matrix()
        self.B = self._init_emission_matrix()
        self.pi = self._init_initial_distribution()

    def _init_transition_matrix(self) -> np.ndarray:
        """
        Initialize transition matrix with strong self-transitions.
        """
        A = np.full((self.n_states, self.n_states), 0.05)
        np.fill_diagonal(A, 0.85)
        A = A / A.sum(axis=1, keepdims=True)
        return A

    def _init_emission_matrix(self) -> np.ndarray:
        """
        Initialize emission probabilities uniformly.
        Refined later by Baum–Welch (Task 3).
        """
        B = np.full((self.n_states, self.n_obs), 1.0 / self.n_obs)
        return B

    def _init_initial_distribution(self) -> np.ndarray:
        """
        Assume the system usually starts in a normal state.
        """
        pi = np.full(self.n_states, 0.05)
        pi[0] = 0.85  # Normal
        return pi / pi.sum()
