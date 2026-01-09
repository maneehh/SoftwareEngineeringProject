import numpy as np

# Stable math helper 

def logsumexp(a: np.ndarray, axis=None) -> np.ndarray:
    a_max = np.max(a, axis=axis, keepdims=True)
    out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis) if axis is not None else out.squeeze()

#  Forward–Backward (posterior inference)

def forward_backward(A: np.ndarray, B: np.ndarray, pi: np.ndarray, obs_ids: list[int]):
    """
    Returns:
      gamma: (T, N) posterior state probabilities P(S_t=i | O_1..O_T)
      loglik: log P(O_1..O_T | A,B,pi)
    """
    O = np.asarray(obs_ids, dtype=int)
    T = len(O)
    N = A.shape[0]

    logA = np.log(A)
    logB = np.log(B)
    logpi = np.log(pi)

    # Forward pass: log_alpha[t,i] = log P(O_0..O_t, S_t=i)
    log_alpha = np.full((T, N), -np.inf)
    log_alpha[0] = logpi + logB[:, O[0]]
    for t in range(1, T):
        for j in range(N):
            log_alpha[t, j] = logB[j, O[t]] + logsumexp(log_alpha[t-1] + logA[:, j], axis=0)

    # Backward pass: log_beta[t,i] = log P(O_{t+1}..O_{T-1} | S_t=i)
    log_beta = np.full((T, N), -np.inf)
    log_beta[T - 1] = 0.0  # log(1)
    for t in range(T - 2, -1, -1):
        for i in range(N):
            log_beta[t, i] = logsumexp(logA[i, :] + logB[:, O[t + 1]] + log_beta[t + 1, :], axis=0)

    # Log-likelihood of the full sequence
    loglik = float(logsumexp(log_alpha[T - 1], axis=0))

    # Posterior: gamma[t,i] ∝ alpha[t,i] * beta[t,i]
    log_gamma = log_alpha + log_beta
    log_gamma -= logsumexp(log_gamma, axis=1)[:, None]  # normalize per time step
    gamma = np.exp(log_gamma)

    return gamma, loglik


#  Viterbi (most likely hidden state path) 

def viterbi_decode(A: np.ndarray, B: np.ndarray, pi: np.ndarray, obs_ids: list[int]):
    """
    Returns:
      path: list[int] best hidden-state indices (length T)
      best_logprob: log probability of the best path
    """
    O = np.asarray(obs_ids, dtype=int)
    T = len(O)
    N = A.shape[0]

    logA = np.log(A)
    logB = np.log(B)
    logpi = np.log(pi)

    delta = np.full((T, N), -np.inf)  # best logprob ending in state i at time t
    psi = np.zeros((T, N), dtype=int) # backpointers

    delta[0] = logpi + logB[:, O[0]]

    for t in range(1, T):
        for j in range(N):
            scores = delta[t - 1] + logA[:, j]
            psi[t, j] = int(np.argmax(scores))
            delta[t, j] = scores[psi[t, j]] + logB[j, O[t]]

    best_last = int(np.argmax(delta[T - 1]))
    best_logprob = float(delta[T - 1, best_last])

    # Backtrack
    path = [0] * T
    path[T - 1] = best_last
    for t in range(T - 2, -1, -1):
        path[t] = int(psi[t + 1, path[t + 1]])

    return path, best_logprob


def map_states(path: list[int], state_names: list[str]) -> list[str]:
    """Utility: convert indices -> human-readable state names."""
    return [state_names[i] for i in path]
