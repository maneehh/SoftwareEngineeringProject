import numpy as np

# Numerically stable helpers

def logsumexp(a: np.ndarray, axis=None) -> np.ndarray:
    """Stable log(sum(exp(a)))"""
    a_max = np.max(a, axis=axis, keepdims=True)
    out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis) if axis is not None else out.squeeze()

def normalize_rows(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    M = np.clip(M, eps, None)
    return M / M.sum(axis=1, keepdims=True)

def normalize_vec(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.clip(v, eps, None)
    return v / v.sum()


#  Forward / Backward in log-space 

def forward_log(A: np.ndarray, B: np.ndarray, pi: np.ndarray, O: np.ndarray) -> np.ndarray:
    """
    Returns log_alpha[t, i] = log P(O_0..O_t, S_t=i)
    """
    N = A.shape[0]
    T = len(O)
    logA = np.log(A)
    logB = np.log(B)
    logpi = np.log(pi)

    log_alpha = np.full((T, N), -np.inf)
    log_alpha[0] = logpi + logB[:, O[0]]

    for t in range(1, T):
        for j in range(N):
            log_alpha[t, j] = logB[j, O[t]] + logsumexp(log_alpha[t-1] + logA[:, j], axis=0)
    return log_alpha

def backward_log(A: np.ndarray, B: np.ndarray, O: np.ndarray) -> np.ndarray:
    """
    Returns log_beta[t, i] = log P(O_{t+1}..O_{T-1} | S_t=i)
    """
    N = A.shape[0]
    T = len(O)
    logA = np.log(A)
    logB = np.log(B)

    log_beta = np.full((T, N), -np.inf)
    log_beta[T-1] = 0.0  # log(1)

    for t in range(T-2, -1, -1):
        for i in range(N):
            log_beta[t, i] = logsumexp(
                logA[i, :] + logB[:, O[t+1]] + log_beta[t+1, :],
                axis=0
            )
    return log_beta


# Baum–Welch training (single sequence)

def baum_welch_train(
    A: np.ndarray,
    B: np.ndarray,
    pi: np.ndarray,
    obs_ids: list[int],
    max_iter: int = 50,
    tol: float = 1e-4,
    smoothing: float = 1e-3,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """
    Task 3 deliverable
    Returns (A_new, B_new, pi_new, loglik_history).
    """
    O = np.asarray(obs_ids, dtype=int)
    N = A.shape[0]
    M = B.shape[1]
    loglik_hist: list[float] = []

    # Ensure valid stochastic matrices
    A = normalize_rows(A, eps)
    B = normalize_rows(B, eps)
    pi = normalize_vec(pi, eps)

    for it in range(max_iter):
        # E-step: compute posteriors
        log_alpha = forward_log(A, B, pi, O)
        log_beta = backward_log(A, B, O)

        loglik = logsumexp(log_alpha[-1], axis=0).item()
        loglik_hist.append(float(loglik))

        # log_gamma[t,i] ∝ log_alpha[t,i] + log_beta[t,i]
        log_gamma = log_alpha + log_beta
        log_gamma = log_gamma - logsumexp(log_gamma, axis=1)[:, None]  # normalize per t
        gamma = np.exp(log_gamma)

        # xi[t,i,j] ∝ alpha[t,i] * A[i,j] * B[j,O[t+1]] * beta[t+1,j]
        T = len(O)
        xi = np.zeros((T-1, N, N), dtype=float)
        logA = np.log(A)
        logB = np.log(B)

        for t in range(T-1):
            # build log_xi matrix
            log_xi = (
                log_alpha[t][:, None]
                + logA
                + logB[:, O[t+1]][None, :]
                + log_beta[t+1][None, :]
            )
            log_xi = log_xi - logsumexp(log_xi, axis=(0, 1))  # normalize whole matrix
            xi[t] = np.exp(log_xi)

        # M-step: re-estimate parameters with smoothing
        pi_new = normalize_vec(gamma[0] + smoothing, eps)

        A_num = xi.sum(axis=0) + smoothing
        A_den = gamma[:-1].sum(axis=0)[:, None] + smoothing * N
        A_new = normalize_rows(A_num / A_den, eps)

        B_num = np.full((N, M), smoothing, dtype=float)
        for t in range(T):
            B_num[:, O[t]] += gamma[t]
        B_new = normalize_rows(B_num, eps)

        # Convergence check (diagnostic)
        if it > 0 and abs(loglik_hist[-1] - loglik_hist[-2]) < tol:
            A, B, pi = A_new, B_new, pi_new
            break

        A, B, pi = A_new, B_new, pi_new

    return A, B, pi, loglik_hist
