import numpy as np

def logsumexp(a: np.ndarray, axis=None) -> np.ndarray:
    a_max = np.max(a, axis=axis, keepdims=True)
    out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis) if axis is not None else out.squeeze()

def normalize_rows(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    M = np.clip(M, eps, None)
    return M / M.sum(axis=1, keepdims=True)

def normalize_vec(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.clip(v, eps, None)
    return v / v.sum()

def forward_log(A: np.ndarray, B: np.ndarray, pi: np.ndarray, O: np.ndarray) -> np.ndarray:
    N = A.shape[0]
    T = len(O)
    logA = np.log(A); logB = np.log(B); logpi = np.log(pi)
    log_alpha = np.full((T, N), -np.inf)
    log_alpha[0] = logpi + logB[:, O[0]]
    for t in range(1, T):
        for j in range(N):
            log_alpha[t, j] = logB[j, O[t]] + logsumexp(log_alpha[t-1] + logA[:, j], axis=0)
    return log_alpha

def backward_log(A: np.ndarray, B: np.ndarray, O: np.ndarray) -> np.ndarray:
    N = A.shape[0]
    T = len(O)
    logA = np.log(A); logB = np.log(B)
    log_beta = np.full((T, N), -np.inf)
    log_beta[T-1] = 0.0
    for t in range(T-2, -1, -1):
        for i in range(N):
            log_beta[t, i] = logsumexp(logA[i, :] + logB[:, O[t+1]] + log_beta[t+1, :], axis=0)
    return log_beta

def baum_welch_train(A, B, pi, obs_ids, max_iter=60, tol=1e-4, smoothing=1e-3, eps=1e-12):
    

    O = np.asarray(obs_ids, dtype=int)
    N = A.shape[0]
    M = B.shape[1]
    A = normalize_rows(A, eps)
    B = normalize_rows(B, eps)
    pi = normalize_vec(pi, eps)

    loglik_hist = []

    for it in range(max_iter):
        log_alpha = forward_log(A, B, pi, O)
        log_beta = backward_log(A, B, O)

        loglik = float(logsumexp(log_alpha[-1], axis=0))
        loglik_hist.append(loglik)

        # gamma
        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1)[:, None]
        gamma = np.exp(log_gamma)

        # xi
        T = len(O)
        xi = np.zeros((T-1, N, N), dtype=float)
        logA = np.log(A); logB = np.log(B)
        for t in range(T-1):
            log_xi = (
                log_alpha[t][:, None]
                + logA
                + logB[:, O[t+1]][None, :]
                + log_beta[t+1][None, :]
            )
            log_xi -= logsumexp(log_xi, axis=(0, 1))
            xi[t] = np.exp(log_xi)

        # M-step with smoothing
        pi_new = normalize_vec(gamma[0] + smoothing, eps)

        A_num = xi.sum(axis=0) + smoothing
        A_den = gamma[:-1].sum(axis=0)[:, None] + smoothing * N
        A_new = normalize_rows(A_num / A_den, eps)

        B_num = np.full((N, M), smoothing, dtype=float)
        for t in range(T):
            B_num[:, O[t]] += gamma[t]
        B_new = normalize_rows(B_num, eps)

        if it > 0 and abs(loglik_hist[-1] - loglik_hist[-2]) < tol:
            A, B, pi = A_new, B_new, pi_new
            break

        A, B, pi = A_new, B_new, pi_new

    return A, B, pi, loglik_hist

# Initial vs learned parameters
A0, B0, pi0 = hmm.A.copy(), hmm.B.copy(), hmm.pi.copy()

A, B, pi, ll_hist = baum_welch_train(A0, B0, pi0, obs_ids)

print("✅ Task 3 completed")
print("Iterations:", len(ll_hist))
print("LogLik first/last:", ll_hist[0], "→", ll_hist[-1])

print("\nInitial pi:", np.round(pi0, 3))
print("Learned pi:", np.round(pi, 3))

print("\nInitial A (rounded):\n", np.round(A0, 3))
print("\nLearned A (rounded):\n", np.round(A, 3))

print("\nInitial B shape:", B0.shape, " Learned B shape:", B.shape)
