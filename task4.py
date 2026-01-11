import numpy as np

def forward_backward(A, B, pi, obs_ids):
    O = np.asarray(obs_ids, dtype=int)
    T = len(O); N = A.shape[0]
    logA = np.log(A); logB = np.log(B); logpi = np.log(pi)

    log_alpha = np.full((T, N), -np.inf)
    log_alpha[0] = logpi + logB[:, O[0]]
    for t in range(1, T):
        for j in range(N):
            log_alpha[t, j] = logB[j, O[t]] + logsumexp(log_alpha[t-1] + logA[:, j], axis=0)

    log_beta = np.full((T, N), -np.inf)
    log_beta[T-1] = 0.0
    for t in range(T-2, -1, -1):
        for i in range(N):
            log_beta[t, i] = logsumexp(logA[i, :] + logB[:, O[t+1]] + log_beta[t+1, :], axis=0)

    loglik = float(logsumexp(log_alpha[-1], axis=0))

    log_gamma = log_alpha + log_beta
    log_gamma -= logsumexp(log_gamma, axis=1)[:, None]
    gamma = np.exp(log_gamma)
    return gamma, loglik

def viterbi_decode(A, B, pi, obs_ids):
    O = np.asarray(obs_ids, dtype=int)
    T = len(O); N = A.shape[0]
    logA = np.log(A); logB = np.log(B); logpi = np.log(pi)

    delta = np.full((T, N), -np.inf)
    psi = np.zeros((T, N), dtype=int)

    delta[0] = logpi + logB[:, O[0]]

    for t in range(1, T):
        for j in range(N):
            scores = delta[t-1] + logA[:, j]
            psi[t, j] = int(np.argmax(scores))
            delta[t, j] = scores[psi[t, j]] + logB[j, O[t]]

    best_last = int(np.argmax(delta[-1]))
    best_logprob = float(delta[-1, best_last])

    path = [0] * T
    path[-1] = best_last
    for t in range(T-2, -1, -1):
        path[t] = int(psi[t+1, path[t+1]])

    return path, best_logprob

gamma, seq_loglik = forward_backward(A, B, pi, obs_ids)
path, best_path_logprob = viterbi_decode(A, B, pi, obs_ids)

decoded_states = [state_names[i] for i in path]

print("✅ Task 4 completed")
print("Sequence log-likelihood:", seq_loglik)
print("Viterbi best-path logprob:", best_path_logprob)
print("First 20 decoded states:", decoded_states[:20])
print("Gamma shape:", gamma.shape)
