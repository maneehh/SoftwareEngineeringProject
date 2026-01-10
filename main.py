import numpy as np

# Task 1
from observation import build_observations

# Task 2
from hmm_model import HMMModel

# Task 3
from training import baum_welch_train

# Task 4
from inference import forward_backward, viterbi_decode, map_states

# Task 5
from prediction_alerting import HMMMonitor

# Task 6
from evaluation import (
    entropy_over_time,
    uncertainty_reduction_over_time,
    evaluate_incident_detection,
    Incident,
)

# Task 7
from visualization import save_dashboard


def main(df):
    """
    df: pandas DataFrame with raw logs / metrics
        (timestamp, cpu_pct, mem_pct, latency_ms, error_count, auth_fail_count)
    """

    # Task 1 — Observation modeling
    obs_ids, vocab, table = build_observations(df)
    print(f"[Task 1] Generated {len(obs_ids)} observations, |V|={len(vocab)}")

    # Task 2 — HMM initialization
    state_names = ["Normal", "Degraded", "ResourceContention", "SecurityThreat"]
    hmm = HMMModel(states=state_names, n_observations=len(vocab))
    print("[Task 2] HMM initialized")

    # Task 3 — Parameter learning (Baum–Welch)
    A, B, pi, loglik_hist = baum_welch_train(
        hmm.A, hmm.B, hmm.pi, obs_ids,
        max_iter=50, tol=1e-4
    )
    print(f"[Task 3] Training completed in {len(loglik_hist)} iterations")

    # Task 4 — Inference & decoding
    gamma, loglik = forward_backward(A, B, pi, obs_ids)
    path, best_logprob = viterbi_decode(A, B, pi, obs_ids)
    decoded_states = map_states(path, state_names)

    print(f"[Task 4] Sequence log-likelihood: {loglik:.2f}")
    print(f"[Task 4] Viterbi log-probability: {best_logprob:.2f}")

    # Task 5 — Prediction & early warning
    monitor = HMMMonitor(A, B, pi, state_names=state_names)
    for o in obs_ids:
        monitor.update(o)

    future_states = monitor.most_likely_future_states(k=5)
    print("[Task 5] Most likely future states (5 steps):", future_states)

    # Task 6 — Evaluation
    # Example synthetic incident
    incidents = [
        Incident(t_start=20, t_end=40),   # example interval
    ]
    critical_states = {2, 3}  # ResourceContention, SecurityThreat

    eval_metrics = evaluate_incident_detection(
        decoded_path=path,
        incidents=incidents,
        critical_states=critical_states,
    )

    entropy_t = entropy_over_time(gamma)
    uncertainty_red = uncertainty_reduction_over_time(gamma)

    print("[Task 6] Evaluation metrics:", eval_metrics)
    print(f"[Task 6] Mean entropy: {entropy_t.mean():.3f}")
    print(f"[Task 6] Mean uncertainty reduction: {uncertainty_red.mean():.3f}")

    # Task 7 — Visualization
    files = save_dashboard(
        A=A,
        path=path,
        gamma=gamma,
        state_names=state_names,
        out_dir="outputs",
        prefix="task7",
    )

    print("[Task 7] Visualization files saved:", files)

    return {
        "A": A,
        "B": B,
        "pi": pi,
        "gamma": gamma,
        "path": path,
        "decoded_states": decoded_states,
        "evaluation": eval_metrics,
        "outputs": files,
    }
