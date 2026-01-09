import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional


def plot_transition_matrix(A: np.ndarray, state_names: Optional[List[str]] = None,
                           title: str = "Transition Matrix (A)"):
    """
    Heatmap-like visualization of A (no seaborn; pure matplotlib).
    """
    N = A.shape[0]
    labels = state_names or [f"S{i}" for i in range(N)]

    fig, ax = plt.subplots()
    im = ax.imshow(A, aspect="auto")
    plt.colorbar(im, ax=ax)

    ax.set_title(title)
    ax.set_xlabel("To state")
    ax.set_ylabel("From state")
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    # annotate probabilities (optional but impressive)
    for i in range(N):
        for j in range(N):
            ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.tight_layout()
    return fig


def plot_state_timeline(path: List[int], state_names: Optional[List[str]] = None,
                        title: str = "Most Likely Hidden State Timeline (Viterbi)"):
    """
    Visualizes the decoded path over time as a step plot.
    """
    path = np.asarray(path, dtype=int)
    T = len(path)
    N = int(path.max()) + 1 if T > 0 else 1
    labels = state_names or [f"S{i}" for i in range(N)]

    fig, ax = plt.subplots()
    ax.step(range(T), path, where="post")
    ax.set_title(title)
    ax.set_xlabel("Time step (window index)")
    ax.set_ylabel("Hidden state")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    fig.tight_layout()
    return fig


def plot_confidence(gamma: np.ndarray, title: str = "Inference Confidence Over Time"):
    """
    Shows confidence as max posterior probability per time step.
    gamma: (T, N) from forward_backward()
    """
    gamma = np.asarray(gamma, dtype=float)
    conf = gamma.max(axis=1)
    T = len(conf)

    fig, ax = plt.subplots()
    ax.plot(range(T), conf)
    ax.set_title(title)
    ax.set_xlabel("Time step (window index)")
    ax.set_ylabel("Max posterior probability")
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    return fig


def plot_entropy(gamma: np.ndarray, eps: float = 1e-12,
                 title: str = "Posterior Entropy (Uncertainty) Over Time"):
    """
    Entropy H_t = -sum_i gamma_t(i) log gamma_t(i)
    Lower entropy => more confident inference.
    """
    gamma = np.clip(np.asarray(gamma, dtype=float), eps, 1.0)
    gamma = gamma / gamma.sum(axis=1, keepdims=True)
    H = -np.sum(gamma * np.log(gamma), axis=1)
    T = len(H)

    fig, ax = plt.subplots()
    ax.plot(range(T), H)
    ax.set_title(title)
    ax.set_xlabel("Time step (window index)")
    ax.set_ylabel("Entropy")
    fig.tight_layout()
    return fig


def save_dashboard(A: np.ndarray, path: List[int], gamma: np.ndarray,
                   state_names: Optional[List[str]] = None,
                   out_dir: str = "outputs", prefix: str = "hmm"):
    """
    Saves a small dashboard as separate PNG files.
    Keeps Task 7 modular; final main will call this later.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

    figs = [
        (plot_transition_matrix(A, state_names), f"{prefix}_A_transition.png"),
        (plot_state_timeline(path, state_names), f"{prefix}_state_timeline.png"),
        (plot_confidence(gamma), f"{prefix}_confidence.png"),
        (plot_entropy(gamma), f"{prefix}_entropy.png"),
    ]

    for fig, name in figs:
        fig.savefig(os.path.join(out_dir, name), dpi=200)
        plt.close(fig)

    return [name for _, name in figs]
