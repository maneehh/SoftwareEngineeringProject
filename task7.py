import numpy as np
import matplotlib.pyplot as plt
import os

def plot_transition_matrix(A, state_names, title="Transition Matrix (Learned A)"):
    fig, ax = plt.subplots()
    im = ax.imshow(A, aspect="auto")
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("To state")
    ax.set_ylabel("From state")
    ax.set_xticks(range(len(state_names)))
    ax.set_yticks(range(len(state_names)))
    ax.set_xticklabels(state_names, rotation=45, ha="right")
    ax.set_yticklabels(state_names)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            ax.text(j, i, f"{A[i,j]:.2f}", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    return fig

def plot_state_timeline(path, state_names, title="Viterbi State Timeline"):
    path = np.asarray(path, dtype=int)
    fig, ax = plt.subplots()
    ax.step(range(len(path)), path, where="post")
    ax.set_title(title)
    ax.set_xlabel("Time step (window index)")
    ax.set_ylabel("Hidden state")
    ax.set_yticks(range(len(state_names)))
    ax.set_yticklabels(state_names)
    fig.tight_layout()
    return fig

def plot_confidence(gamma, title="Confidence Over Time"):
    conf = np.asarray(gamma).max(axis=1)
    fig, ax = plt.subplots()
    ax.plot(range(len(conf)), conf)
    ax.set_title(title)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Max posterior probability")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig

def plot_entropy(gamma, eps=1e-12, title="Posterior Entropy Over Time"):
    g = np.clip(np.asarray(gamma), eps, 1.0)
    g = g / g.sum(axis=1, keepdims=True)
    H = -np.sum(g * np.log(g), axis=1)
    fig, ax = plt.subplots()
    ax.plot(range(len(H)), H)
    ax.set_title(title)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Entropy")
    fig.tight_layout()
    return fig

# Show plots inline
figA = plot_transition_matrix(A, state_names)
plt.show()

figT = plot_state_timeline(path, state_names)
plt.show()

figC = plot_confidence(gamma)
plt.show()

figH = plot_entropy(gamma)
plt.show()

# Save plots to files
os.makedirs("outputs", exist_ok=True)
figA.savefig("outputs/learned_A_heatmap.png", dpi=300, bbox_inches="tight")
figT.savefig("outputs/state_timeline.png", dpi=300, bbox_inches="tight")
figC.savefig("outputs/confidence.png", dpi=300, bbox_inches="tight")
figH.savefig("outputs/entropy.png", dpi=300, bbox_inches="tight")

# Save initial A vs learned A 
figA0 = plot_transition_matrix(A0, state_names, title="Transition Matrix (Initial A0)")
figA0.savefig("outputs/initial_A_heatmap.png", dpi=300, bbox_inches="tight")
plt.close(figA0)

print("✅ Task 7 completed")
print("Saved to outputs/: learned_A_heatmap.png, initial_A_heatmap.png, state_timeline.png, confidence.png, entropy.png")
