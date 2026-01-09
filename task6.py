import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Set


@dataclass(frozen=True)
class Incident:
    """
    Represents a known incident interval on the timeline.
    t_start and t_end are inclusive indices in the windowed time steps (Task 1).
    label can be a state index (e.g., SecurityThreat) OR None if only "incident/non-incident" is known.
    """
    t_start: int
    t_end: int
    label: Optional[int] = None


# Uncertainty metrics (from posterior gamma) 

def entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    """Shannon entropy H(p) for a probability vector p."""
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))

def entropy_over_time(gamma: np.ndarray) -> np.ndarray:
    """
    gamma: (T, N) posterior state probabilities from forward_backward()
    returns: H_t for each time step, shape (T,)
    """
    return np.array([entropy(gamma[t]) for t in range(gamma.shape[0])], dtype=float)

def uncertainty_reduction_over_time(gamma: np.ndarray, baseline: str = "uniform") -> np.ndarray:
    """
    Measures uncertainty reduction per time step.
    If baseline="uniform": reduction_t = H(uniform) - H(gamma_t)
    Larger = more confident inference.
    """
    T, N = gamma.shape
    if baseline != "uniform":
        raise ValueError("Only baseline='uniform' is supported for now.")
    H0 = float(np.log(N))  # entropy of uniform distribution
    Ht = entropy_over_time(gamma)
    return H0 - Ht


# State-sequence validation

def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], n_states: int) -> np.ndarray:
    C = np.zeros((n_states, n_states), dtype=int)
    for t, p in zip(y_true, y_pred):
        C[int(t), int(p)] += 1
    return C

def classification_report(y_true: Sequence[int], y_pred: Sequence[int], n_states: int) -> Dict[str, object]:
    """
    Returns accuracy + per-class precision/recall/f1 and macro averages.
    No external dependencies (sklearn not required).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    C = confusion_matrix(y_true, y_pred, n_states)
    tp = np.diag(C).astype(float)
    fp = C.sum(axis=0) - tp
    fn = C.sum(axis=1) - tp

    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    acc = float((y_true == y_pred).mean())
    return {
        "accuracy": acc,
        "confusion_matrix": C,
        "precision_per_class": precision,
        "recall_per_class": recall,
        "f1_per_class": f1,
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
    }


# Incident-based validation

def build_incident_mask(T: int, incidents: List[Incident]) -> np.ndarray:
    """mask[t]=1 if t is inside any incident interval."""
    mask = np.zeros(T, dtype=int)
    for inc in incidents:
        s = max(0, inc.t_start)
        e = min(T - 1, inc.t_end)
        if s <= e:
            mask[s:e+1] = 1
    return mask

def evaluate_incident_detection(
    decoded_path: Sequence[int],
    incidents: List[Incident],
    critical_states: Set[int],
) -> Dict[str, float]:
    """
    Validates decoded hidden states against known incident intervals.
    Treats prediction "incident" if decoded state ∈ critical_states.

    Metrics:
      - incident_coverage: fraction of incident windows labeled critical
      - false_alarm_rate: fraction of non-incident windows labeled critical
      - mean_detection_delay: average delay from incident start to first critical state (only for incidents detected)
    """
    path = np.asarray(decoded_path, dtype=int)
    T = len(path)
    incident_mask = build_incident_mask(T, incidents)

    predicted_incident = np.array([1 if s in critical_states else 0 for s in path], dtype=int)

    # Coverage within incidents
    inc_windows = incident_mask.sum()
    if inc_windows == 0:
        incident_coverage = 0.0
    else:
        incident_coverage = float((predicted_incident[incident_mask == 1].sum()) / inc_windows)

    # False alarms outside incidents
    non_inc_windows = (incident_mask == 0).sum()
    if non_inc_windows == 0:
        false_alarm_rate = 0.0
    else:
        false_alarm_rate = float((predicted_incident[incident_mask == 0].sum()) / non_inc_windows)

    # Detection delay per incident
    delays = []
    for inc in incidents:
        s = max(0, inc.t_start)
        e = min(T - 1, inc.t_end)
        if s > e:
            continue
        # first t in [s,e] with predicted incident
        idx = np.where(predicted_incident[s:e+1] == 1)[0]
        if len(idx) > 0:
            delays.append(int(idx[0]))  # offset from start
    mean_detection_delay = float(np.mean(delays)) if delays else float("nan")

    return {
        "incident_coverage": incident_coverage,
        "false_alarm_rate": false_alarm_rate,
        "mean_detection_delay_steps": mean_detection_delay,
    }
