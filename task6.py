import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set

@dataclass(frozen=True)
class Incident:
    t_start: int
    t_end: int
    label: Optional[int] = None

def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], n_states: int) -> np.ndarray:
    C = np.zeros((n_states, n_states), dtype=int)
    for t, p in zip(y_true, y_pred):
        C[int(t), int(p)] += 1
    return C

def classification_report(y_true: Sequence[int], y_pred: Sequence[int], n_states: int) -> Dict[str, object]:
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

    return {
        "accuracy": float((y_true == y_pred).mean()),
        "confusion_matrix": C,
        "precision_per_class": precision,
        "recall_per_class": recall,
        "f1_per_class": f1,
        "macro_f1": float(np.mean(f1)),
    }

def build_incident_mask(T: int, incidents: List[Incident]) -> np.ndarray:
    mask = np.zeros(T, dtype=int)
    for inc in incidents:
        s = max(0, inc.t_start)
        e = min(T-1, inc.t_end)
        if s <= e:
            mask[s:e+1] = 1
    return mask

def evaluate_incident_detection(decoded_path: Sequence[int], incidents: List[Incident], critical_states: Set[int]):
    path = np.asarray(decoded_path, dtype=int)
    T = len(path)
    incident_mask = build_incident_mask(T, incidents)
    predicted_incident = np.array([1 if s in critical_states else 0 for s in path], dtype=int)

    inc_windows = int(incident_mask.sum())
    non_inc_windows = int((incident_mask == 0).sum())

    incident_coverage = float(predicted_incident[incident_mask == 1].sum() / inc_windows) if inc_windows else 0.0
    false_alarm_rate = float(predicted_incident[incident_mask == 0].sum() / non_inc_windows) if non_inc_windows else 0.0

    delays = []
    for inc in incidents:
        s = max(0, inc.t_start)
        e = min(T-1, inc.t_end)
        idx = np.where(predicted_incident[s:e+1] == 1)[0]
        if len(idx) > 0:
            delays.append(int(idx[0]))

    mean_delay = float(np.mean(delays)) if delays else float("nan")

    return {
        "incident_coverage": incident_coverage,
        "false_alarm_rate": false_alarm_rate,
        "mean_detection_delay_steps": mean_delay,
    }

# Align true states and predicted states
T = len(path)
true_aligned = true_states[:T]

rep = classification_report(true_aligned, path, n_states=len(state_names))

# incidents from simulation (convert)
incidents = [Incident(i.t_start, i.t_end, i.state_label) for i in sim_incidents]
incident_metrics = evaluate_incident_detection(path, incidents, critical_states={2, 3})

print("✅ Task 6 completed")
print("Accuracy:", rep["accuracy"])
print("Macro-F1:", rep["macro_f1"])
print("Confusion matrix:\n", rep["confusion_matrix"])
print("Incident metrics:", incident_metrics)
