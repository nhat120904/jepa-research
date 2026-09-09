"""Finite statistical-experiment illustration, not a learned robot experiment.

Classical Le Cam deficiency, with TV = 0.5 * L1. A post-processing
kernel must be shared across hidden states. Requires numpy and scipy.
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


def deficiency(source, target):
    """inf_K max_theta TV(source[theta] @ K, target[theta])."""
    states, m = source.shape
    n = target.shape[1]
    nk = m * n
    size = nk + states * n + 1
    objective = np.zeros(size)
    objective[-1] = 1
    eq, rhs_eq, ub, rhs_ub = [], [], [], []
    for i in range(m):
        row = np.zeros(size)
        row[i * n:(i + 1) * n] = 1
        eq.append(row)
        rhs_eq.append(1)
    for theta in range(states):
        for j in range(n):
            row = np.zeros(size)
            row[j:nk:n] = source[theta]
            row[nk + theta * n + j] = -1
            ub.append(row)
            rhs_ub.append(target[theta, j])
            row = -row
            row[nk + theta * n + j] = -1
            ub.append(row)
            rhs_ub.append(-target[theta, j])
        row = np.zeros(size)
        row[nk + theta * n:nk + (theta + 1) * n] = 0.5
        row[-1] = -1
        ub.append(row)
        rhs_ub.append(0)
    result = linprog(objective, A_ub=ub, b_ub=rhs_ub,
                     A_eq=eq, b_eq=rhs_eq, bounds=(0, None), method="highs")
    if not result.success:
        raise RuntimeError(result.message)
    return {"distance": float(result.fun), "kernel": result.x[:nk].reshape(m, n).tolist()}


real = np.array([[0.9, 0.1], [0.1, 0.9]])
cases = {
    "correct": real.copy(),
    "erased_information": np.full((2, 2), 0.5),
    "hallucinated_information": np.array([[0.99, 0.01], [0.01, 0.99]]),
    "invertible_signal_relabeling": real[:, ::-1],
}
rows = {}
for name, model in cases.items():
    rows[name] = {
        "unconditional_signal_TV": float(0.5 * np.abs(real.mean(0) - model.mean(0)).sum()),
        "conditional_TV_without_adapter": float((0.5 * np.abs(real - model).sum(1)).max()),
        "real_to_model": deficiency(real, model),
        "model_to_real": deficiency(model, real),
        "real_optimal_classification_accuracy": float(real.max(0).sum() / 2),
        "model_optimal_classification_accuracy": float(model.max(0).sum() / 2),
    }
assert abs(rows["erased_information"]["model_to_real"]["distance"] - 0.4) < 1e-9
assert abs(rows["hallucinated_information"]["real_to_model"]["distance"] - 0.09) < 1e-9
assert rows["invertible_signal_relabeling"]["real_to_model"]["distance"] < 1e-9
assert rows["invertible_signal_relabeling"]["model_to_real"]["distance"] < 1e-9
payload = {
    "status": "known_principle_illustration_only",
    "assumptions": "Two equally likely hidden states; one observation; optimal randomized decision rules; exact channels.",
    "does_not_establish": ["novelty", "neural training gains", "robot planning gains", "failure of full conditional likelihood", "sequential guarantee"],
    "cases": rows,
}
Path(__file__).with_name("channel_example.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
