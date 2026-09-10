from __future__ import annotations

import numpy as np


# State order: WAKE, LIGHT, DEEP, REM
DEFAULT_INITIAL = np.array(
    [0.55, 0.42, 0.02, 0.01],
    dtype=float,
)

DEFAULT_TRANSITION = np.array(
    [
        # to:   W      L      D      R
        [      0.72,  0.24,  0.01,  0.03],  # from W
        [      0.08,  0.72,  0.12,  0.08],  # from L
        [      0.01,  0.17,  0.80,  0.02],  # from D
        [      0.08,  0.12,  0.01,  0.79],  # from R
    ],
    dtype=float,
)


def viterbi_smooth(
    emissions: np.ndarray,
    transition: np.ndarray = DEFAULT_TRANSITION,
    initial: np.ndarray = DEFAULT_INITIAL,
) -> np.ndarray:
    """
    Choose a whole-night stage path using epoch probabilities and transition
    weights. The transition weights are temporary heuristics.
    """
    emission = np.asarray(emissions, dtype=float)

    if emission.ndim != 2 or emission.shape[1] != 4:
        raise ValueError("emissions must have shape (epochs, 4)")

    if emission.shape[0] == 0:
        return np.zeros(0, dtype=int)

    eps = 1e-12

    log_emission = np.log(np.clip(emission, eps, 1.0))
    log_transition = np.log(np.clip(transition, eps, 1.0))
    log_initial = np.log(np.clip(initial, eps, 1.0))

    n_epochs = emission.shape[0]
    n_states = 4

    score = np.full((n_epochs, n_states), -np.inf, dtype=float)
    backpointer = np.zeros((n_epochs, n_states), dtype=int)

    score[0] = log_initial + log_emission[0]

    for t in range(1, n_epochs):
        for current in range(n_states):
            candidates = (
                score[t - 1]
                + log_transition[:, current]
            )

            previous = int(np.argmax(candidates))
            backpointer[t, current] = previous
            score[t, current] = (
                candidates[previous]
                + log_emission[t, current]
            )

    path = np.zeros(n_epochs, dtype=int)
    path[-1] = int(np.argmax(score[-1]))

    for t in range(n_epochs - 1, 0, -1):
        path[t - 1] = backpointer[t, path[t]]

    return path
