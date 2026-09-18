"""Direct Monte Carlo intervention for the extended BAF simulator.

Only the S input to the approval policy is set to zero, at every time step.
Natural group membership remains in the bank, drift, graph, and exit mechanism.
The estimand is approval among the initial cohort: exited agents contribute zero.
With network interference this is a simulator-defined intervention disparity,
not an identification claim for the paper's simpler causal graph.
"""
from copy import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


class ReferenceGroupPolicy:
    """Block S -> decision, including policy probabilities used by the transition."""
    def __init__(self, model):
        self.model = model

    @property
    def params(self):
        return self.model.params

    def predict(self, s, X):
        return self.model.predict(np.zeros(len(s), dtype=int), X)


@dataclass
class InterventionRollout:
    s: np.ndarray
    Xs: list
    As: list

    def __post_init__(self):
        self.s = np.asarray(self.s, dtype=int).copy()
        if not np.isin(self.s, [0, 1]).all() or not all(np.any(self.s == g) for g in (0, 1)):
            raise ValueError('Direct fairness requires both groups in the initial cohort.')
        if not self.Xs or len(self.Xs) != len(self.As):
            raise ValueError('Intervention features and activity must have matching horizons.')
        self.Xs = [np.asarray(x, dtype=float).copy() for x in self.Xs]
        self.As = [np.asarray(a).copy() for a in self.As]
        for x, a in zip(self.Xs, self.As):
            if x.ndim != 2 or len(x) != len(self.s) or not np.isfinite(x).all():
                raise ValueError('Intervention features must be finite and aligned.')
            if a.shape != self.s.shape or not np.isin(a, [0, 1]).all():
                raise ValueError('Intervention activity must be aligned and binary.')


class FeatureHistory(list):
    """A factual feature history carrying its explicitly simulated intervention.

    Existing notebook result tuples keep their shape. Consumers read the attached
    rollout, rather than a mutable global 'last simulation' cache.
    """
    def __init__(self, values, intervention):
        super().__init__(values)
        self.intervention = intervention


def sample_intervention(model, bank, agent, steps, *, feature_update, seed=2026,
                        enable_opt_out=True, **settings):
    """Freeze the current policy, sample once, and restore the caller's RNG state."""
    from simulator import run_simulation

    if steps < 1:
        raise ValueError('The intervention horizon must be positive.')
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        with torch.no_grad():
            result = run_simulation(
                ReferenceGroupPolicy(model), copy(agent), steps,
                repayment_model=bank, feature_update=feature_update,
                enable_opt_out=enable_opt_out, seed=seed, **settings,
            )
        return InterventionRollout(result[0], result[3], result[9])
    finally:
        np.random.set_state(state)


def direct_long_term_loss(model, rollout):
    """Unweighted terminal surrogate, with frozen RRM features/activity.

    Preserve the legacy one-sided softplus(sum)-1 convention, replacing only
    the ratio estimator. Inactive applicants have approval 0, hence negative
    outcome surrogate 1. No survivor-only denominator and no label fitting.
    """
    parameter = model.linear.weight
    X = torch.as_tensor(rollout.Xs[-1], dtype=parameter.dtype, device=parameter.device)
    s = torch.as_tensor(rollout.s, dtype=parameter.dtype, device=parameter.device)
    active = torch.as_tensor(rollout.As[-1], dtype=parameter.dtype, device=parameter.device)
    h, _ = model.forward(torch.zeros_like(s), X)
    positive = active * F.softplus(h)
    negative = active * F.softplus(-h) + (1 - active)
    return torch.relu(positive[s == 1].mean() + negative[s == 0].mean() - 1)


def direct_fairness_series(model, rollout):
    """Signed difference of initial-cohort approval probabilities, S=1 minus S=0."""
    gaps = []
    with torch.no_grad():
        for X, active in zip(rollout.Xs, rollout.As):
            _, p = model.predict(np.zeros_like(rollout.s), X)
            p = np.asarray(p, dtype=float)
            if p.shape != rollout.s.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                raise ValueError('Approval probabilities must be finite and in [0,1].')
            approval = active * p
            gaps.append(approval[rollout.s == 1].mean() - approval[rollout.s == 0].mean())
    return np.asarray(gaps)


def require_intervention(Xs):
    rollout = getattr(Xs, 'intervention', None)
    if rollout is None:
        raise ValueError('Direct fairness needs new intervention trajectories; rerun the BAF simulation.')
    if len(rollout.Xs) != len(Xs):
        raise ValueError('Factual and intervention horizons must match.')
    return rollout
