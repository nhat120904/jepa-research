"""Exact policy evaluation and regret.

Because the hidden-state / observation spaces are tiny, we compute the EXACT
expected return of a policy by enumerating:

    sum over true hidden state z* (weighted by the prior)
        sum over observation sequences (weighted by P(o | z*, probe))

The policy re-plans (closed loop) at every step from its exact Bayesian
posterior; observations are drawn from the TRUE hidden state.  No Monte-Carlo
noise, no sampling: the returned numbers are exact expectations.

Compute is measured separately, at the ROOT decision (the first ``act`` call
from the prior), which is the representative planning cost per episode and is
directly comparable across policies.
"""

from __future__ import annotations

from dataclasses import dataclass

from .core import Belief, ComputeCounter, Task


@dataclass
class EvalResult:
    ret: float                 # exact expected return (net of probe cost)
    compute: dict              # root-decision compute breakdown
    n_probes_expected: float   # expected number of probes taken


def _rollout(task, policy, goal, z_star, belief, budget, cost, sink):
    action = policy.act(belief, budget, goal, sink)
    kind, val = action
    if kind == "commit":
        return task.reward(z_star, val, goal), 0.0
    probe = val
    total_r = 0.0
    total_p = 0.0
    for o in task.obs_space(probe):
        po = task.obs_prob(z_star, probe, o)
        if po <= 0.0:
            continue
        post = belief.updated(probe, o)
        r, np_ = _rollout(task, policy, goal, z_star, post, budget - 1,
                          cost + task.probe_cost, sink)
        total_r += po * r
        total_p += po * np_
    return total_r - task.probe_cost, total_p + 1.0


def exact_return(task: Task, policy, goal, budget: int | None = None) -> EvalResult:
    if budget is None:
        budget = task.max_budget
    b0 = Belief.prior(task)

    # root-decision compute (single act() from the prior)
    root_counter = ComputeCounter()
    policy.act(b0, budget, goal, root_counter)

    # exact expected return via enumeration; compute here is discarded
    sink = ComputeCounter()
    ret = 0.0
    exp_probes = 0.0
    for k, z in enumerate(task.hidden_states):
        pz = task.prior[k]
        if pz <= 0.0:
            continue
        r, npr = _rollout(task, policy, goal, z, b0, budget, 0.0, sink)
        ret += pz * r
        exp_probes += pz * npr
    return EvalResult(ret=ret, compute=root_counter.as_dict(), n_probes_expected=exp_probes)


def fully_observed_return(task: Task, goal) -> float:
    """Upper bound: the hidden state is known, so always commit optimally."""
    ret = 0.0
    for k, z in enumerate(task.hidden_states):
        pz = task.prior[k]
        if pz <= 0.0:
            continue
        a = task.preferred_action(z, goal)
        ret += pz * task.reward(z, a, goal)
    return ret
