"""Closed-loop arms and the H_rep continuation oracle (docs/GATE_ABC_PROTOCOL.md)."""

import math

import numpy as np
import torch

from ti_wm.contract import (
    anchor_fraction, candidate_seed, continuation_seed, medoid_index, select_candidate,
)
from ti_wm.pusht_runtime import (
    BANK, MAX_STEPS, done, final_frames, make_env, physical_scores, reset_branch, run_prefix,
)

K_SELECT = 8
REPS = 4
SELECT_REPS = (0, 1)
EVAL_REPS = (2, 3)


def run_official(root, runner):
    """Unmodified select_action loop through the gym.make wrappers (TimeLimit 300)."""
    from lerobot.common.envs.utils import preprocess_observation

    env = make_env(max_episode_steps=MAX_STEPS)
    runner.policy.reset()
    torch.manual_seed(root)
    obs, _ = env.reset(seed=root)
    steps, success, max_coverage = 0, False, float(env.unwrapped._get_coverage())
    try:
        while True:
            batch = preprocess_observation({"pixels": obs["pixels"][None], "agent_pos": obs["agent_pos"][None]})
            batch = {k: v.to(runner.device) for k, v in batch.items()}
            with torch.inference_mode():
                action = runner.policy.select_action(batch)
            obs, _, terminated, truncated, info = env.step(action.to("cpu").numpy()[0])
            steps += 1
            max_coverage = max(max_coverage, float(info["coverage"]))
            if terminated:
                success = bool(info["is_success"])
            if terminated or truncated:
                break
    finally:
        env.close()
    return {"success": success, "steps": steps, "max_coverage": max_coverage}


def _displacements(state, branches):
    a0, b0, r0 = np.array(state.env.agent.position), np.array(state.env.block.position), state.env.block.angle
    return {"agent_disp": [float(np.linalg.norm(np.array(b.env.agent.position) - a0)) for b in branches],
            "block_disp": [float(np.linalg.norm(np.array(b.env.block.position) - b0)) for b in branches],
            "block_rot": [float(abs(b.env.block.angle - r0)) for b in branches]}


def run_arm(root, arm, runner, scorer=None, cloner=None, keep_states=False, prog=None,
            p0_branches=BANK, diag=False, bank_size=BANK, rank=None):
    """One closed-loop episode. P0 simulates p0_branches branches for offline analysis only.

    bank_size < BANK draws only the first bank_size candidates of the nested bank (same seeds).
    """
    kwargs = {} if cloner is None else {"cloner": cloner}
    state = reset_branch(root)
    decisions, states, banks = [], [], []
    d = 0
    while not done(state):
        bank = runner.bank(state.hist, [candidate_seed(root, d, k) for k in range(bank_size)])
        record = {"d": d, "t": state.t}
        if arm == "P0":
            branches = [run_prefix(state, bank[k], **kwargs) for k in range(p0_branches)]
            record["phys"] = physical_scores(branches)
            if scorer is not None:
                record["vis"] = scorer.scores(final_frames(branches))
            if prog is not None:
                record["prog"] = prog.scores(branches)
            if diag:
                record.update(_displacements(state, branches))
            chosen = 0
        elif arm in ("PHYS8", "VIS8", "PROG8", "RANK8"):
            branches = [run_prefix(state, bank[k], **kwargs) for k in range(K_SELECT)]
            record["phys"] = physical_scores(branches)
            if scorer is not None:
                record["vis"] = scorer.scores(final_frames(branches))
            if prog is not None:
                record["prog"] = prog.scores(branches)
            if rank is not None:
                record["rank"] = rank.scores(state, branches)
            chosen = select_candidate(record[{"PHYS8": "phys", "VIS8": "vis", "PROG8": "prog", "RANK8": "rank"}[arm]])
        elif arm == "MEDOID8":
            chosen = medoid_index(bank[:K_SELECT].tolist())
            branches = {chosen: run_prefix(state, bank[chosen], **kwargs)}
        else:
            raise ValueError(arm)
        record["chosen"] = int(chosen)
        decisions.append(record)
        if keep_states:
            states.append(state)
            banks.append(bank[:K_SELECT].copy())
        state = branches[chosen]
        d += 1
    result = {"success": state.success, "steps": state.t, "max_coverage": state.max_coverage,
              "decisions": decisions}
    if keep_states:
        result["_states"], result["_banks"], result["_final"] = states, banks, state
    return result


def run_hrep(root, runner, states, banks, cloner=None):
    """Continuation oracle at one outcome-independent anchor of the P0 trajectory."""
    kwargs = {} if cloner is None else {"cloner": cloner}
    n = len(states)
    anchor = min(int(math.floor(anchor_fraction(root) * n)), n - 1)
    start, bank = states[anchor], banks[anchor]
    conts = [(k, r, run_prefix(start, bank[k], **kwargs)) for k in range(K_SELECT) for r in range(REPS)]
    dec = 0
    while True:
        active = [i for i, (_, _, b) in enumerate(conts) if not done(b)]
        if not active:
            break
        seeds = [continuation_seed(root, conts[i][0], conts[i][1], dec) for i in active]
        chunks = runner.draw([conts[i][2].hist for i in active], seeds)
        for j, i in enumerate(active):
            k, r, b = conts[i]
            conts[i] = (k, r, run_prefix(b, chunks[j], **kwargs))
        dec += 1
    success = np.zeros((K_SELECT, REPS), dtype=bool)
    for k, r, b in conts:
        success[k, r] = b.success
    return {"anchor": anchor, "n_decisions": n, "anchor_t": start.t, "success": success.tolist()}


def hrep_estimates(success):
    """Split (selection seeds vs evaluation seeds) and naive estimates for one anchor."""
    s = np.asarray(success, dtype=float)
    sel, ev = s[:, list(SELECT_REPS)].mean(axis=1), s[:, list(EVAL_REPS)].mean(axis=1)
    chosen = select_candidate(sel.tolist())
    all_mean = s.mean(axis=1)
    return {"split": float(ev[chosen] - ev[0]), "chosen": int(chosen),
            "naive": float(all_mean.max() - all_mean[0]), "default": float(all_mean[0])}
