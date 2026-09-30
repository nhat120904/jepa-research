"""Batched closed loop with full candidate logging, and on-policy data helpers.

docs/CTA_ONPOLICY_DATA_PROTOCOL_20260927.md. Many roots advance in lockstep (as scripts/d_collect.py does), so every
decision simulates all K candidates, records native coverage and geometry labels, and scores the bank with every
registered scorer. The acting arm only chooses which simulated branch becomes the next state; learned arms never read
the simulated futures. Compute-node only for anything that loads models or environments.
"""

import time

import numpy as np
import torch

from ti_wm.contract import candidate_seed, select_candidate
from ti_wm.cta import action_features, goal_scores, proprio
from ti_wm.cta_geometry import registration_score
from ti_wm.cta_parallel import normalized_score
from ti_wm.gates import cluster_ratio

K = 8                                # policy samples per decision (candidate 0 = P0)
STATE_CHUNK = 4                      # states per scorer call: 4 x 8 candidates x 16 goals = 512 reader rows
# Privileged selectors on the simulated futures: native coverage (PHYS) / geometry (GEOM). The "8" versions only see
# the policy's own K samples; the "16" versions see the whole deployment bank (policy + perturbed, Round 5).
ORACLES = ("PHYS8", "GEOM8", "PHYS16", "GEOM16", "GEOM64")   # GEOM64: geometry-best of the whole bank (K-scaling)
CODE_SAMPLES = 8                     # "parallel_rs": sampled codes per candidate (RESEARCH_DESIGN §4: average answers)
RESTRICT = "@8"                      # suffix: a learned arm restricted to the policy's K samples of a larger bank
PENALTY = "~"                        # suffix "~lam": score - lam * sigma_k, sigma_k = 0 for policy samples and the
                                     # perturbation scale (SIGMAS) of perturbed copy k-K; a prior that larger
                                     # departures from the policy's own samples need larger predicted gains
SIGMAS = (4., 8., 12., 16., 20., 24., 28., 32.)   # perturbed-bank offsets in world units (1 px = 512/96)
WORKSPACE = (0., 512.)
PERTURB_STREAM, MIX_STREAM = 5000, 6000          # candidate_seed "candidate" slots reserved for these draws


# ----------------------------------------------------------------------------- training-data design

def perturb(chunks, root, decision, sigmas=SIGMAS):
    """(K, T, 2) policy chunks -> copies shifted by one constant 2-D offset per candidate, candidate k with scale
    sigmas[k]. Deterministic in (root, decision). Training data. The design keeps deployment banks to the policy's own
    samples (RESEARCH_DESIGN / proposal: candidates are plausible actions of a trained policy); Round 5 and the
    penalty test used them at deployment as an out-of-design experiment (docs/CTA_ROUND6_PROTOCOL.md)."""
    chunks = np.asarray(chunks, np.float64)
    rng = np.random.default_rng(candidate_seed(root, decision, PERTURB_STREAM))
    offset = rng.normal(size=(len(chunks), 1, 2)) * np.asarray(sigmas, np.float64)[:, None, None]
    return np.clip(chunks + offset, *WORKSPACE).astype(np.float32)


def mixed_bank(chunks, roots, decision):
    """Round-5 deployment bank: (A, K, T, 2) policy samples -> (A, 2K, T, 2) = policy samples, then their perturbed
    copies (same construction as the Round-4 perturbed training banks). Candidate 0 stays the P0 sample."""
    chunks = np.asarray(chunks, np.float32)
    return np.stack([np.concatenate([c, perturb(c, r, decision)]) for c, r in zip(chunks, roots)])


def oracle_step(root, decision):
    """Mixed-execution episodes (odd roots): execute the geometry-best standard candidate on half the decisions."""
    if root % 2 == 0:
        return False
    return bool(np.random.default_rng(candidate_seed(root, decision, MIX_STREAM)).random() < 0.5)


def block_pose(branch):
    return np.array([*branch.env.block.position, branch.env.block.angle], np.float64)


# ----------------------------------------------------------------------------- scorers

class BatchScorer:
    """Scores many states' banks at once with every registered scorer.

    planner: ti_wm.cta_runtime.Planner (parent checkpoint: enc, reader, full, direct, wm; PCA; goal tokens).
    extra:   {name: (kind, module)} with kind in {"parallel", "frame", "direct"} for Round-3/4 networks, and for
             Round 6 "parallel_r" (module = (world model, its own code reader)) and "code_r" (module = a code reader
             applied to the source code of the ACTUAL future: privileged CODE tier).
    Parent scorers: FULL (reader on the actual end frame), CODE (reader on the source code of the actual future),
    CTA8E (AR world model, expected code given its greedy prefix), DIRECT8 (parent direct scorer).
    """

    PARENT = {"FULL": "future", "CODE": "code", "CTA8E": "ar_soft", "DIRECT8": "parent_direct"}
    NEEDS_FUTURE = ("future", "code", "code_r", "future_v2", "code_v2")
    # v2 checkpoints (scripts/cta_train_v2.py) carry their own encoder and readers:
    #   "frame_r"   (endpoint world model, its FULL reader)        deployable
    #   "future_v2" FULL reader on the actual future               privileged
    #   "code_v2"   (source encoder, code reader) on the actual future  privileged

    def __init__(self, planner, extra=None):
        self.p = planner
        self.spec = {name: (kind, None) for name, kind in self.PARENT.items()}
        self.spec.update(extra or {})
        self.device = planner.device
        self._step, self._rs_cache = None, (None, None)   # scores() step counter; sampled-code answers of that step

    def kind(self, name):
        return self.spec[name][0]

    def context(self, states, n=K):
        cur = np.stack([s.hist[-1]["pixels"] for s in states])
        prev = np.stack([s.hist[-2]["pixels"] for s in states])
        pos = torch.as_tensor(np.stack([np.stack([s.hist[-1]["agent_pos"], s.hist[-2]["agent_pos"]]) for s in states]),
                              device=self.device).float()
        ctx = {"cur": self.p.tokens(cur), "prev": self.p.tokens(prev, 8), "prop": proprio(pos[:, 0], pos[:, 1])}
        return {k: v.repeat_interleave(n, 0) for k, v in ctx.items()}, pos[:, 0]

    def _one(self, name, ctx, act, fut):
        kind, module = self.spec[name]
        m, goals = self.p.models, self.p.goals
        if kind == "future":
            return goal_scores(m["full"], ctx, {"end": fut["end"], "prop": fut["prop"]}, goals)
        if kind == "code":
            return goal_scores(m["reader"], ctx, m["enc"](ctx, fut), goals)
        if kind == "ar_soft":
            mem = m["wm"].encode(ctx, act)
            code = m["wm"].logits(mem, m["wm"].decode(mem)).float().softmax(-1) @ self.p.codebook
            return goal_scores(m["reader"], ctx, code, goals)
        if kind == "parent_direct":
            return goal_scores(m["direct"], ctx, act, goals)
        if kind == "parallel":
            expected, _ = module(ctx, act)
            return goal_scores(m["reader"], ctx, expected, goals)
        if kind == "parallel_r":
            wm, reader = module
            expected, _ = wm(ctx, act)
            return goal_scores(reader, ctx, expected, goals)
        if kind == "parallel_rs":
            # reader answers on codes sampled from the WM's factorized categorical (hard FSQ values, like the source
            # codes the reader was trained on); stat "mean" averages the answers, "std" is their spread.
            # One generator per call with a fixed seed: every candidate uses the same noise stream.
            # The answers are computed once per scores() step and shared by the "mean" and "std" names.
            wm, reader, stat = module
            key = (self._step, id(wm), id(reader))
            if self._step is None or self._rs_cache[0] != key:
                _, logits = wm(ctx, act)
                gen = torch.Generator(device=logits[0].device).manual_seed(0)
                answers = []
                for _ in range(CODE_SAMPLES):
                    code = torch.stack([getattr(wm, f"values_{j}")[torch.multinomial(
                        p.float().softmax(-1).flatten(0, -2), 1, generator=gen).view(p.shape[:-1])]
                        for j, p in enumerate(logits)], dim=-1)
                    answers.append(goal_scores(reader, ctx, code, goals))
                self._rs_cache = (key, torch.stack(answers))
            answers = self._rs_cache[1]
            return answers.mean(0) if stat == "mean" else answers.std(0)
        if kind == "code_r":
            return goal_scores(module, ctx, m["enc"](ctx, fut), goals)
        if kind == "frame":
            return goal_scores(m["full"], ctx, module(ctx, act), goals)
        if kind == "frame_r":
            wm, full = module
            return goal_scores(full, ctx, wm(ctx, act), goals)
        if kind == "future_v2":
            return goal_scores(module, ctx, {"end": fut["end"], "prop": fut["prop"]}, goals)
        if kind == "code_v2":
            enc, reader = module
            return goal_scores(reader, ctx, enc(ctx, fut), goals)
        if kind == "direct":
            return goal_scores(module, ctx, act, goals)
        raise ValueError(kind)

    @torch.inference_mode()
    def scores(self, states, chunks, branches, segs, names):
        """states: A states; chunks (A, B, T, 2); branches/segs: A*B simulated candidates in bank order (B = bank size).
        Returns {name: (A, B) float32}. Every candidate is scored independently of its bank mates."""
        bank = np.asarray(chunks).shape[1]
        step = max(1, STATE_CHUNK * K // bank)          # keep reader rows per call at the Round-4 size
        out = {n: np.zeros((len(states), bank), np.float32) for n in names}
        future = any(self.kind(n) in self.NEEDS_FUTURE for n in names)
        for s in range(0, len(states), step):
            part = states[s:s + step]
            b = len(part)
            ctx, agent = self.context(part, bank)
            act = action_features(torch.as_tensor(np.asarray(chunks[s:s + b]), device=self.device),
                                  agent[:, None]).flatten(0, 1)
            fut = self.p.future(branches[s * bank:(s + b) * bank], segs[s * bank:(s + b) * bank]) if future else None
            self._step = (self._step or 0) + 1
            with self.p.amp():
                for n in names:
                    if self.kind(n) == "dinowm":        # ti_wm.dinowm_scorer: reads the live envs, not our tokens
                        out[n][s:s + b] = self.spec[n][1].score([st.env for st in part], np.asarray(chunks[s:s + b]))
                        continue
                    out[n][s:s + b] = self._one(n, ctx, act, fut).float().view(b, bank).cpu().numpy()
        return out


# ----------------------------------------------------------------------------- lockstep closed loop

def base_name(arm):
    arm = arm.split(PENALTY)[0]
    return arm[:-len(RESTRICT)] if arm.endswith(RESTRICT) else arm


def deviation_scale(size):
    """sigma_k of each candidate of a bank of `size` (policy samples first, then their perturbed copies)."""
    return np.r_[np.zeros(K), np.asarray(SIGMAS, np.float64)][:size] if size > K else np.zeros(size)


def choose(arm, j, cov, geom, scores):
    """Index into the full bank. "8" oracles and "@8" arms only consider the policy's first K candidates."""
    if arm == "P0":
        return 0
    limit = K if (arm in ("PHYS8", "GEOM8") or arm.endswith(RESTRICT)) else None
    if arm in ("PHYS8", "PHYS16"):
        row = cov[j]
    elif arm in ("GEOM8", "GEOM16", "GEOM64"):
        row = geom[j]
    else:
        row = np.asarray(scores[base_name(arm)][j], np.float64)
        if PENALTY in arm:
            row = row - float(arm.split(PENALTY)[1]) * deviation_scale(len(row))
    return select_candidate([float(x) for x in row[:limit]])


def run_arm(arm, roots, runner, cloner, scorer, names, reset_branch, done, run_segment, max_decisions=None,
            bank=None, record=None, draw=K):
    """One arm on many roots in lockstep. Every decision simulates the whole bank; the arm picks the next state.

    draw:   policy samples per decision (default K). Candidate k is seeded by candidate_seed(root, d, k) only, so a
            larger draw keeps the first K candidates of the K-bank (K-scaling test).
    bank:   optional callable (policy chunks (A, K, T, 2), active roots, decision) -> (A, B, T, 2) deployment bank
            (Round 5: mixed_bank). Default: the policy's K samples.
    record: optional callable(root, decision, state_before, chunks (B, T, 2), scores {name: (B,)}, chosen) called
            before the chosen branch replaces the state (demo rendering). It must not modify its arguments.
    Returns (episodes, log). episodes: per-root dicts; log: decision-level arrays (root, decision, t, chosen,
    cov (D, B), geom (D, B), score_<name> (D, B)). Normalized score excludes reset coverage (Round-3 definition).
    """
    if arm not in ("P0",) + ORACLES and base_name(arm) not in names:
        raise ValueError(f"acting arm {arm} must also be a logged scorer")
    states = [reset_branch(r) for r in roots]
    for s in states:
        s.max_coverage = 0.
    log = {k: [] for k in ("root", "decision", "t", "chosen", "cov", "geom")}
    log.update({f"score_{n}": [] for n in names})
    timing = {"policy": 0., "simulate": 0., "score": 0.}
    d = 0
    while max_decisions is None or d < max_decisions:
        active = [i for i, s in enumerate(states) if not done(s)]
        if not active:
            break
        t0 = time.perf_counter()
        flat = runner.draw([states[i].hist for i in active for _ in range(draw)],
                           [candidate_seed(roots[i], d, k) for i in active for k in range(draw)])
        chunks = np.asarray(flat).reshape(len(active), draw, *np.asarray(flat).shape[1:])
        if bank is not None:
            chunks = bank(chunks, [roots[i] for i in active], d)
        size = chunks.shape[1]
        t1 = time.perf_counter()
        sims = [run_segment(states[i], chunks[j, k], cloner) for j, i in enumerate(active) for k in range(size)]
        branches, segs = [x[0] for x in sims], [x[1] for x in sims]
        cov = np.array([b.coverage for b in branches], np.float64).reshape(-1, size)
        geom = registration_score(np.stack([block_pose(b) for b in branches])).reshape(-1, size)
        t2 = time.perf_counter()
        scores = scorer.scores([states[i] for i in active], chunks, branches, segs, names) if names else {}
        t3 = time.perf_counter()
        for j, i in enumerate(active):
            c = choose(arm, j, cov, geom, scores)
            log["root"].append(roots[i])
            log["decision"].append(d)
            log["t"].append(states[i].t)
            log["chosen"].append(c)
            log["cov"].append(cov[j])
            log["geom"].append(geom[j])
            for n in names:
                log[f"score_{n}"].append(scores[n][j])
            if record is not None:
                record(roots[i], d, states[i], chunks[j], {n: scores[n][j] for n in names}, c)
            states[i] = branches[j * size + c]
        timing["policy"] += t1 - t0
        timing["simulate"] += t2 - t1
        timing["score"] += t3 - t2
        d += 1
    episodes = [{"root": r, "success": bool(s.success), "steps": int(s.t), "max_coverage": float(s.max_coverage),
                 "score": normalized_score(s.max_coverage, s.env.success_threshold), "reset_excluded": True}
                for r, s in zip(roots, states)]
    for s in states:
        s.env.close()
    arrays = {k: np.asarray(v) for k, v in log.items()}
    return episodes, arrays, timing


# ----------------------------------------------------------------------------- analysis

def chosen_retention(labels, chosen, roots):
    """Retained gap of the choices actually made: sum(y[chosen]-y[0]) / sum(max y - y[0]), root-cluster CI."""
    labels, chosen, roots = np.asarray(labels, float), np.asarray(chosen, int), np.asarray(roots)
    gain = labels[np.arange(len(labels)), chosen] - labels[:, 0]
    head = labels.max(1) - labels[:, 0]
    uniq = np.unique(roots)
    num = np.array([gain[roots == r].sum() for r in uniq])
    den = np.array([head[roots == r].sum() for r in uniq])
    return cluster_ratio(num, den) if den.sum() > 0 else dict.fromkeys(("ratio", "lo", "hi"), float("nan"))


def headroom(labels):
    """Mean per-decision oracle gain over candidate 0 and the share of decisions with any gain."""
    labels = np.asarray(labels, float)
    gain = labels.max(1) - labels[:, 0]
    return {"mean_gain": float(gain.mean()) if len(gain) else float("nan"),
            "share_with_gain": float((gain > 0).mean()) if len(gain) else float("nan")}
