"""Closed-loop CTA planning on PushT (docs/CTA_E2E_PROTOCOL.md). Compute-node only.

All arms share the seeded nested bank. P0: candidate 0. PHYS8: 8-step coverage oracle. FULL8: tier-1 reader on the
actual futures. CODE8: reader on source codes of the actual futures. CTA8: reader on WM-predicted codes. DIRECT8:
D_direct(C, A, q). CTA8 and DIRECT8 are scored from (state, actions) BEFORE any candidate is simulated; the
candidates are simulated afterwards only to log their coverage.
"""

import time

import numpy as np
import torch

from ti_wm.codec import pool_grid
from ti_wm.contract import candidate_seed, select_candidate
from ti_wm.cta import CodeWM, Scorer, SourceEncoder, action_features, goal_scores, proprio
from ti_wm.pusht_runtime import MAX_STEPS, Branch, done, physical_scores, reset_branch, run_prefix
from ti_wm.sibling import project

K = 8
BEFORE = ("CTA8", "CTA8S", "CTA8E", "DIRECT8")  # scored from actions only
SAMPLES = 4                           # CTA8S: sampled codes per candidate, reader answers averaged
AFTER = ("PHYS8", "FULL8", "CODE8")   # scored from simulated candidates
KEEP = (2, 4, 6)          # intermediate frames kept per segment; the end frame (step 8) is the branch's last frame


def run_segment(branch, actions, cloner):
    """run_prefix that also returns the frames after steps KEEP (last frame repeated if the segment ends early)."""
    out = Branch(cloner(branch.env), list(branch.hist), branch.t, branch.success, branch.coverage, branch.max_coverage)
    frames = {}
    for step, action in enumerate(actions, start=1):
        if out.success or out.t >= MAX_STEPS:
            break
        obs, _, terminated, _, info = out.env.step(np.asarray(action, dtype=np.float32))
        out.t += 1
        out.hist = [out.hist[-1], obs]
        out.coverage = float(info["coverage"])
        out.max_coverage = max(out.max_coverage, out.coverage)
        out.success = bool(terminated)
        if step in KEEP:
            frames[step] = obs["pixels"]
    last = out.hist[-1]["pixels"]
    return out, np.stack([frames.get(s, last) for s in KEEP])


class Planner:
    def __init__(self, checkpoint, visual, goal_frames):
        self.visual, self.device = visual, visual.device
        blob = torch.load(checkpoint, map_location=self.device)
        cfg, state = blob["config"], blob["state"]
        m = cfg["m"]
        self.models = {"enc": SourceEncoder(m, conditional=cfg["conditional"], path=cfg["path"]),
                       "reader": Scorer("code", m=m), "full": Scorer("future"),
                       "direct": Scorer("action", layers=cfg["direct_layers"]), "wm": CodeWM(m)}
        for k, mod in self.models.items():
            mod.load_state_dict(state[k])
            mod.to(self.device).eval()
        self.mean, self.basis = blob["pca_mean"].to(self.device), blob["pca_basis"].to(self.device)
        self.goals = self.tokens(goal_frames)
        self.codebook = self.models["enc"].fsq.codebook

    def amp(self):
        return torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda")

    @torch.inference_mode()
    def tokens(self, frames, side=None):
        """(..., H, W, 3) uint8 -> (..., tokens, D) fp16, as scripts/cta_encode.py."""
        frames = np.ascontiguousarray(frames)
        lead = frames.shape[:-3]
        x = project(self.visual.features(frames.reshape(-1, *frames.shape[-3:])), self.mean, self.basis)
        x = (pool_grid(x, side) if side else x).half()
        return x.reshape(*lead, *x.shape[1:])

    def context(self, state, k):
        cur, prev = state.hist[-1], state.hist[-2]
        pos = torch.as_tensor(np.stack([cur["agent_pos"], prev["agent_pos"]]), device=self.device).float()
        ctx = {"cur": self.tokens(cur["pixels"][None]), "prev": self.tokens(prev["pixels"][None], 8),
               "prop": proprio(pos[0], pos[1])[None]}
        return {key: v.repeat_interleave(k, 0) for key, v in ctx.items()}, pos[0]

    def future(self, branches, segs=None):
        end = np.stack([b.hist[-1]["pixels"] for b in branches])
        fut = {"end": self.tokens(end),
               "prop": proprio(torch.as_tensor(np.stack([b.hist[-1]["agent_pos"] for b in branches])),
                               torch.as_tensor(np.stack([b.hist[-2]["agent_pos"] for b in branches]))).to(self.device)}
        if segs is not None:
            fut["seg"] = self.tokens(np.stack(segs), 8)
        return fut

    @torch.inference_mode()
    def scores(self, arm, state, chunks=None, branches=None, segs=None, seed=0):
        k = len(chunks) if chunks is not None else len(branches)
        ctx, agent = self.context(state, k)
        with self.amp():
            if arm in BEFORE:
                act = action_features(torch.as_tensor(np.asarray(chunks), device=self.device), agent)
                wm = self.models["wm"]
                if arm == "DIRECT8":
                    s = goal_scores(self.models["direct"], ctx, act, self.goals)
                elif arm == "CTA8":
                    code = self.codebook[wm.decode(wm.encode(ctx, act))]
                    s = goal_scores(self.models["reader"], ctx, code, self.goals)
                elif arm == "CTA8E":   # expected code under the WM given its greedy prefix (ladder tier pred_soft)
                    mem = wm.encode(ctx, act)
                    code = wm.logits(mem, wm.decode(mem)).float().softmax(-1) @ self.codebook
                    s = goal_scores(self.models["reader"], ctx, code, self.goals)
                else:   # CTA8S: the paper's sampled variant, answers (not codes) averaged; seeded per decision
                    mem = wm.encode(ctx, act)
                    torch.manual_seed(seed)
                    s = sum(goal_scores(self.models["reader"], ctx, self.codebook[wm.decode(mem, sample=True)],
                                        self.goals) for _ in range(SAMPLES)) / SAMPLES
            elif arm == "FULL8":
                s = goal_scores(self.models["full"], ctx, self.future(branches), self.goals)
            elif arm == "CODE8":
                code = self.models["enc"](ctx, self.future(branches, segs))
                s = goal_scores(self.models["reader"], ctx, code, self.goals)
            else:
                raise ValueError(arm)
        return s.float().cpu().tolist()


def run_episode(root, arm, runner, cloner, planner=None, log_candidates=True, max_decisions=None):
    """One closed-loop episode. Returns success, steps and per-decision scores/coverage."""
    state = reset_branch(root)
    decisions, d = [], 0
    while not done(state) and (max_decisions is None or d < max_decisions):
        bank = runner.bank(state.hist, [candidate_seed(root, d, k) for k in range(1 if arm == "P0" else K)])
        record = {"d": d, "t": state.t}
        if arm in BEFORE:
            t0 = time.perf_counter()
            record["score"] = planner.scores(arm, state, chunks=bank, seed=candidate_seed(root, d, 1000))
            record["score_seconds"] = time.perf_counter() - t0
        if arm == "P0":
            branches = {0: run_prefix(state, bank[0], cloner=cloner)}
            chosen = 0
        else:
            if arm == "CODE8":
                pairs = [run_segment(state, bank[j], cloner) for j in range(K)]
                sims, segs = [p[0] for p in pairs], [p[1] for p in pairs]
            elif arm in AFTER or log_candidates:
                sims, segs = [run_prefix(state, bank[j], cloner=cloner) for j in range(K)], None
            else:
                sims = None
            if sims is not None:
                record["phys"] = physical_scores(sims)
            if arm in AFTER:
                t0 = time.perf_counter()
                record["score"] = record["phys"] if arm == "PHYS8" else planner.scores(arm, state, branches=sims, segs=segs)
                record["score_seconds"] = time.perf_counter() - t0
            chosen = select_candidate(record["score"])
            branches = dict(enumerate(sims)) if sims is not None else {chosen: run_prefix(state, bank[chosen], cloner=cloner)}
        record["chosen"] = int(chosen)
        decisions.append(record)
        state = branches[chosen]
        d += 1
    return {"success": state.success, "steps": state.t, "max_coverage": state.max_coverage, "decisions": decisions}
