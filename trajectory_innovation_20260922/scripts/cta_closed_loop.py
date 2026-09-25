"""CTA closed loop on PushT (docs/CTA_E2E_PROTOCOL.md).

Arms (comma list): P0, PHYS8, FULL8, CODE8, CTA8, DIRECT8, all on the same seeded nested bank per root.
A preflight first checks that closed-loop scoring reproduces the training job's dev-ladder scores.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from ti_wm.contract import require_compute
from ti_wm.cta_runtime import Planner, run_episode
from ti_wm.pusht_runtime import CLONERS, PolicyRunner, VisualScorer
from ti_wm.wb import Logger

PREFLIGHT_DECISIONS, PREFLIGHT_MEDIAN_REL, PREFLIGHT_ARGMAX, PREFLIGHT_RHO = 32, 0.05, 0.85, 0.80
PREFLIGHT_TIERS = {"FULL8": "full", "CODE8": "code", "CTA8": "pred", "DIRECT8": "direct"}


class _Stored:
    def __init__(self, frame, pos, prev_pos):
        self.hist = [{"pixels": frame, "agent_pos": prev_pos}, {"pixels": frame, "agent_pos": pos}]


def preflight(planner, dev_shard, train_run):
    """Scores from stored dev decisions (raw frames) must match the training job's cached-feature dev scores."""
    d = np.load(dev_shard)
    saved = np.load(train_run / "dev_scores.npz")
    n = min(PREFLIGHT_DECISIONS, len(d["root"]))
    assert np.array_equal(saved["root"][:n], d["root"][:n]), "dev shard does not match the saved dev scores"
    out = {}
    for arm, tier in PREFLIGHT_TIERS.items():
        now = []
        for i in range(n):
            state = _Stored(d["ctx"][i], d["ctx_pos"][i][0], d["ctx_pos"][i][1])
            state.hist[0]["pixels"] = d["ctx_prev"][i]
            branches = [_Stored(d["end"][i, k], d["end_pos"][i, k], d["end_prev_pos"][i, k]) for k in range(8)]
            now.append(planner.scores(arm, state, chunks=d["chunk"][i], branches=branches, segs=list(d["seg"][i])))
        now, ref = np.array(now), saved[tier][:n]
        scale = max(float(ref.std()), 1e-6)
        diff = np.abs(now - ref)
        rel = diff / scale
        # bf16 kernels differ with batch size, so near-ties can flip; judge argmax only where the best candidate
        # leads by more than 10x the observed numerical noise
        top = np.sort(ref, axis=1)
        clear = (top[:, -1] - top[:, -2]) > 10 * max(float(np.median(diff)), 1e-6)
        agree = now.argmax(1) == ref.argmax(1)
        rho = [spearmanr(a, b).statistic for a, b in zip(now, ref) if np.ptp(a) > 0 and np.ptp(b) > 0]
        out[arm] = {"median_rel": float(np.median(rel)), "max_rel": float(rel.max()),
                    "argmax_agreement": float(agree[clear].mean()) if clear.any() else 1.0, "clear_banks": int(clear.sum()),
                    "all_banks_argmax_agreement": float(agree.mean()),
                    "within_bank_spearman": float(np.mean(rho)) if rho else float("nan")}
        # CTA8 decodes discrete codes greedily: bf16 noise can flip a token, so for it the within-bank rank
        # agreement is the pipeline check (a feature/proprio/action bug drives it toward 0), not argmax equality
        bad_argmax = out[arm]["argmax_agreement"] < PREFLIGHT_ARGMAX and arm != "CTA8"
        if (out[arm]["median_rel"] > PREFLIGHT_MEDIAN_REL or bad_argmax
                or not out[arm]["within_bank_spearman"] >= PREFLIGHT_RHO):
            raise RuntimeError(f"closed-loop {arm} disagrees with the training job's dev scores: {out[arm]}")
    return out


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(a):
    require_compute()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    smoke_report = json.loads((a.smoke / "smoke.json").read_text())
    assert smoke_report["status"] == "SMOKE_PASS"
    cloner = CLONERS[smoke_report["clone_method"]]
    arms = a.arms.split(",")
    runner = PolicyRunner(a.prep / "checkpoint", a.device)
    planner = Planner(a.train_run / "cta.pt", VisualScorer(a.device), np.load(a.smoke / "goal_frames.npz")["frames"])
    pre = preflight(planner, a.dev_shard, a.train_run)
    print(f"preflight: {pre}", flush=True)
    job, task = os.environ.get("SLURM_ARRAY_JOB_ID", os.environ.get("SLURM_JOB_ID")), os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    log = Logger(a.run, "closed_loop", {"arms": arms, "first": a.first, "count": a.count, "train_run": str(a.train_run),
                                        "preflight": pre, "max_decisions": a.max_decisions}, name=f"cl_{job}_{task}")
    ckpt_hash = sha256(a.train_run / "cta.pt")
    wins = {arm: 0 for arm in arms}
    done = {json.loads(line)["root"] for d in a.skip_done for f in Path(d).glob("roots_*.jsonl")
            for line in f.read_text().splitlines() if line.strip()}
    todo = [r for r in range(a.first, a.first + a.count) if r not in done]
    print(f"{len(todo)} roots to run, {a.count - len(todo)} already done in {a.skip_done}", flush=True)
    output = a.run / f"roots_{a.first}_{a.first + a.count - 1}.jsonl"
    with open(output, "w") as stream:
        for n, root in enumerate(todo):
            t0 = time.perf_counter()
            record = {"root": root, "arms": arms, "checkpoint_sha256": ckpt_hash, "preflight": pre}
            for arm in arms:
                record[arm] = run_episode(root, arm, runner, cloner, planner, log_candidates=not a.no_log_candidates,
                                          max_decisions=a.max_decisions)
                wins[arm] += record[arm]["success"]
            record["seconds"] = time.perf_counter() - t0
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            log.log({"root": root, "seconds": record["seconds"],
                     **{f"{arm}/success": float(record[arm]["success"]) for arm in arms},
                     **{f"{arm}/running_success": wins[arm] / (n + 1) for arm in arms}}, step=n)
            print(root, {arm: record[arm]["success"] for arm in arms}, f"{record['seconds']:.0f}s", flush=True)
    log.summary({"success_rate": {arm: wins[arm] / max(len(todo), 1) for arm in arms}, "preflight": pre})
    log.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("run", "prep", "smoke", "train_run", "dev_shard"):
        p.add_argument(f"--{name.replace('_', '-')}", dest=name, type=Path, required=True)
    p.add_argument("--first", type=int, required=True)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--arms", default="P0,FULL8,CTA8,DIRECT8")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-decisions", type=int, default=None, help="smoke only")
    p.add_argument("--no-log-candidates", action="store_true", help="CTA8/DIRECT8: skip simulating unchosen candidates")
    p.add_argument("--skip-done", type=Path, nargs="*", default=[], help="skip roots already recorded in these runs")
    main(p.parse_args())
