"""GPU smoke for gates A-C: equivalence, nesting, clone exactness, timing, dev goal map."""

import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from ti_wm.arms import run_arm
from ti_wm.contract import candidate_seed, require_compute
from ti_wm.pusht_runtime import (
    BANK, CLONERS, LIVE_TOLERANCE, PolicyRunner, VisualScorer, clone_integrity, final_frames, physical_state,
    reset_branch, run_prefix,
)

DEV_ROOTS = range(5000, 5032)
GOAL_FRAMES = 16
MIN_GOAL_FRAMES = 8


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def official_equivalence(runner, seeds):
    """Bitwise: select_action's first chunk vs our denoise with an identically seeded CUDA generator."""
    from lerobot.common.envs.utils import preprocess_observation

    diffs = []
    for s in seeds:
        branch = reset_branch(s)
        obs = branch.hist[-1]
        runner.policy.reset()
        torch.manual_seed(s)
        batch = preprocess_observation({"pixels": obs["pixels"][None], "agent_pos": obs["agent_pos"][None]})
        batch = {k: v.to(runner.device) for k, v in batch.items()}
        with torch.inference_mode():
            first = runner.policy.select_action(batch)
        official = torch.stack([first] + list(runner.policy._queues["action"]), dim=1)
        cond = runner.policy.diffusion._prepare_global_conditioning(runner.batch([branch.hist]))
        ours = runner._actions(cond, [torch.Generator(runner.device).manual_seed(s)])
        diffs.append(float((official - ours).abs().max()))
    return diffs


def clone_checks(runner, cloner, roots, decisions=10):
    """(a) two clones + same chunk -> identical; (b) clone vs stepping the LIVE object in place.

    The live object has been stepped along the trajectory, so (b) detects any lost contact cache.
    """
    repeat, versus_source, integrity = [], [], []
    for root in roots:
        state = reset_branch(root)
        for d in range(decisions):
            integrity.append(clone_integrity(state.env, cloner))
            chunk = runner.bank(state.hist, [candidate_seed(root, d, 0)])[0]
            a, b = run_prefix(state, chunk, cloner), run_prefix(state, chunk, cloner)
            repeat.append(float(np.abs(physical_state(a.env) - physical_state(b.env)).max()
                                + np.abs(a.hist[-1]["pixels"].astype(int) - b.hist[-1]["pixels"]).max()))
            steps = 0
            for action in chunk:
                if steps >= a.t - state.t:
                    break
                state.env.step(np.asarray(action, dtype=np.float32))
                steps += 1
            versus_source.append(float(np.abs(physical_state(state.env) - physical_state(a.env)).max()))
            state = a
            if state.success or state.t >= 300:
                break
    return {"repeat_max": max(repeat), "vs_source_max": max(versus_source),
            "vs_source_nonzero": int(sum(v > 0 for v in versus_source)), "n": len(repeat),
            "pre_step_max_abs": max(i["pre_step_max_abs"] for i in integrity),
            "bodies_in_space": all(i["agent_in_space"] and i["block_in_space"] for i in integrity),
            "cog_equal": all(i["cog_equal"] for i in integrity)}


def main(run, prep):
    require_compute()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    report = {"status": "RUNNING"}
    try:
        preparation = json.loads((prep / "preparation.json").read_text())
        ckpt = prep / "checkpoint"
        weights_sha = sha256(ckpt / "model.safetensors")
        assert weights_sha == preparation["checkpoint_hashes"]["model.safetensors"], "checkpoint hash drift"
        runner = PolicyRunner(ckpt)
        report["load"] = {"missing": runner.missing, "unexpected": runner.unexpected,
                          "removed_fields": runner.removed_fields, "parameters": runner.parameters,
                          "weights_sha256": weights_sha, "slice": [runner.start, runner.end]}
        assert not runner.missing and not runner.unexpected

        report["official_equivalence_max_abs"] = official_equivalence(runner, [5100, 5101, 5102])

        state = reset_branch(5103)
        seeds = [candidate_seed(5103, 0, k) for k in range(BANK)]
        b32, b8, b1 = runner.bank(state.hist, seeds), runner.bank(state.hist, seeds[:8]), runner.bank(state.hist, seeds[:1])
        report["nested"] = {"k8_vs_k32_max_abs": float(np.abs(b32[:8] - b8).max()),
                            "k1_vs_k32_max_abs": float(np.abs(b32[:1] - b1).max()),
                            "bank_shape": list(b32.shape)}

        report["clone"] = {}
        for name, fn in CLONERS.items():
            try:
                report["clone"][name] = clone_checks(runner, fn, [5004, 5005])
            except Exception:
                report["clone"][name] = {"error": traceback.format_exc()}
        ok = [n for n in CLONERS if report["clone"][n].get("repeat_max") == 0.0
              and report["clone"][n].get("vs_source_max", float("inf")) <= LIVE_TOLERANCE]
        report["clone_method"] = ok[0] if ok else None
        report["clone_rule"] = f"first of {list(CLONERS)} with repeat_max == 0 and vs_source_max <= {LIVE_TOLERANCE}"
        assert report["clone_method"], "no clone method is both repeatable and faithful to the live continuation"
        cloner = CLONERS[report["clone_method"]]

        scorer = VisualScorer()
        dino_weights = Path("/mnt/data/nhatnc129/jepa/cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth")
        report["dino_weights_sha256"] = sha256(dino_weights)

        # Timing and bank diversity on a short P0-style walk.
        timing = {"bank32_s": [], "branch32_s": [], "dino32_s": []}
        diversity, distinct_phys = [], []
        state = reset_branch(5006)
        scorer.set_goal(torch.zeros(256, 384))
        for d in range(12):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            bank = runner.bank(state.hist, [candidate_seed(5006, d, k) for k in range(BANK)])
            torch.cuda.synchronize(); t1 = time.perf_counter()
            branches = [run_prefix(state, bank[k], cloner) for k in range(BANK)]
            t2 = time.perf_counter()
            scorer.scores(final_frames(branches)); torch.cuda.synchronize(); t3 = time.perf_counter()
            timing["bank32_s"].append(t1 - t0); timing["branch32_s"].append(t2 - t1); timing["dino32_s"].append(t3 - t2)
            flat = bank.reshape(BANK, -1)
            dist = np.linalg.norm(flat[:, None] - flat[None], axis=-1)
            diversity.append(float(dist[np.triu_indices(BANK, 1)].mean() / np.sqrt(bank.shape[1])))
            distinct_phys.append(len(set(round(b.coverage, 12) for b in branches[:8])) > 1)
            state = branches[0]
            if state.success or state.t >= 300:
                break
        report["timing_mean_s"] = {k: float(np.mean(v)) for k, v in timing.items()}
        report["bank_diversity_mean_step_l2_px"] = diversity
        report["decisions_with_distinct_phys_k8"] = f"{sum(distinct_phys)}/{len(distinct_phys)}"

        # Development goal map (protocol: first 16 successful P0 dev episodes, terminal frames).
        dev, frames = [], []
        t0 = time.perf_counter()
        for root in DEV_ROOTS:
            out = run_arm(root, "P0", runner, scorer=None, cloner=cloner, keep_states=True)
            dev.append({"root": root, "success": out["success"], "steps": out["steps"]})
            if out["success"] and len(frames) < GOAL_FRAMES:
                frames.append(out["_final"].hist[-1]["pixels"])
        report["dev_p0_seconds"] = time.perf_counter() - t0
        report["dev"] = {"episodes": dev, "success": sum(e["success"] for e in dev), "n": len(dev),
                         "goal_frames": len(frames)}
        assert len(frames) >= MIN_GOAL_FRAMES, "too few dev successes for the goal map"
        frames = np.stack(frames)
        goal_map = scorer.features(frames).mean(dim=0).float().cpu()
        np.savez_compressed(run / "goal_frames.npz", frames=frames)
        torch.save(goal_map, run / "goal_map.pt")
        report["goal_map_sha256"] = sha256(run / "goal_map.pt")

        per_root = report["dev_p0_seconds"] / len(DEV_ROOTS)
        report["projection"] = {"p0_seconds_per_root_incl_32_branches": per_root,
                                "note": "main per root ~ OFFICIAL + P0 + PHYS8 + VIS8 + MEDOID8 + H_rep"}
        report["status"] = "SMOKE_PASS"
    except Exception:
        report["status"] = "SMOKE_FAILED"
        report["error"] = traceback.format_exc()
        raise
    finally:
        (run / "smoke.json").write_text(json.dumps(report, indent=2, default=str))
        print(json.dumps({k: v for k, v in report.items() if k != "dev"}, indent=2, default=str), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prep", type=Path, required=True)
    args = parser.parse_args()
    main(args.run, args.prep)
