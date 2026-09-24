"""Gate C2 stage 1: collect P0 rollouts, encode, train the hindsight progress reader (+ goal-only control)."""

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from ti_wm.contract import candidate_seed, require_compute
from ti_wm.progress import ProgressReader, episode_bounds, pool_tokens, proprio_vector, sample_pairs
from ti_wm.pusht_runtime import CLONERS, PolicyRunner, VisualScorer, done, reset_branch, run_prefix

TRAIN_ROOTS = range(20000, 20800)
VAL_ROOTS = range(20800, 21000)
STEPS, BATCH, LR, WD = 20_000, 256, 3e-4, 0.05
ENV_BATCH = 50
VAL_PAIRS = 20_000


def snapshot(state, episode):
    return {"episode": episode, "t": state.t, "cur": state.hist[-1]["pixels"], "prev": state.hist[-2]["pixels"],
            "proprio": proprio_vector(state.hist[-1]["agent_pos"], state.hist[-2]["agent_pos"]),
            "success": state.success}


def collect(runner, roots, cloner):
    """Batched candidate-0 rollouts; store the state before every decision plus the final state."""
    records, outcomes = [], []
    roots = list(roots)
    for start in range(0, len(roots), ENV_BATCH):
        chunk = roots[start:start + ENV_BATCH]
        states = [reset_branch(r) for r in chunk]
        per_episode = [[] for _ in chunk]
        d = 0
        while True:
            active = [i for i, s in enumerate(states) if not done(s)]
            if not active:
                break
            for i in active:
                per_episode[i].append(snapshot(states[i], chunk[i]))
            chunks = runner.draw([states[i].hist for i in active], [candidate_seed(chunk[i], d, 0) for i in active])
            for j, i in enumerate(active):
                states[i] = run_prefix(states[i], chunks[j], cloner)
            d += 1
        for i, s in enumerate(states):
            per_episode[i].append(snapshot(s, chunk[i]))
            records.extend(per_episode[i])
            outcomes.append({"root": chunk[i], "success": s.success, "steps": s.t})
        print(f"collected {start + len(chunk)}/{len(roots)}", flush=True)
    return records, outcomes


def encode(visual, records):
    def feats(key):
        out = []
        for s in range(0, len(records), 512):
            frames = np.stack([r[key] for r in records[s:s + 512]])
            out.append(pool_tokens(visual.features(frames)).half())
        return torch.cat(out)
    return {"cur": feats("cur"), "prev": feats("prev"),
            "proprio": torch.from_numpy(np.stack([r["proprio"] for r in records])).to(visual.device),
            "times": np.array([r["t"] for r in records]), "episode": np.array([r["episode"] for r in records])}


def batch_inputs(data, i, j, blank):
    cur, prev, prop = data["cur"][i], data["prev"][i], data["proprio"][i]
    if blank:
        cur, prev, prop = torch.zeros_like(cur), torch.zeros_like(prev), torch.zeros_like(prop)
    return cur, prev, data["cur"][j], prop


def train(data, blank, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    end_of = episode_bounds(data["episode"])
    model = ProgressReader().to(data["cur"].device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    losses = []
    for step in range(STEPS):
        i, j, y = sample_pairs(data["times"], end_of, rng, BATCH)
        i_t, j_t = torch.from_numpy(i).to(model.cls.device), torch.from_numpy(j).to(model.cls.device)
        pred = model(*batch_inputs(data, i_t, j_t, blank))
        loss = torch.nn.functional.mse_loss(pred, torch.from_numpy(y).to(pred.device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(float(loss))
        if step % 2000 == 0:
            print(f"{'goal-only' if blank else 'full'} step {step} loss {np.mean(losses[-200:]):.4f}", flush=True)
    return model.eval(), float(np.mean(losses[-500:]))


@torch.inference_mode()
def validate(model, data, blank):
    rng = np.random.default_rng(12345)
    i, j, y = sample_pairs(data["times"], episode_bounds(data["episode"]), rng, VAL_PAIRS)
    preds = []
    for s in range(0, VAL_PAIRS, 1024):
        it = torch.from_numpy(i[s:s + 1024]).to(model.cls.device)
        jt = torch.from_numpy(j[s:s + 1024]).to(model.cls.device)
        preds.append(model(*batch_inputs(data, it, jt, blank)).float().cpu().numpy())
    p = np.concatenate(preds)
    steps_err = np.abs(np.expm1(p) - np.expm1(y))
    return {"spearman": float(spearmanr(p, y).statistic), "mse_log": float(np.mean((p - y) ** 2)),
            "median_abs_err_steps": float(np.median(steps_err)), "pairs": VAL_PAIRS}


def main(run, prep, smoke, seed=0, control=True):
    require_compute()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    report = {"status": "RUNNING"}
    try:
        smoke_report = json.loads((smoke / "smoke.json").read_text())
        assert smoke_report["status"] == "SMOKE_PASS"
        cloner = CLONERS[smoke_report["clone_method"]]
        runner = PolicyRunner(prep / "checkpoint")
        visual = VisualScorer()
        t0 = time.perf_counter()
        train_rec, train_out = collect(runner, TRAIN_ROOTS, cloner)
        val_rec, val_out = collect(runner, VAL_ROOTS, cloner)
        report["collection_seconds"] = time.perf_counter() - t0
        report["train_episodes"] = {"n": len(train_out), "success": sum(o["success"] for o in train_out),
                                    "states": len(train_rec)}
        report["val_episodes"] = {"n": len(val_out), "success": sum(o["success"] for o in val_out),
                                  "states": len(val_rec)}
        (run / "outcomes.json").write_text(json.dumps({"train": train_out, "val": val_out}))
        train_data, val_data = encode(visual, train_rec), encode(visual, val_rec)
        del train_rec, val_rec
        t0 = time.perf_counter()
        report["seed"] = seed
        full, full_loss = train(train_data, blank=False, seed=seed)
        report["final_train_loss"] = {"full": full_loss}
        report["val"] = {"full": validate(full, val_data, False)}
        if control:
            goal_only, goal_loss = train(train_data, blank=True, seed=seed)
            report["final_train_loss"]["goal_only"] = goal_loss
            report["val"]["goal_only"] = validate(goal_only, val_data, True)
            torch.save({"state_dict": goal_only.state_dict()}, run / "reader_goal_only.pt")
        report["training_seconds"] = time.perf_counter() - t0
        torch.save({"state_dict": full.state_dict(), "config": {"width": 256, "layers": 4, "heads": 4},
                    "seed": seed}, run / "reader.pt")
        report["status"] = "TRAINED"
    except Exception:
        report["status"] = "FAILED"
        report["error"] = traceback.format_exc()
        raise
    finally:
        (run / "train_report.json").write_text(json.dumps(report, indent=2, default=float))
        print(json.dumps(report, indent=2, default=float), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prep", type=Path, required=True)
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-control", action="store_true")
    a = parser.parse_args()
    main(a.run, a.prep, a.smoke, a.seed, not a.no_control)
