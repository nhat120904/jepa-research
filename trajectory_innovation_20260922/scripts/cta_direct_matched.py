"""Parameter-matched direct scorer for the PushT v2 comparison (scripts/cta_train_v2.py, checkpoint 55616).

The v2 DIRECT (8 layers) has 6.57 M parameters against 10.95 M for CTA (world model + code reader). This trains only a
deeper direct scorer D_direct(C, A, g) (--layers, default 14 -> ~11.3 M) with everything else as the v2 stage 1:
same banks and sources, same batch stream (same seed and the same rng calls: copy_norms, then one train_batch per
step), same ranking loss, dropout, optimizer, schedule, steps and checkpoint selection on the v2 selection banks.
Output: direct_matched.pt ({"config", "direct"}) and its offline ladder on the selection banks. Compute node only.
"""
import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cta_train_v2 as v2  # noqa: E402
from ti_wm.contract import require_compute  # noqa: E402
from ti_wm.cta import Scorer  # noqa: E402
from ti_wm.sibling import rank_loss  # noqa: E402


def main(a):
    require_compute()
    device = torch.device("cuda")
    amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    ref = json.loads((a.v2 / "config.json").read_text())
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    cfg = {k: ref[k] for k in ("m", "lr", "wd", "warmup", "dropout", "steps1", "decisions", "spread_frac", "eval_every",
                               "select_limit", "sources", "selection", "target")}
    cfg.update(direct_layers=a.layers, seed=a.seed, reference=str(a.v2), job=os.environ.get("SLURM_JOB_ID"))
    if a.smoke:                                       # code path only, not a result
        cfg.update(steps1=40, eval_every=20, select_limit=40)
    if ref.get("hindsight", 0.0) or ref.get("aux") or ref.get("success_bonus", 0.0):
        raise ValueError("reference run used v2.1 options; not reproduced here")
    banks = [v2.Bank(Path(p), n) for n, p in cfg["sources"].items()]
    pool = v2.Pool(banks)
    sel = [v2.Bank(Path(p), n) for n, p in cfg["selection"].items()]
    goals = torch.from_numpy(np.load(v2.GOALS)).to(device)
    v2.copy_norms(pool, rng)                          # same rng stream as the reference run's stage 1
    direct = Scorer("action", layers=a.layers, dropout=cfg["dropout"]).to(device)
    cfg["params"] = int(sum(p.numel() for p in direct.parameters()))
    a.run.mkdir(parents=True, exist_ok=True)
    logf = (a.run / "metrics.jsonl").open("a")
    log = lambda d: (logf.write(json.dumps(d) + "\n"), logf.flush(), print(d, flush=True))
    log({"config": cfg})
    opt = torch.optim.AdamW(direct.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = v2.schedule(opt, cfg["steps1"], cfg["warmup"])
    sel_goals = goals[list(v2.SELECT_GOALS)]
    mods = {"direct": direct}
    best, t0 = (-float("inf"), None, None), time.perf_counter()
    for step in range(cfg["steps1"]):
        direct.train()
        ctx, fut, act, y, goal = v2.train_batch(pool, goals, rng, cfg, device)
        with amp():
            s = direct(ctx, act, goal).float().view(-1, v2.K)
        loss = rank_loss(s, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(direct.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0:
            log({"step": step, "rank_direct": float(loss)})
        if (step + 1) % cfg["eval_every"] == 0 or step + 1 == cfg["steps1"]:
            direct.eval()
            sc = v2.score_banks(sel, ("direct",), mods, {}, sel_goals, device, amp, cfg["select_limit"])
            val = v2.pooled_gap(sel, sc, "direct", cfg["select_limit"])["retained_gap"]["ratio"]
            val = val if math.isfinite(val) else -float("inf")
            if val > best[0] or best[2] is None:
                best = (val, step + 1, copy.deepcopy({k: t.detach().cpu() for k, t in direct.state_dict().items()}))
            log({"step": step + 1, "selection_gap": val})
    direct.load_state_dict(best[2])
    direct.eval().requires_grad_(False)
    cfg.update(selected={"gap": best[0], "step": best[1]}, hours=(time.perf_counter() - t0) / 3600)
    torch.save({"config": cfg, "direct": direct.state_dict()}, a.run / "direct_matched.pt")
    lim = 40 if a.smoke else None
    sc = v2.score_banks(sel, ("direct",), mods, {}, goals, device, amp, lim)
    ladder = {"pooled": v2.pooled_gap(sel, sc, "direct", lim, ci=True)}
    for j, b in enumerate(sel):
        ladder[b.name] = v2.pooled_gap([b], {"direct": [sc["direct"][j]]}, "direct", lim, ci=True)
    (a.run / "offline_ladder.json").write_text(json.dumps({"config": cfg, "ladder": ladder}, indent=2))
    print(json.dumps({"params": cfg["params"], "selected": cfg["selected"],
                      "retained_gap": {k: v["retained_gap"] for k, v in ladder.items()}}, indent=1), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--v2", type=Path, required=True, help="reference v2 training dir (config.json)")
    p.add_argument("--layers", type=int, default=14)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    main(p.parse_args())
