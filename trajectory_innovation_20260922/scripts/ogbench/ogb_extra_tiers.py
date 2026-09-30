"""Extra offline tiers on a v2 OGBench checkpoint: read the code through the codec's decoder (CPU or GPU).

Question: CODE < FULL could come from the reader, not from the code. The code reader must decode where the cube went
from 16 abstract tokens and then compare it with the goal image; FULL compares the end frame with the goal image patch
by patch on the same grid. A decoded reader D(C, S, g) = FULL(C, G(C, S), g) still reads only S (G is the distortion
decoder of stage 1), so it is a legitimate CTA reader. Tiers (held-out split, own goal and the 5 official goals):
  full, code, cta, endpoint, direct (as in the ladder); code_dec = FULL on G(C, source code);
  cta_dec = FULL on G(C, expected code of the CTA world model). No training.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import ogb_cta_train_v2 as v2  # noqa: E402
from ti_wm.cta_ogb import fut_from_frames, zeros_prop  # noqa: E402

K = v2.K


@torch.inference_mode()
def main(a):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, int(a.threads)))
    blob = torch.load(a.ckpt / "cta_ogb.pt", map_location="cpu")
    cfg = blob["config"]
    mods, wms = v2.build_stage1(cfg, "cpu"), v2.build_wms(cfg, "cpu")
    for k, m in mods.items():
        m.load_state_dict(blob["stage1"][k])
        m.to(device).eval()
    for k, w in wms.items():
        w.load_state_dict(blob["wms"][k])
        w.to(device).eval()
    held = v2.Data(a.data / "heldout")
    n = held.n if not a.limit else min(a.limit, held.n)
    goals = ["own", 1, 2, 3, 4, 5]
    tiers = ("full", "code", "code_dec", "cta", "cta_dec", "endpoint", "direct")
    out = {(t, g): np.zeros((n, K), np.float32) for t in tiers for g in goals}
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if device.type == "cuda" else (lambda: torch.autocast("cpu", enabled=False))
    for s in range(0, n, a.batch):
        idx = np.arange(s, min(s + a.batch, n))
        ctx, frames, own_goal, act, _ = held.s.batch(idx, device)
        fut = fut_from_frames(frames)
        prop0 = zeros_prop(len(act), device)
        with amp():
            code = mods["enc"](ctx, fut)
            expected = wms["cta"](ctx, act)[0]
            ev = {"full": (fut, mods["full"]), "code": (code, mods["reader"]),
                  "code_dec": ({"end": mods["dec"](ctx, code)[0], "prop": prop0}, mods["full"]),
                  "cta": (expected, mods["reader"]),
                  "cta_dec": ({"end": mods["dec"](ctx, expected)[0], "prop": prop0}, mods["full"]),
                  "endpoint": (dict(wms["endpoint"](ctx, act)), mods["full"]), "direct": (act, mods["direct"])}
            for g in goals:
                goal = own_goal if g == "own" else held.goal_tok[g].to(device)[None].expand(len(act), -1, -1)
                for t in tiers:
                    x, reader = ev[t]
                    out[(t, g)][s:s + len(idx)] = reader(ctx, x, goal).float().view(len(idx), K).cpu().numpy()
        print(f"{s + len(idx)}/{n}", flush=True)
    held.n = n
    held.s.root, held.s.label, held.cubes = held.s.root[:n], held.s.label[:n], held.cubes[:n]
    rep = v2.gaps(held, out, tiers, goals, ci=True)
    summary = {t: {"own": rep[t]["own"]["retained_gap"]["ratio"], "mean5": v2.mean_gap(rep, t, [1, 2, 3, 4, 5])} for t in tiers}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "extra_tiers.json").write_text(json.dumps({"summary": summary, "ladder": rep, "ckpt": str(a.ckpt)}, indent=2))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("/mnt/data/nhatnc129/jepa/ogbench/enc/visual-cube-single-play-v0"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    main(p.parse_args())
