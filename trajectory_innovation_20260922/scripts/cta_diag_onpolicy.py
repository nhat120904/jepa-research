"""On-policy closed-loop diagnostic with full candidate logging (docs/CTA_ONPOLICY_DATA_PROTOCOL_20260927.md).

closed:    run acting arms on a root range in lockstep; at every decision simulate all 8 candidates, record native
           coverage + geometry labels, and score the bank with every logged scorer (cross-scoring).
aggregate: success / normalized score with paired CIs, and the on-policy ranking matrix: each scorer's retained gap
           on the states each arm actually visits, for both labels.
"""
import argparse
import functools
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cta_closed_loop as closed  # noqa: E402
import cta_round3 as r3  # noqa: E402
from cta_reader_refine import write_json  # noqa: E402
from ti_wm.contract import candidate_seed, require_compute  # noqa: E402
from ti_wm.cta import Scorer  # noqa: E402
from ti_wm.cta_batch import K, ORACLES, BatchScorer, base_name, chosen_retention, headroom, mixed_bank, run_arm  # noqa: E402
from ti_wm.cta_eval import ranking_metrics  # noqa: E402
from ti_wm.cta_parallel import EndpointWM, ParallelFSQWM  # noqa: E402
from ti_wm.cta_runtime import Planner, keep_steps, run_segment  # noqa: E402
from ti_wm.gates import mcnemar_exact, paired_diff  # noqa: E402
from ti_wm.pusht_runtime import CLONERS, PolicyRunner, VisualScorer, done, reset_branch  # noqa: E402
from ti_wm.pusht_canonical import CanonicalPolicyRunner  # noqa: E402

R3_NETS = {"nll": ("NLL8", "parallel"), "task": ("CTA3", "parallel"), "frame": ("FRAME8", "frame"),
           "direct": ("DIRECT3", "direct")}
R4_NETS = {"nll": ("NLL4", "parallel"), "task": ("CTA4", "parallel"), "task_std": ("CTA4S", "parallel"),
           "direct": ("DIRECT4", "direct")}
ONPOLICY8_NETS = {"task": ("CTA8O", "parallel"), "direct": ("DIRECT8O", "direct"),
                  "frame": ("ENDPOINT8O", "frame")}
CHECK_TOL = {"median_rel": 0.05, "rho": 0.80}


def load_round6(path, device):
    """Round-6 networks (scripts/cta_round6.py): CTA6 = task WM read by the Round-6 reader, CODE6 = that reader on
    the actual future's source code (privileged tier), DIRECT6 = direct scorer."""
    blob = torch.load(path, map_location="cpu")
    cfg = blob["config"]
    reader, task = Scorer("code", m=cfg["m"]), ParallelFSQWM(m=cfg["m"])
    direct = Scorer("action", layers=cfg["direct_layers"])
    for key, net in (("reader", reader), ("task", task), ("direct", direct)):
        net.load_state_dict(blob["networks"][key], strict=True)
        net.to(device).eval().requires_grad_(False)
    return {"CTA6": ("parallel_r", (task, reader)), "CODE6": ("code_r", reader), "DIRECT6": ("direct", direct)}


def load_v2(path, device):
    """v2 checkpoint (scripts/cta_train_v2.py): every module retrained; its own encoder and readers."""
    from ti_wm.cta import FutureDecoder, SourceEncoder  # noqa: F401
    blob = torch.load(path, map_location="cpu")
    cfg = blob["config"]
    chunk = cfg.get("chunk", 8)                     # executed actions per candidate the checkpoint was trained on
    mods = {"enc": SourceEncoder(cfg["m"]), "reader": Scorer("code", m=cfg["m"]), "full": Scorer("future"),
            "direct": Scorer("action", layers=cfg["direct_layers"], chunk=chunk)}
    for k, net in mods.items():
        net.load_state_dict(blob["stage1"][k], strict=True)
    wm, endpoint = ParallelFSQWM(m=cfg["m"], chunk=chunk), EndpointWM(chunk=chunk)
    wm.load_state_dict(blob["wms"]["cta"], strict=True)
    endpoint.load_state_dict(blob["wms"]["endpoint"], strict=True)
    for net in list(mods.values()) + [wm, endpoint]:
        net.to(device).eval().requires_grad_(False)
    return {"CTAV2": ("parallel_r", (wm, mods["reader"])), "ENDV2": ("frame_r", (endpoint, mods["full"])),
            "CTAV2S": ("parallel_rs", (wm, mods["reader"], "mean")),   # answers averaged over sampled codes
            "CTAV2SD": ("parallel_rs", (wm, mods["reader"], "std")),   # their spread (uncertainty, not a selector)
            "DIRV2": ("direct", mods["direct"]), "FULLV2": ("future_v2", mods["full"]),
            "CODEV2": ("code_v2", (mods["enc"], mods["reader"]))}


def load_nets(path, mapping, device):
    blob = torch.load(path, map_location="cpu")
    layers = blob["config"]["direct_layers"]
    out, modules = {}, {}
    for key, (name, kind) in mapping.items():
        net = {"parallel": lambda: ParallelFSQWM(m=blob["config"]["m"]), "frame": EndpointWM,
               "direct": lambda: Scorer("action", layers=layers)}[kind]()
        net.load_state_dict(blob["networks"][key], strict=True)
        net.to(device).eval().requires_grad_(False)
        out[name], modules[key] = (kind, net), net
    return out, modules, blob.get("config", {})


def consistency_check(scorer, r3_planner, runner, cloner, roots, names):
    """Batched scores on live states must reproduce the validated sequential Planner/R3Planner paths."""
    sequential = {"FULL": "FULL8", "CODE": "CODE8", "CTA8E": "CTA8E", "DIRECT8": "DIRECT8",
                  "CTA3": "CTA3", "NLL8": "NLL8", "FRAME8": "FRAME8", "DIRECT3": "DIRECT3"}
    states = [reset_branch(r) for r in roots]
    flat = runner.draw([s.hist for s in states for _ in range(K)],
                       [candidate_seed(r, 0, k) for r in roots for k in range(K)])
    chunks = np.asarray(flat).reshape(len(roots), K, *np.asarray(flat).shape[1:])
    sims = [run_segment(states[j], chunks[j, k], cloner) for j in range(len(roots)) for k in range(K)]
    branches, segs = [x[0] for x in sims], [x[1] for x in sims]
    checked = [n for n in names if n in sequential]
    batched = scorer.scores(states, chunks, branches, segs, checked)
    report = {}
    for n in checked:
        ref = np.array([r3_planner.scores(sequential[n], states[j], chunks=chunks[j],
                                          branches=branches[j * K:(j + 1) * K], segs=segs[j * K:(j + 1) * K])
                        for j in range(len(roots))])
        now = batched[n]
        diff = np.abs(now - ref)
        rel = diff / max(float(ref.std()), 1e-6)
        # Rank agreement only where the bank's score spread clearly exceeds the numerical noise; near-tied banks
        # reorder under bf16/batch-size noise without any pipeline error (55138_1: median_rel .002, rho .66).
        noise = max(float(np.median(diff)), 1e-6)
        clear = [(a, b) for a, b in zip(now, ref) if np.ptp(b) > 10 * noise and np.ptp(a) > 0]
        rho = [spearmanr(a, b).statistic for a, b in clear]
        report[n] = {"median_rel": float(np.median(rel)), "max_rel": float(rel.max()),
                     "rho_clear_banks": float(np.mean(rho)) if rho else float("nan"), "clear_banks": len(rho)}
        if report[n]["median_rel"] > CHECK_TOL["median_rel"] or (rho and not report[n]["rho_clear_banks"] >= CHECK_TOL["rho"]):
            raise RuntimeError(f"batched scorer {n} disagrees with the sequential path: {report[n]}")
    for s in states + branches:
        s.env.close()
    return report


def run_closed(a):
    require_compute()
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    smoke = json.loads((a.smoke / "smoke.json").read_text())
    if smoke["status"] != "SMOKE_PASS":
        raise ValueError("Invalid clone contract")
    cloner = CLONERS[smoke["clone_method"]]
    runner_type = CanonicalPolicyRunner if getattr(a, "proposal_mode", "batched") == "canonical" else PolicyRunner
    runner = runner_type(a.prep / "checkpoint", "cuda", n_exec=a.n_exec)
    n_exec = runner.end - runner.start
    keep = keep_steps(n_exec)
    segment = functools.partial(run_segment, keep=keep)
    if n_exec != 8:
        # docs/CTA_REPLAN_INTERVAL_PROTOCOL.md: every learned scorer must be trained on chunks of the executed length.
        if a.r3 or a.r4 or a.r6 or a.onpolicy8 or a.extra_direct:
            raise ValueError("8-step networks (r3/r4/r6/onpolicy8/extra-direct) cannot score longer chunks")
        if set(a.log_scorers.split(",")) & set(BatchScorer.PARENT):
            raise ValueError("parent 8-step scorers cannot score longer chunks")
    goal_frames = np.load(a.smoke / "goal_frames.npz")["frames"]
    visual = VisualScorer("cuda")
    planner = Planner(a.parent / "cta.pt", visual, goal_frames)
    extra, hashes = {}, {"parent": closed.sha256(a.parent / "cta.pt")}
    r3_modules = None
    if a.r3:
        more, r3_modules, _ = load_nets(a.r3 / "round3.pt", R3_NETS, device)
        extra.update(more)
        hashes["round3"] = closed.sha256(a.r3 / "round3.pt")
    if a.r4:
        more, _, _ = load_nets(a.r4 / "round4.pt", R4_NETS, device)
        extra.update(more)
        hashes["round4"] = closed.sha256(a.r4 / "round4.pt")
    if a.r6:
        extra.update(load_round6(a.r6 / "round6.pt", device))
        hashes["round6"] = closed.sha256(a.r6 / "round6.pt")
    if a.onpolicy8:
        more, _, _ = load_nets(a.onpolicy8 / "onpolicy8.pt", ONPOLICY8_NETS, device)
        extra.update(more)
        hashes["onpolicy8"] = closed.sha256(a.onpolicy8 / "onpolicy8.pt")
    if a.v2:
        v2_chunk = torch.load(a.v2 / "cta_v2.pt", map_location="cpu")["config"].get("chunk", 8)
        if v2_chunk != n_exec:
            raise ValueError(f"v2 checkpoint trained on {v2_chunk}-step chunks, closed loop executes {n_exec}")
        extra.update(load_v2(a.v2 / "cta_v2.pt", device))
        hashes["v2"] = closed.sha256(a.v2 / "cta_v2.pt")
    for spec in a.extra_direct or []:
        name, path = spec.split("=", 1)
        blob = torch.load(path, map_location="cpu")
        net = Scorer("action", layers=blob["config"]["direct_layers"])
        net.load_state_dict(blob["direct"], strict=True)
        extra[name] = ("direct", net.to(device).eval().requires_grad_(False))
        hashes[name] = closed.sha256(Path(path))
    if a.dinowm:
        from ti_wm.dinowm_scorer import DinoWMScorer
        render = reset_branch(a.first)
        extra["DINOWM"] = ("dinowm", DinoWMScorer(a.dino_root, a.dinowm, device, goal_frames, render.env,
                                                  macro=math.ceil(n_exec / 5)))
        render.env.close()
        hashes["dinowm"] = closed.sha256(a.dinowm / "checkpoints" / "model_latest.pth")
    scorer = BatchScorer(planner, extra)
    arms, names = a.arms.split(","), [n for n in a.log_scorers.split(",") if n]   # "" = no learned scorer
    for arm in arms:
        if arm not in ("P0",) + ORACLES and base_name(arm) not in names:
            raise ValueError(f"acting arm {arm} is not logged")
    bank = {"policy": None, "mixed": mixed_bank}[a.bank]
    report = {"status": "RUNNING", "job": os.environ.get("SLURM_JOB_ID"), "roots": [a.first, a.first + a.count - 1],
              "arms": arms, "log_scorers": names, "hashes": hashes, "max_decisions": a.max_decisions,
              "bank": a.bank, "draw": a.draw, "n_exec": n_exec, "keep_steps": list(keep),
              "proposal_mode": getattr(a, "proposal_mode", "batched"),
              "proposal_microbatch": getattr(runner, "proposal_microbatch", None)}
    write_json(a.run / "closed_report.json", report)
    if r3_modules is not None:
        r3_planner = r3.R3Planner(a.parent, visual, goal_frames, r3_modules)
        report["consistency"] = consistency_check(scorer, r3_planner, runner, cloner, [a.first, a.first + 1], names)
        del r3_planner
        print({"consistency": report["consistency"]}, flush=True)
        write_json(a.run / "closed_report.json", report)
    roots = list(range(a.first, a.first + a.count))
    report["timing"] = {}
    with (a.run / "episodes.jsonl").open("x") as stream:
        for arm in arms:
            t0 = time.perf_counter()
            episodes, log, timing = run_arm(arm, roots, runner, cloner, scorer, names, reset_branch, done,
                                            segment, a.max_decisions, bank=bank, draw=a.draw)
            np.savez(a.run / f"log_{arm}.npz", **log)
            for e in episodes:
                stream.write(json.dumps({"arm": arm, **e}) + "\n")
            stream.flush()
            report["timing"][arm] = {**timing, "seconds": time.perf_counter() - t0}
            print(arm, {"successes": sum(e["success"] for e in episodes), "decisions": len(log["root"]),
                        **report["timing"][arm]}, flush=True)
            write_json(a.run / "closed_report.json", report)
    report["status"] = "DONE"
    write_json(a.run / "closed_report.json", report)


def load_shards(closed_run):
    episodes, logs, reports = {}, {}, []
    for shard in sorted(Path(closed_run).glob("shard_*")):
        rep = json.loads((shard / "closed_report.json").read_text())
        if rep["status"] != "DONE":
            raise ValueError(f"incomplete shard {shard}")
        reports.append(rep)
        for line in (shard / "episodes.jsonl").read_text().splitlines():
            if line.strip():
                e = json.loads(line)
                episodes.setdefault(e["arm"], {})[e["root"]] = e
        for arm in rep["arms"]:
            with np.load(shard / f"log_{arm}.npz") as z:
                part = {k: z[k] for k in z.files}
            for k, v in part.items():
                logs.setdefault(arm, {}).setdefault(k, []).append(v)
    logs = {arm: {k: np.concatenate(v) for k, v in d.items()} for arm, d in logs.items()}
    return episodes, logs, reports


def strip(metrics):
    metrics = dict(metrics)
    metrics.pop("chosen", None)
    return metrics


def merge_runs(closed_runs):
    """Load one or more closed-loop runs. An arm present in several runs (e.g. P0) must have identical outcomes and
    identical choice sequences on every root; then pairing across runs is exact. Returns episodes, logs, reports,
    names per arm."""
    episodes, logs, reports, names, mismatch = {}, {}, [], {}, []
    for index, run in enumerate(closed_runs):
        ep, lg, rep = load_shards(run)
        if any(r["arms"] != rep[0]["arms"] or r["hashes"] != rep[0]["hashes"] for r in rep):
            raise ValueError(f"Inconsistent experiment identity across shards of {run}")
        for arm in rep[0]["arms"]:
            key = arm
            if arm in episodes:
                same = (sorted(ep[arm]) == sorted(episodes[arm])
                        and all(ep[arm][r]["success"] == episodes[arm][r]["success"] for r in ep[arm])
                        and lg[arm]["cov"].shape == logs[arm]["cov"].shape
                        and np.array_equal(lg[arm]["chosen"], logs[arm]["chosen"])
                        and np.allclose(lg[arm]["cov"], logs[arm]["cov"]))
                if same:
                    continue            # identical trajectories: keep one copy, cross-run pairing is exact
                key = f"{arm}@run{index}"
                mismatch.append(arm)
            episodes[key], logs[key], names[key] = ep[arm], lg[arm], rep[0]["log_scorers"]
        reports += rep
    return episodes, logs, reports, names, mismatch


def aggregate(a):
    runs = a.closed_run if isinstance(a.closed_run, (list, tuple)) else [a.closed_run]
    episodes, logs, reports, arm_names, mismatch = merge_runs(runs)
    arms = list(episodes)
    roots = sorted(episodes[arms[0]])
    if a.expect_roots and roots != list(range(a.expect_roots[0], a.expect_roots[1] + 1)):
        raise ValueError(f"Unexpected roots: {roots[:3]}...{roots[-3:]} ({len(roots)})")
    if any(sorted(episodes[arm]) != roots for arm in arms):
        raise ValueError("Arms cover different roots")
    success = {arm: np.array([episodes[arm][r]["success"] for r in roots], float) for arm in arms}
    score = {arm: np.array([episodes[arm][r]["score"] for r in roots]) for arm in arms}
    out = {"status": "DONE", "n_roots": len(roots), "arms": {}, "contrasts": {}, "onpolicy": {},
           "runs": [str(r) for r in runs], "hashes": [r["hashes"] for r in reports],
           "cross_run_mismatch": mismatch,   # arms repeated in several runs whose trajectories differed
           "cross_run_pairing_exact": not mismatch,
           "consistency": [r.get("consistency") for r in reports],
           "scope": "development roots, single training seed; batched runner (internal pairing only)"}
    for arm in arms:
        out["arms"][arm] = {"successes": int(success[arm].sum()), "mean_score": float(score[arm].mean()),
                            "mean_max_coverage": float(np.mean([episodes[arm][r]["max_coverage"] for r in roots])),
                            "decisions": int(len(logs[arm]["root"])),
                            "override_fraction": float(np.mean(logs[arm]["chosen"] != 0)),
                            "seconds": float(sum(r["timing"][arm]["seconds"] for r in reports if arm in r["timing"]))}
    pairs = [(arm, "P0") for arm in arms if arm != "P0"] + [tuple(p.split("-")) for p in a.pairs.split(",") if p]
    for left, right in pairs:
        if left in arms and right in arms:
            out["contrasts"][f"{left}-{right}"] = {"score": paired_diff(score[left], score[right]),
                                                   "success": paired_diff(success[left], success[right]),
                                                   "mcnemar": mcnemar_exact(success[left], success[right])}
    for arm in arms:
        L = logs[arm]
        entry = {}
        for label in ("geom", "cov"):
            y = L[label]
            entry[label] = {"headroom": headroom(y), "own_choices": chosen_retention(y, L["chosen"], L["root"]),
                            "scorers": {n: strip(ranking_metrics(L[f"score_{n}"], y, L["root"], ci=True))
                                        for n in arm_names[arm]}}
        out["onpolicy"][arm] = entry
    names = list(dict.fromkeys(n for arm in arms for n in arm_names[arm]))
    write_json(a.run / "summary.json", out)
    lines = ["| arm | success | score | override | own geom retention | own cov retention |", "|---|---:|---:|---:|---:|---:|"]
    for arm in arms:
        g, c = out["onpolicy"][arm]["geom"]["own_choices"], out["onpolicy"][arm]["cov"]["own_choices"]
        v = out["arms"][arm]
        lines.append(f"| {arm} | {v['successes']} | {v['mean_score']:.3f} | {v['override_fraction']:.2f} | "
                     f"{g['ratio']:.2f} | {c['ratio']:.2f} |")
    lines += ["", "Geometry retained gap of each scorer (columns) on each arm's visited states (rows):", "",
              "| states of \\ scorer | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for arm in arms:
        sc = out["onpolicy"][arm]["geom"]["scorers"]
        cells = [f"{sc[n]['retained_gap']['ratio']:.2f}" if n in sc else "—" for n in names]
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    lines += ["", "Contrasts (score mean [95% CI]; success mean; McNemar p):", ""]
    for k, v in out["contrasts"].items():
        s, u = v["score"], v["success"]
        lines.append(f"- {k}: score {s['mean']:+.3f} [{s['lo']:+.3f}, {s['hi']:+.3f}]; success {u['mean']:+.2f} "
                     f"[{u['lo']:+.2f}, {u['hi']:+.2f}]; p={v['mcnemar']['p']:.3f}")
    (a.run / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("closed", "aggregate"), required=True)
    for key in ("run", "parent", "r3", "r4", "r6", "onpolicy8", "v2", "dinowm", "prep", "smoke"):
        p.add_argument(f"--{key}", type=Path, required=key == "run")
    p.add_argument("--closed-run", type=Path, nargs="+", help="aggregate: one or more closed-loop runs to merge")
    p.add_argument("--dino-root", type=Path, default=Path("/mnt/data/nhatnc129/jepa/dino_wm_original"))
    p.add_argument("--extra-direct", nargs="*", help="NAME=path of scripts/cta_direct_matched.py checkpoints to log")
    p.add_argument("--first", type=int)
    p.add_argument("--count", type=int)
    p.add_argument("--arms", default="P0,PHYS8,GEOM8,FULL,CTA3,NLL8,FRAME8,DIRECT3")
    p.add_argument("--log-scorers", default="FULL,CODE,CTA8E,DIRECT8,CTA3,NLL8,FRAME8,DIRECT3")
    p.add_argument("--pairs", default="GEOM8-PHYS8,CTA3-DIRECT3,CTA3-NLL8,NLL8-DIRECT3,FULL-GEOM8")
    p.add_argument("--expect-roots", type=int, nargs=2)
    p.add_argument("--max-decisions", type=int, default=None, help="smoke only")
    p.add_argument("--draw", type=int, default=K, help="policy samples per decision (K-scaling: 64 keeps the K-bank "
                                                       "as candidates 0-7)")
    p.add_argument("--n-exec", type=int, default=None,
                   help="executed actions per decision (default: native 8); docs/CTA_REPLAN_INTERVAL_PROTOCOL.md")
    p.add_argument("--proposal-mode", choices=("batched", "canonical"), default="batched",
                   help="canonical: fixed proposal shapes independent of active roots and candidate count")
    p.add_argument("--bank", choices=("policy", "mixed"), default="policy",
                   help="mixed: Round-5 deployment bank = 8 policy samples + their 8 perturbed copies")
    args = p.parse_args()
    require_compute()
    args.run.mkdir(parents=True, exist_ok=True)
    {"closed": run_closed, "aggregate": aggregate}[args.mode](args)
