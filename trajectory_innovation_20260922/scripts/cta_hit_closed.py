"""Canonical, deployable-only closed loop for the L=15 native-hit pilot.

Run on a Slurm compute node. Learned scores are computed from C,A before
candidate simulation, with a fixed one-root scoring shape. Future branch data
are used only by the simulator/oracle and diagnostic labels.
"""

import argparse
import functools
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ti_wm.contract import candidate_seed, require_compute
from ti_wm.cta import Scorer, action_features
from ti_wm.cta_batch import BatchScorer, K, run_arm
from ti_wm.cta_parallel import EndpointWM, ParallelFSQWM
from ti_wm.cta_runtime import Planner, keep_steps, run_segment
from ti_wm.pusht_canonical import CanonicalPolicyRunner
from ti_wm.pusht_runtime import CLONERS, DINO_HUB, MAX_STEPS, VisualScorer, done, reset_branch


TI = Path("/mnt/data/nhatnc129/jepa/trajectory_innovation")
DEFAULT_SPECS = {"BASE": (("CTA_BASE", "cta"), ("END_BASE", "endpoint"), ("DIR_BASE", "direct")),
                 "CTRL": (("CTA_CTRL", "cta"), ("END_CTRL", "endpoint"), ("DIR_CTRL", "direct")),
                 "HIT": (("CTA_HIT", "cta"), ("END_HIT", "endpoint"), ("DIR_HIT", "direct"))}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str) + "\n")
    tmp.replace(path)


def dependency_versions():
    versions = {}
    for name in ("torch", "numpy", "gym-pusht", "gymnasium", "pymunk", "pygame", "diffusers", "lerobot"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def model_state_sha256(model):
    """Hash the visual weights actually loaded, on the compute node only."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}\n".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def checkpoint_args(items):
    paths = {}
    for item in items:
        label, path = item.split("=", 1)
        if not label or label in paths:
            raise ValueError(f"Invalid/duplicate checkpoint label {label!r}")
        paths[label] = Path(path).resolve()
    return paths


def arm_specs(items, labels):
    if not items:
        return {name: (label, kind) for label in labels for name, kind in DEFAULT_SPECS.get(label, ())}
    out = {}
    for item in items:
        name, spec = item.split("=", 1)
        label, kind = spec.rsplit(":", 1)
        if name in out or name in ("P0", "GEOM8") or label not in labels:
            raise ValueError(f"Invalid arm specification {item}")
        if kind not in ("cta", "endpoint", "direct"):
            raise ValueError(f"Unsupported evidence channel {kind}")
        out[name] = (label, kind)
    return out


class TokenHost(Planner):
    """Reuse validated tokenization, without loading historical planner weights."""

    def __init__(self, pca, visual, goal_frames):
        self.visual, self.device = visual, visual.device
        blob = torch.load(pca, map_location="cpu", weights_only=False)
        self.mean = blob["pca_mean"].to(self.device)
        self.basis = blob["pca_basis"].to(self.device)
        self.models = {}  # no source encoder, future reader, or parent predictor
        self.goals = self.tokens(goal_frames)

    def future(self, *args, **kwargs):
        raise RuntimeError("Deployable pilot cannot tokenize future observations")


def load_deployable(paths, specs, n_exec, device):
    """Only instantiate modules used by an explicit deployable arm."""
    blobs, cache, extra, identity = {}, {}, {}, {}
    for label in sorted({label for label, _ in specs.values()}):
        path = paths[label]
        blob = torch.load(path, map_location="cpu", weights_only=False)
        cfg = blob["config"]
        if cfg.get("chunk", 8) != n_exec:
            raise ValueError(f"{label} trained for {cfg.get('chunk', 8)} steps; deployment executes {n_exec}")
        for key in ("stage1", "wms"):
            if key not in blob:
                raise ValueError(f"{label} missing v2 checkpoint key {key}")
        blobs[label] = blob
        identity[label] = {"path": str(path), "sha256": sha256(path), "config": cfg,
                           "loaded_modules": []}

    def module(label, key):
        if (label, key) not in cache:
            blob, cfg = blobs[label], blobs[label]["config"]
            constructors = {"reader": lambda: Scorer("code", m=cfg["m"]),
                            "full": lambda: Scorer("future"),
                            "direct": lambda: Scorer("action", layers=cfg["direct_layers"], chunk=n_exec),
                            "cta": lambda: ParallelFSQWM(m=cfg["m"], chunk=n_exec),
                            "endpoint": lambda: EndpointWM(chunk=n_exec)}
            net = constructors[key]()
            group = "wms" if key in ("cta", "endpoint") else "stage1"
            net.load_state_dict(blob[group][key], strict=True)
            cache[label, key] = net.to(device).eval().requires_grad_(False)
            identity[label]["loaded_modules"].append(key)
        return cache[label, key]

    for name, (label, kind) in specs.items():
        if kind == "cta":
            extra[name] = ("parallel_r", (module(label, "cta"), module(label, "reader")))
        elif kind == "endpoint":
            extra[name] = ("frame_r", (module(label, "endpoint"), module(label, "full")))
        else:
            extra[name] = ("direct", module(label, "direct"))
    return extra, identity


class DeployableScorer(BatchScorer):
    """Fixed root microbatch and separately timed context/net inference."""

    def __init__(self, host, extra):
        super().__init__(host, extra)
        self.spec = dict(extra)  # remove every historical/privileged parent arm
        self.reset_costs()

    def reset_costs(self):
        self.costs = {"shared_context_seconds": 0., "context_calls": 0,
                      "scorer_seconds": {name: 0. for name in self.spec},
                      "scorer_calls": {name: 0 for name in self.spec}}

    def sync(self):
        torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def scores(self, states, chunks, branches, segs, names):
        if branches is not None or segs is not None:
            raise RuntimeError("Learned inference must precede candidate simulation")
        chunks = np.asarray(chunks)
        if chunks.shape[1:3] != (K, 15):
            raise ValueError(f"Expected native K8/L15 bank, got {chunks.shape}")
        out = {n: np.empty((len(states), K), np.float32) for n in names}
        for j, state in enumerate(states):
            self.sync()
            t0 = time.perf_counter()
            ctx, agent = self.context([state], K)
            act = action_features(torch.as_tensor(chunks[j:j + 1], device=self.device), agent[:, None]).flatten(0, 1)
            self.sync()
            self.costs["shared_context_seconds"] += time.perf_counter() - t0
            self.costs["context_calls"] += 1
            with self.p.amp():
                for name in names:
                    t0 = time.perf_counter()
                    values = self._one(name, ctx, act, None).float().reshape(K).cpu().numpy()
                    if not np.isfinite(values).all():
                        raise FloatingPointError(f"Nonfinite scores for {name}")
                    out[name][j] = values
                    self.costs["scorer_seconds"][name] += time.perf_counter() - t0
                    self.costs["scorer_calls"][name] += 1
        return out


class CachedScores:
    """run_arm adapter: ignores simulation data and returns precomputed scores."""

    def __init__(self, runner):
        self.runner = runner
        self.native_hit = []

    def scores(self, states, chunks, branches, segs, names):
        cache = self.runner.cache
        if cache is None or cache["hist_ids"] != [id(s.hist) for s in states]:
            raise RuntimeError("Pre-simulation score/state alignment failed")
        if not np.array_equal(chunks, cache["chunks"]) or list(names) != cache["names"]:
            raise RuntimeError("Pre-simulation score/action alignment failed")
        # Labels are collected only after learned scores have already been
        # computed. No branch object is passed to a learned model.
        self.native_hit.extend(np.asarray([b.success for b in branches], bool).reshape(len(states), K))
        self.runner.cache = None
        return cache["scores"]


class ScoringPolicy:
    """Policy wrapper scores C,A immediately after draw, before physics."""

    def __init__(self, policy, scorer, names):
        self.policy, self.scorer, self.names = policy, scorer, list(names)
        self.reset_costs()

    def reset_costs(self):
        self.policy_seconds = 0.
        self.policy_root_banks = 0
        self.cache = None
        self.scorer.reset_costs()

    def draw(self, hists, seeds):
        if self.cache is not None:
            raise RuntimeError("Previous score cache was not consumed")
        if len(hists) != len(seeds) or len(hists) % K:
            raise ValueError("run_arm draw must contain whole K8 root banks")
        root_hists = hists[::K]
        if any(hists[j + k] is not hists[j] for j in range(0, len(hists), K) for k in range(K)):
            raise RuntimeError("Policy history/candidate alignment failed")
        self.scorer.sync()
        t0 = time.perf_counter()
        actions = self.policy.draw(hists, seeds)
        self.scorer.sync()
        self.policy_seconds += time.perf_counter() - t0
        self.policy_root_banks += len(root_hists)
        chunks = np.asarray(actions).reshape(len(root_hists), K, 15, 2)
        states = [SimpleNamespace(hist=h) for h in root_hists]
        scores = self.scorer.scores(states, chunks, None, None, self.names)
        self.cache = {"hist_ids": [id(h) for h in root_hists], "chunks": chunks.copy(),
                      "names": self.names, "scores": scores}
        return actions


def compare_logs(left, right, root):
    result = {}
    for key in left:
        a, b = left[key][left["root"] == root], right[key][right["root"] == root]
        same_shape = a.shape == b.shape
        exact = same_shape and bool(np.array_equal(a, b))
        # Native polygon coverage has demonstrated float64-ULP label noise.
        ok = exact if key != "cov" else same_shape and bool(np.allclose(a, b, atol=1e-12, rtol=0))
        result[key] = {"equal": exact, "contract_pass": ok,
                       "max_abs": float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())
                       if same_shape and a.size else None}
    return result


def qualification(policy, scorer, cloner, segment, roots, decisions, out):
    states = [reset_branch(r) for r in roots[:2]]
    report = {"roots": roots[:2], "decisions": decisions, "scope": "development smoke only"}
    try:
        first = states[0]
        seeds = [candidate_seed(roots[0], 0, k) for k in range(K)]
        bank = policy.bank(first.hist, seeds)
        longer = policy.bank(first.hist, [candidate_seed(roots[0], 0, k) for k in range(16)])
        single = policy.bank(first.hist, seeds[:1])
        paired = policy.draw([s.hist for s in states for _ in range(K)],
                             [candidate_seed(r, 0, k) for r in roots[:2] for k in range(K)])
        paired = np.asarray(paired).reshape(2, K, 15, 2)
        report["proposal"] = {"prefix_1_8": bool(np.array_equal(single, bank[:1])),
                              "prefix_8_16": bool(np.array_equal(bank, longer[:K])),
                              "grouping": bool(np.array_equal(bank, paired[0]))}
        names = list(scorer.spec)
        alone = scorer.scores([first], bank[None], None, None, names)
        together = scorer.scores(states, paired, None, None, names)
        report["learned_score_grouping"] = {n: {"bitwise_equal": bool(np.array_equal(alone[n][0], together[n][0])),
                                                    "max_abs": float(np.abs(alone[n][0] - together[n][0]).max())}
                                            for n in names}
    finally:
        for state in states:
            state.env.close()
    report["anchors"] = {}
    for arm in ("P0", "GEOM8"):
        logs, outcomes = [], []
        for selected in ([roots[0]], roots[:2]):
            native_labels = []
            def labelled_segment(*args, **kwargs):
                branch, frames = segment(*args, **kwargs)
                native_labels.append(bool(branch.success))
                return branch, frames
            episodes, log, _ = run_arm(arm, selected, policy, cloner, None, [], reset_branch, done, labelled_segment,
                                       max_decisions=decisions)
            log["native_hit"] = np.asarray(native_labels, bool).reshape(-1, K)
            suffix = "alone" if len(selected) == 1 else "paired"
            np.savez(out / f"qualification_{arm}_{suffix}.npz", **log)
            logs.append(log)
            outcomes.append(next(ep for ep in episodes if ep["root"] == roots[0]))
        report["anchors"][arm] = compare_logs(*logs, roots[0])
        report["anchors"][arm]["episode_outcome"] = {
            "contract_pass": all(outcomes[0][key] == outcomes[1][key]
                                 for key in ("root", "success", "steps", "reset_excluded")),
            "alone": outcomes[0], "paired": outcomes[1]}
    checks = (all(report["proposal"].values())
              and all(v["bitwise_equal"] for v in report["learned_score_grouping"].values())
              and all(v["contract_pass"] for d in report["anchors"].values() for v in d.values()))
    report["status"] = "PASS" if checks else "FAIL"
    write_json(out / "qualification.json", report)
    if not checks:
        raise RuntimeError("Canonical pilot numerical qualification failed; inspect qualification.json")
    return report


def main(a):
    require_compute()
    if a.n_exec != 15 or a.count < 2 or a.first < 2200 or a.first + a.count > 2400:
        raise ValueError("This pilot supports L15 and at least two development roots in 2200–2399 only")
    a.run.mkdir(parents=True, exist_ok=True)
    if (a.run / "episodes.jsonl").exists() or (a.run / "closed_report.json").exists():
        raise FileExistsError("Refusing to overwrite a pilot run")
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    paths = checkpoint_args(a.checkpoint)
    specs = arm_specs(a.arm_spec, paths)
    arms = [n for n in a.arms.split(",") if n]
    if len(set(arms)) != len(arms) or any(n not in {"P0", "GEOM8", *specs} for n in arms):
        raise ValueError(f"Invalid acting arms {arms}")
    extra, identity = load_deployable(paths, specs, a.n_exec, device)
    smoke = json.loads((a.smoke / "smoke.json").read_text())
    if smoke["status"] != "SMOKE_PASS":
        raise ValueError("Invalid simulator clone contract")
    cloner = CLONERS[smoke["clone_method"]]
    goal_path = a.smoke / "goal_frames.npz"
    with np.load(goal_path) as data:
        goals = data["frames"]
    if len(goals) != 16:
        raise ValueError(f"Expected deployment average over 16 goal images, got {len(goals)}")
    visual = VisualScorer(device)
    host = TokenHost(a.pca, visual, goals)
    scorer = DeployableScorer(host, extra)
    policy = CanonicalPolicyRunner(a.prep / "checkpoint", device, n_exec=a.n_exec)
    if policy.end - policy.start != a.n_exec or policy.proposal_microbatch != K:
        raise ValueError("Canonical policy alignment/microbatch contract changed")
    roots = list(range(a.first, a.first + a.count))
    names = list(specs)
    segment = functools.partial(run_segment, keep=keep_steps(a.n_exec))
    report = {"schema": "cta_hit_closed_v1", "status": "RUNNING", "job": os.environ["SLURM_JOB_ID"],
              "roots": roots, "arms": arms, "log_scorers": names, "checkpoint_specs": specs,
              "checkpoints": identity, "n_exec": a.n_exec, "draw": K, "bank": "policy",
              "proposal_mode": "canonical", "proposal_microbatch": policy.proposal_microbatch,
              "clone_method": smoke["clone_method"],
              "score_root_microbatch": 1, "goal_images": len(goals), "max_decisions": a.max_decisions,
              "hashes": {"pca": sha256(a.pca), "goal_frames": sha256(goal_path),
                         "smoke": sha256(a.smoke / "smoke.json"),
                         "preparation": sha256(a.prep / "preparation.json"),
                         "visual_model_state": model_state_sha256(visual.model),
                         "visual_source": {str(p.relative_to(DINO_HUB)): sha256(p)
                                           for p in sorted(Path(DINO_HUB).rglob('*.py'))},
                         "policy_files": {str(p.relative_to(a.prep / 'checkpoint')): sha256(p)
                                          for p in sorted((a.prep / 'checkpoint').rglob('*')) if p.is_file()},
                         "script": sha256(Path(__file__)),
                         "runtime": {p.name: sha256(p) for p in sorted((Path(__file__).resolve().parents[1] / 'ti_wm').glob('*.py'))}},
              "backend": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                          "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                          "gpu": torch.cuda.get_device_name(device), "cudnn_deterministic": True,
                          "dependencies": dependency_versions(),
                          "cudnn_benchmark": False, "cudnn_tf32": True, "matmul_tf32": False,
                          "reader_autocast": "bfloat16", "cached_visual_dtype": "float16"},
              "learned_input_contract": "Scores computed from observed C and proposed A before simulation; reader C,S,g",
              "timing": {}}
    write_json(a.run / "closed_report.json", report)
    if a.qualify_decisions > 0:
        report["qualification"] = qualification(policy, scorer, cloner, segment, roots,
                                                 a.qualify_decisions, a.run)
        write_json(a.run / "closed_report.json", report)
    runner = ScoringPolicy(policy, scorer, names)
    cached = CachedScores(runner)
    with (a.run / "episodes.jsonl").open("x") as stream:
        for arm in arms:
            runner.reset_costs()
            cached.native_hit = []
            t0 = time.perf_counter()
            episodes, log, diagnostic = run_arm(arm, roots, runner, cloner, cached, names, reset_branch, done,
                                                 segment, a.max_decisions)
            if runner.cache is not None:
                raise RuntimeError("Unconsumed final score cache")
            for ep in episodes:
                ep["terminated"] = bool(ep["success"] or ep["steps"] >= MAX_STEPS)
                ep["truncated_by_max_decisions"] = not ep["terminated"]
                stream.write(json.dumps({"arm": arm, **ep}) + "\n")
            stream.flush()
            log["native_hit"] = np.asarray(cached.native_hit, bool)
            np.savez(a.run / f"log_{arm}.npz", **log)
            costs = {**scorer.costs, "proposal_seconds": runner.policy_seconds,
                     "proposal_root_banks": runner.policy_root_banks,
                     "simulator_seconds": diagnostic["simulate"], "cache_read_seconds": diagnostic["score"],
                     "diagnostic_wall_seconds": time.perf_counter() - t0,
                     "run_arm_draw_including_cross_scoring_seconds": diagnostic["policy"],
                     "deployment_timing_claim": False}
            report["timing"][arm] = costs
            report.setdefault("completed_arms", []).append(arm)
            write_json(a.run / "closed_report.json", report)
            print(json.dumps({"arm": arm, "successes": sum(e["success"] for e in episodes),
                              "n": len(episodes), "decisions": len(log["root"]), "timing": costs}), flush=True)
    report["status"] = "DONE"
    write_json(a.run / "closed_report.json", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--checkpoint", action="append", required=True, help="LABEL=/absolute/path/cta_v2.pt")
    p.add_argument("--arm-spec", action="append", help="NAME=LABEL:cta|endpoint|direct; defaults BASE/CTRL/HIT")
    p.add_argument("--arms", default="P0,GEOM8,CTA_BASE,END_BASE,DIR_BASE,CTA_CTRL,END_CTRL,DIR_CTRL,CTA_HIT,END_HIT,DIR_HIT")
    p.add_argument("--prep", type=Path, default=TI / "prepare_53776")
    p.add_argument("--smoke", type=Path, default=TI / "gate_smoke_53803")
    p.add_argument("--pca", type=Path, default=TI / "cta_feat_54490/pca.pt")
    p.add_argument("--first", type=int, default=2200)
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--n-exec", type=int, default=15)
    p.add_argument("--max-decisions", type=int, help="Smoke only; truncated outputs excluded from success inference")
    p.add_argument("--qualify-decisions", type=int, default=2)
    main(p.parse_args())
