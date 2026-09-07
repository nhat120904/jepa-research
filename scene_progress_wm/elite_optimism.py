"""Does CEM select the candidates the cost is most wrong about?

This is the instrument gate.  Every later question -- whether an expectile with
``tau > 0.5`` helps, whether a belief encoder ranks better than a frame encoder,
whether a predictive-state objective beats a recurrent LeWM -- is a question
about *candidate ranking*, and none of them can be answered by a validation MAE.
The five progress arms differ by 0.005 in expectile loss and 0.003 in MAE while
plausibly differing a great deal in which action chunk they put first.

So the instrument measures ranking directly, against ground truth obtained by
executing candidates in the simulator and then letting OGBench's own Markov
controllers finish the task.  Three properties make the numbers comparable
across cost arms:

* **Scale-free on the model side.**  ``latent_l2`` returns a latent distance and
  a progress head returns a normalised remaining time in ``[0, 1]``; the two are
  not comparable as numbers.  Only the *order* a cost induces is used, so the
  primary metric is a pairwise inversion rate, not a calibration error.

* **A margin in physical units.**  A pair only counts when the true costs differ
  by more than ``delta`` environment steps, so near-ties -- where an inversion is
  meaningless -- are excluded rather than counted as errors.  ``delta`` is swept
  rather than chosen.

* **A ratio, not a level.**  ``c_true`` comes from a fixed repair order, not a
  search, so it is an *upper bound* on the true remaining cost and the absolute
  inversion rate is inflated.  The headline quantity is therefore the
  amplification ratio between strata, where that bias is common and cancels.

The strata answer the actual question.  If the inversion rate on final elites is
no higher than on the initial proposal distribution, CEM is not seeking out the
cost's blind spots and model exploitation is not the bottleneck.  If it is
markedly higher, the cost is being selected against, and that is the effect any
conservative objective has to remove.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from scene_progress_wm.scene_render import env_successes


PROTOCOL = "scene_progress_wm_elite_optimism_v1"

#: Ordered so that a rising inversion rate across the first four reads as
#: "optimisation is walking into the cost's error".  ``selected`` is a single
#: chunk per replan and is used only for regret, never in the equal-n contrasts.
STRATA = ("random_init", "gen_mid", "gen_last", "elite", "selected")
POPULATION_STRATA = STRATA[:-1]

#: The drawer position ``DrawerMarkovOracle`` is driven to when the goal cube
#: has to be placed inside it first.  ``ScenePredicates`` calls the drawer open
#: at ``<= -0.12``; -0.16 is the value OGBench's own Scene tasks use.
DRAWER_OPEN_POS = -0.16
DRAWER_OPEN_THRESHOLD = -0.12


# --------------------------------------------------------------------------
# ground truth
# --------------------------------------------------------------------------
class OracleRepair:
    """Environment steps for OGBench's own controllers to satisfy the goal.

    The skills are the four Markov oracles Scene ships with, parameterised by
    the *goal row* rather than by a task definition, so an arbitrary dataset
    goal is reachable: the cube oracle is pointed at the goal cube position,
    the drawer and window oracles at the goal joint values.  That is what makes
    this usable as ``c_true`` on the progress-WM arena, where a goal is a row of
    play data and not one of the five scripted tasks.

    Two properties are deliberate and must be quoted wherever the number is:

    * The repair **order** is fixed (cube, then drawer, then window, then the
      buttons that lock them), not searched.  The result is an upper bound on
      the minimum remaining cost, not the minimum.
    * A repair that has not reached the goal within ``max_skills`` is reported
      as **censored** rather than as a large finite cost, so a failure of the
      oracle cannot masquerade as a hard candidate.

    The buttons are locks: state ``1`` leaves the drawer or window free and
    state ``0`` holds it, which is why every drawer or window repair is preceded
    by an unlock and why the buttons are repaired last.
    """

    def __init__(self, raw: Any, max_skills: int = 8) -> None:
        from ogbench.manipspace.oracles.markov.button_markov import ButtonMarkovOracle
        from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle
        from ogbench.manipspace.oracles.markov.drawer_markov import DrawerMarkovOracle
        from ogbench.manipspace.oracles.markov.window_markov import WindowMarkovOracle

        self.raw = raw
        self.max_skills = int(max_skills)
        self.oracle_classes = {
            "button": ButtonMarkovOracle,
            "cube": CubeMarkovOracle,
            "drawer": DrawerMarkovOracle,
            "window": WindowMarkovOracle,
        }
        # the per-skill step caps the Scene gate already validated
        self.max_steps = {"button": 70, "drawer": 85, "window": 85, "cube": 200}

    # -- skill configuration ------------------------------------------------
    def _configure(self, kind: str, target: Any) -> dict[str, Any]:
        import mujoco

        info = self.raw.compute_ob_info()
        if kind == "button":
            index = int(target)
            desired = (int(self.raw._cur_button_states[index]) + 1) % 2
            info.update(
                {
                    "privileged/target_button": index,
                    "privileged/target_button_state": desired,
                    "privileged/target_button_top_pos": self.raw._data.site_xpos[
                        self.raw._button_site_ids[index]
                    ].copy(),
                }
            )
            return info

        if kind == "drawer":
            self.raw._model.site("drawer_handle_center_target").pos[1] = float(target)
            mujoco.mj_kinematics(self.raw._model, self.raw._data)
            info = self.raw.compute_ob_info()
            info["privileged/target_drawer_handle_pos"] = self.raw._data.site_xpos[
                self.raw._drawer_target_site_id
            ].copy()
            return info

        if kind == "window":
            self.raw._model.site("window_handle_center_target").pos[0] = float(target)
            mujoco.mj_kinematics(self.raw._model, self.raw._data)
            info = self.raw.compute_ob_info()
            info["privileged/target_window_handle_pos"] = self.raw._data.site_xpos[
                self.raw._window_target_site_id
            ].copy()
            return info

        if kind == "cube":
            info.update(
                {
                    "privileged/target_block": 0,
                    "privileged/target_block_pos": np.asarray(target, dtype=np.float64),
                    "privileged/target_block_yaw": np.array([0.0]),
                }
            )
            return info

        raise ValueError(f"unknown skill kind: {kind}")

    def _refresh(self, target_fields: dict[str, Any]) -> dict[str, Any]:
        # Rebuild the moving proprio/object fields but hold the target fixed for
        # the whole closed-loop skill: recomputing a button target after it has
        # toggled would request a second toggle.
        info = self.raw.compute_ob_info()
        info.update(target_fields)
        return info

    def _drawer_is_open(self) -> bool:
        return float(self.raw._data.joint("drawer_slide").qpos[0]) <= DRAWER_OPEN_THRESHOLD

    def _next_skill(self, match: dict[str, bool], goal_state, goal_buttons):
        """The next repair, or ``None`` when nothing in the library applies."""
        buttons = np.asarray(self.raw._cur_button_states)

        if not match["cube"]:
            goal_cube = np.asarray(goal_state["cube_pos"], dtype=np.float64)
            # a cube whose goal is inside the drawer needs the drawer open first
            if self.raw._is_in_drawer(goal_cube) and not self._drawer_is_open():
                if int(buttons[0]) != 1:
                    return ("button", 0)
                return ("drawer", DRAWER_OPEN_POS)
            return ("cube", goal_cube)

        if not match["drawer"]:
            if int(buttons[0]) != 1:
                return ("button", 0)
            return ("drawer", float(goal_state["drawer"]))

        if not match["window"]:
            if int(buttons[1]) != 1:
                return ("button", 1)
            return ("window", float(goal_state["window"]))

        # locks last: setting a button back to 0 is what freezes the joint it guards
        if not match["button_0"]:
            return ("button", 0)
        if not match["button_1"]:
            return ("button", 1)
        return None

    def _run(self, skill) -> int:
        kind, target = skill
        info = self._configure(kind, target)
        target_fields = {
            key: copy.deepcopy(value)
            for key, value in info.items()
            if key.startswith("privileged/target_")
        }
        oracle = self.oracle_classes[kind](env=self.raw, min_norm=0.4)
        oracle.reset(None, info)
        oracle._max_step = self.max_steps[kind]
        # drop the oracle's random post-skill pose; it is not part of the cost
        oracle._final_pos = np.array([0.53, 0.15, 0.31], dtype=np.float64)
        oracle._final_yaw = 0.0

        steps = 0
        for _ in range(self.max_steps[kind]):
            action = np.asarray(oracle.select_action(None, info), dtype=np.float32)
            _, _, _, truncated, _ = self.raw.step(action)
            if truncated:
                raise RuntimeError("Scene rollout truncated inside an oracle repair")
            steps += 1
            # stop the moment the goal predicate holds, otherwise finishing the
            # skill would charge steps the task did not need
            if env_successes(self.raw)["success"]:
                break
            if oracle.done:
                break
            info = self._refresh(target_fields)
        return steps

    def steps_to_goal(self, goal_state, goal_buttons) -> dict[str, Any]:
        """Repair the current state to the goal; report the step count."""
        total = 0
        trace: list[dict[str, Any]] = []
        for _ in range(self.max_skills):
            match = env_successes(self.raw)
            if match["success"]:
                return {"steps": total, "censored": False, "trace": trace}
            skill = self._next_skill(match, goal_state, goal_buttons)
            if skill is None:
                break
            steps = self._run(skill)
            total += steps
            trace.append({"skill": skill[0], "env_steps": int(steps)})
        return {
            "steps": total,
            "censored": not env_successes(self.raw)["success"],
            "trace": trace,
        }


# --------------------------------------------------------------------------
# candidate capture
# --------------------------------------------------------------------------
def make_recorder(record_steps, topk: int):
    """A ``CEMSolver`` callback that keeps candidates from chosen generations.

    Built here rather than at import time because the base class lives in the
    vendored ``stable_worldmodel`` and importing it at module scope would make
    this file unusable outside a configured job.
    """
    from stable_worldmodel.planning.solver.callbacks.common import Callback

    class CandidateRecorder(Callback):
        name = "candidate_recorder"
        output_key = "candidate_recorder"

        def __init__(self) -> None:
            super().__init__(reduction="none")
            self.record_steps = set(int(s) for s in record_steps)
            self.topk = int(topk)
            self.frames: dict[int, dict[str, np.ndarray]] = {}

        def reset(self) -> None:
            super().reset()
            self.frames = {}

        def compute(self, **state: Any):
            step = int(state["step"])
            if step not in self.record_steps:
                return None
            # batch_size is 1 throughout this program, so index 0 is the env
            self.frames[step] = {
                "candidates": state["candidates"][0].detach().cpu().numpy().copy(),
                "costs": state["costs"][0].detach().cpu().numpy().copy(),
                "topk_inds": state["topk_inds"][0].detach().cpu().numpy().copy(),
            }
            return None

    return CandidateRecorder()


@dataclass
class StratumDraw:
    """Candidates drawn from one stratum of a single replan."""

    stratum: str
    actions: np.ndarray  # (n, horizon, action_block * action_dim)
    model_cost: np.ndarray  # (n,)
    source_step: int


def draw_strata(
    frames: dict[int, dict[str, np.ndarray]],
    *,
    n_per_stratum: int,
    cem_steps: int,
    rng: np.random.Generator,
) -> list[StratumDraw]:
    """Equal-n draws from the initial, middle, final and elite populations.

    Equal n is what makes the inversion rates comparable: an unequal draw would
    let sample size, rather than the cost, move the ratio the gate turns on.
    """
    first, mid, last = 0, cem_steps // 2, cem_steps - 1
    draws: list[StratumDraw] = []
    for stratum, step in (("random_init", first), ("gen_mid", mid), ("gen_last", last)):
        frame = frames[step]
        take = min(n_per_stratum, frame["candidates"].shape[0])
        index = rng.choice(frame["candidates"].shape[0], size=take, replace=False)
        draws.append(
            StratumDraw(
                stratum=stratum,
                actions=frame["candidates"][index],
                model_cost=frame["costs"][index],
                source_step=step,
            )
        )

    elites = frames[last]["topk_inds"]
    take = min(n_per_stratum, elites.shape[0])
    index = rng.choice(elites.shape[0], size=take, replace=False)
    chosen = elites[index]
    draws.append(
        StratumDraw(
            stratum="elite",
            actions=frames[last]["candidates"][chosen],
            model_cost=frames[last]["costs"][chosen],
            source_step=last,
        )
    )
    return draws


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def _decisive_pairs(true_cost: np.ndarray, delta: float):
    """Index pairs whose true costs differ by more than ``delta`` steps."""
    diff = true_cost[:, None] - true_cost[None, :]
    upper = np.triu(np.ones_like(diff, dtype=bool), k=1)
    return upper & (np.abs(diff) > delta)


def inversion_rate(model_cost: np.ndarray, true_cost: np.ndarray, delta: float):
    """Fraction of decisive pairs the cost orders backwards.

    Only the model's *order* is used, so this is directly comparable between a
    latent distance and a normalised remaining time.  Pairs whose true costs are
    within ``delta`` environment steps are dropped rather than scored: an
    inversion there is not a mistake worth counting.
    """
    model_cost = np.asarray(model_cost, dtype=np.float64)
    true_cost = np.asarray(true_cost, dtype=np.float64)
    decisive = _decisive_pairs(true_cost, delta)
    total = int(decisive.sum())
    if total == 0:
        return {"rate": None, "pairs": 0, "inversions": 0}
    true_diff = true_cost[:, None] - true_cost[None, :]
    model_diff = model_cost[:, None] - model_cost[None, :]
    # the cost prefers i over j (model_diff < 0) while the truth prefers j
    wrong = decisive & (np.sign(model_diff) != np.sign(true_diff)) & (model_diff != 0)
    inversions = int(wrong.sum())
    return {
        "rate": inversions / total,
        "pairs": total,
        "inversions": inversions,
    }


def kendall_tau(model_cost: np.ndarray, true_cost: np.ndarray):
    """Concordant-minus-discordant over untied pairs; no SciPy dependency."""
    model_cost = np.asarray(model_cost, dtype=np.float64)
    true_cost = np.asarray(true_cost, dtype=np.float64)
    model_diff = model_cost[:, None] - model_cost[None, :]
    true_diff = true_cost[:, None] - true_cost[None, :]
    upper = np.triu(np.ones_like(model_diff, dtype=bool), k=1)
    untied = upper & (model_diff != 0) & (true_diff != 0)
    total = int(untied.sum())
    if total == 0:
        return None
    concordant = int((untied & (np.sign(model_diff) == np.sign(true_diff))).sum())
    return (2.0 * concordant - total) / total


def topk_recall(model_cost: np.ndarray, true_cost: np.ndarray, k: int):
    """Share of the truly-best ``k`` that the cost also puts in its best ``k``."""
    k = int(min(k, len(model_cost)))
    if k <= 0:
        return None
    model_best = set(np.argsort(np.asarray(model_cost))[:k].tolist())
    true_best = set(np.argsort(np.asarray(true_cost))[:k].tolist())
    return len(model_best & true_best) / float(k)


def bootstrap_ci(values, *, reps: int = 10000, seed: int = 0, alpha: float = 0.05):
    """Percentile CI over whatever unit the caller resampled (replans here)."""
    values = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if values.size == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(reps, values.size), replace=True).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "lo": float(np.quantile(draws, alpha / 2)),
        "hi": float(np.quantile(draws, 1 - alpha / 2)),
        "n": int(values.size),
    }


@dataclass
class ReplanRecord:
    """Everything measured at one instrumented replan."""

    episode: int
    replan: int
    strata: dict[str, Any] = field(default_factory=dict)
    selected: dict[str, Any] | None = None
    delta_probe: dict[str, Any] | None = None
    match_before: dict[str, bool] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "episode": self.episode,
            "replan": self.replan,
            "strata": self.strata,
            "selected": self.selected,
            "delta_probe": self.delta_probe,
            "match_before": self.match_before,
        }


def summarise(records: list[ReplanRecord], deltas, *, seed: int = 0) -> dict[str, Any]:
    """Per-stratum inversion rates, and the amplification the gate turns on.

    Bootstrap is over replans, because that is the unit the strata are paired
    within: every stratum in a record saw the same snapshot and the same goal.
    """
    out: dict[str, Any] = {"deltas": list(deltas), "per_delta": {}}
    for delta in deltas:
        key = f"delta_{delta:g}"
        per_stratum: dict[str, Any] = {}
        # kept aligned by record index so the elite-vs-baseline contrast stays
        # paired: both strata in a record saw one snapshot and one goal
        by_stratum: dict[str, list] = {}
        pairs_seen: dict[str, list] = {}
        for stratum in POPULATION_STRATA:
            rates: list = []
            taus, recalls, npairs = [], [], []
            for record in records:
                entry = record.strata.get(stratum)
                if entry is None:
                    rates.append(None)
                    continue
                model = np.asarray(entry["model_cost"], dtype=np.float64)
                true = np.asarray(entry["true_cost"], dtype=np.float64)
                keep = ~np.asarray(entry["censored"], dtype=bool)
                if keep.sum() < 2:
                    rates.append(None)
                    continue
                model, true = model[keep], true[keep]
                measured = inversion_rate(model, true, delta)
                rates.append(measured["rate"])
                npairs.append(measured["pairs"])
                taus.append(kendall_tau(model, true))
                recalls.append(topk_recall(model, true, max(1, len(model) // 3)))
            by_stratum[stratum] = rates
            pairs_seen[stratum] = npairs
            per_stratum[stratum] = {
                "inversion_rate": bootstrap_ci(rates, seed=seed),
                "kendall_tau": bootstrap_ci(taus, seed=seed + 1),
                "topk_recall": bootstrap_ci(recalls, seed=seed + 2),
                "decisive_pairs_total": int(sum(npairs)),
                "decisive_pairs_min": int(min(npairs)) if npairs else 0,
            }

        # The gate reads the paired difference, not two independent intervals:
        # the strata are matched within a replan, so pairing is what removes the
        # between-replan variance that would otherwise swamp the effect.
        paired = [
            e - r
            for e, r in zip(by_stratum.get("elite", []), by_stratum.get("random_init", []))
            if e is not None and r is not None
        ]
        base = per_stratum.get("random_init", {}).get("inversion_rate", {}).get("mean")
        elite = per_stratum.get("elite", {}).get("inversion_rate", {}).get("mean")
        per_stratum["elite_amplification"] = {
            # undefined rather than infinite when the baseline orders perfectly;
            # the paired difference stays readable in that case
            "ratio": (elite / base) if (base not in (None, 0.0) and elite is not None) else None,
            "elite_rate": elite,
            "random_init_rate": base,
            "paired_difference": bootstrap_ci(paired, seed=seed + 4),
        }
        out["per_delta"][key] = per_stratum

    regrets = [
        r.selected["regret"]
        for r in records
        if r.selected is not None and r.selected.get("regret") is not None
    ]
    out["selected_regret_steps"] = bootstrap_ci(regrets, seed=seed + 3)

    censored, total = 0, 0
    for record in records:
        for entry in record.strata.values():
            flags = np.asarray(entry["censored"], dtype=bool)
            censored += int(flags.sum())
            total += int(flags.size)
    out["censoring"] = {
        "censored": censored,
        "total": total,
        "rate": (censored / total) if total else None,
    }

    # Does a candidate chunk do anything at all?  This needs no oracle, so it
    # separates "the candidates are equivalent" from "c_true cannot see the
    # difference" -- the ambiguity that made jobs 49557/49558 unreadable.
    out["component_change"] = {}
    for stratum in POPULATION_STRATA:
        changed, net, gained, lost, seen = 0, 0, 0, 0, 0
        for record in records:
            entry = record.strata.get(stratum)
            if entry is None or "component" not in entry:
                continue
            for item in entry["component"]:
                seen += 1
                changed += int(bool(item["changed"]))
                net += int(item["net"])
                gained += len(item["gained"])
                lost += len(item["lost"])
        out["component_change"][stratum] = {
            "candidates": seen,
            "changed_any_rate": (changed / seen) if seen else None,
            "components_gained": gained,
            "components_lost": lost,
            "mean_net": (net / seen) if seen else None,
        }
    return out


__all__ = [
    "DRAWER_OPEN_POS",
    "POPULATION_STRATA",
    "PROTOCOL",
    "STRATA",
    "OracleRepair",
    "ReplanRecord",
    "StratumDraw",
    "bootstrap_ci",
    "draw_strata",
    "inversion_rate",
    "kendall_tau",
    "make_recorder",
    "summarise",
    "topk_recall",
]
