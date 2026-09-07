"""Closed-loop CEM planning on OGBench-Scene with a learned latent world model.

The loop is deliberately bespoke rather than routed through ``World.evaluate``: the
arena resets from a dataset row and takes its goal from another dataset row, which the
Hydra/lance evaluation path does not express for Scene. Everything that *does* exist
upstream is reused unchanged -- ``CEMSolver``, ``ShootingCostEvaluator``, ``PlanConfig``
and the objective protocol -- so an arm differs from the baseline only in its objective.

Success is ``SceneEnv._compute_successes``: OGBench's own predicate over cube, both
buttons, drawer and window, with OGBench's own tolerances, evaluated against targets set
from the goal row.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

from scene_progress_wm.scene_data import EPISODE_LEN, GoalMatch, goal_match
from scene_progress_wm.scene_lewm import LeWMConfig, image_stats
from scene_progress_wm.scene_render import (
    env_successes,
    restore_state,
    set_goal_targets,
)

PROTOCOL = "scene_progress_wm_eval_v1"
STATE_COLUMNS = ("cube_x", "cube_y", "cube_z", "drawer", "window", "drawer_site_y")


@dataclass(frozen=True)
class EpisodeSpec:
    episode: int
    start_row: int
    goal_row: int

    def to_json(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class PlanSettings:
    horizon: int = 5
    receding_horizon: int = 5
    num_samples: int = 300
    cem_steps: int = 30
    topk: int = 30
    var_scale: float = 1.0

    def to_json(self) -> dict:
        return asdict(self)


def build_episode_specs(
    num_rows: int,
    goal_offset: int,
    num_episodes: int,
    seed: int,
    track: np.ndarray | None = None,
    buttons: np.ndarray | None = None,
    max_draws: int = 200,
) -> list[EpisodeSpec]:
    """One (start, goal) pair per episode, deterministic in ``seed``.

    The start is drawn uniformly from the rows that leave room for the goal offset, so
    the same seed and offset give the same episodes for every arm.

    When ``track`` and ``buttons`` are given, a start whose state already satisfies the
    goal is rejected and redrawn. The replay gate measured how much this matters: at
    goal offset 25, 17 of 50 sampled episodes were already solved before a single
    action ran. Those hand every arm the same third of its score for free and compress
    exactly the range the offset ladder exists to resolve. The rejection reads only the
    dataset, never the arm, so all arms still see identical episodes.
    """
    total_episodes = num_rows // EPISODE_LEN
    if num_episodes > total_episodes:
        raise ValueError(f"asked for {num_episodes} episodes, split has {total_episodes}")
    if goal_offset >= EPISODE_LEN:
        raise ValueError("goal offset must be shorter than an episode")

    screening = track is not None and buttons is not None
    rng = np.random.default_rng(seed)
    chosen = rng.choice(total_episodes, size=num_episodes, replace=False)
    chosen.sort()
    specs = []
    for episode in chosen:
        for _ in range(max_draws):
            offset = int(rng.integers(0, EPISODE_LEN - goal_offset))
            start = int(episode) * EPISODE_LEN + offset
            if not screening:
                break
            trivial = goal_match(
                state_row(track, start),
                buttons[start],
                state_row(track, start + goal_offset),
                buttons[start + goal_offset],
            ).success
            if not trivial:
                break
        else:
            raise RuntimeError(
                f"episode {int(episode)} has no non-trivial start at offset {goal_offset}"
            )
        specs.append(
            EpisodeSpec(episode=int(episode), start_row=start, goal_row=start + goal_offset)
        )
    return specs


def state_row(track: np.ndarray, row: int) -> dict[str, np.ndarray | float]:
    values = np.asarray(track[row], dtype=np.float64)
    return {
        "cube_pos": values[0:3],
        "drawer": float(values[3]),
        "window": float(values[4]),
        "drawer_site_y": float(values[5]),
    }


class FrameEncoder:
    """Turn a rendered uint8 frame into the model's normalised input."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        mean, std = image_stats(device)
        self.mean = mean.view(1, 3, 1, 1)
        self.std = std.view(1, 3, 1, 1)

    def __call__(self, frame: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(np.ascontiguousarray(frame)).to(self.device)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        return ((tensor - self.mean) / self.std)[0]


class HistoryWindow:
    """Context frames and the action blocks between them.

    Before anything has been executed the window is padded by repeating the first frame
    and pairing it with zero action blocks, which is the convention the upstream
    ``HistoryBuffer`` uses.
    """

    def __init__(self, history_len: int, action_input_dim: int, device: torch.device) -> None:
        self.history_len = history_len
        self.action_input_dim = action_input_dim
        self.device = device
        self.frames: deque[torch.Tensor] = deque(maxlen=history_len)
        self.blocks: deque[torch.Tensor] = deque(maxlen=max(history_len - 1, 1))

    def reset(self, frame: torch.Tensor) -> None:
        self.frames.clear()
        self.blocks.clear()
        for _ in range(self.history_len):
            self.frames.append(frame)
        for _ in range(self.history_len - 1):
            self.blocks.append(
                torch.zeros(self.action_input_dim, device=self.device, dtype=frame.dtype)
            )

    def push(self, frame: torch.Tensor, block: torch.Tensor) -> None:
        self.frames.append(frame)
        if self.history_len > 1:
            self.blocks.append(block.to(frame.dtype))

    def pixels(self) -> torch.Tensor:
        return torch.stack(list(self.frames), dim=0).unsqueeze(0)

    def action_history(self) -> torch.Tensor:
        if self.history_len <= 1:
            return torch.zeros(
                1, 0, self.action_input_dim, device=self.device, dtype=torch.float32
            )
        return torch.stack(list(self.blocks), dim=0).unsqueeze(0)


def make_solver(model, objective, settings: PlanSettings, config: LeWMConfig, device, seed: int):
    """Compose the upstream evaluator and solver; nothing here is subclassed."""
    import gymnasium as gym
    from stable_worldmodel.planning import CEMSolver, ShootingCostEvaluator
    from stable_worldmodel.policy import PlanConfig

    evaluator = ShootingCostEvaluator(model, objective)
    plan_config = PlanConfig(
        horizon=settings.horizon,
        receding_horizon=settings.receding_horizon,
        action_block=config.action_block,
        history_len=config.history_size,
    )
    solver = CEMSolver(
        cost=evaluator,
        batch_size=1,
        num_samples=settings.num_samples,
        var_scale=settings.var_scale,
        n_steps=settings.cem_steps,
        topk=settings.topk,
        device=device,
        seed=seed,
    )
    solver.configure(
        action_space=gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1, config.action_dim), dtype=np.float32
        ),
        n_envs=1,
        config=plan_config,
    )
    return solver, evaluator, plan_config


def run_episode(
    *,
    raw,
    spec: EpisodeSpec,
    cache: dict[str, np.ndarray],
    track: np.ndarray,
    model,
    objective,
    settings: PlanSettings,
    config: LeWMConfig,
    device: torch.device,
    encoder: FrameEncoder,
    camera: str,
    eval_budget: int,
    plan_seed: int,
    on_replan=None,
    on_start=None,
    on_step=None,
) -> dict:
    """Plan and execute one episode; returns the record for the results file."""
    qpos, qvel, buttons = cache["qpos"], cache["qvel"], cache["button_states"]

    goal_state = state_row(track, spec.goal_row)
    goal_buttons = buttons[spec.goal_row]

    # the goal image is rendered from the goal row, then the episode starts from scratch
    restore_state(raw, qpos[spec.goal_row], qvel[spec.goal_row], goal_buttons)
    goal_frame = encoder(np.asarray(raw.render(camera=camera), dtype=np.uint8))

    restore_state(raw, qpos[spec.start_row], qvel[spec.start_row], buttons[spec.start_row])
    set_goal_targets(raw, goal_state, goal_buttons)

    start_match: GoalMatch = goal_match(
        state_row(track, spec.start_row), buttons[spec.start_row], goal_state, goal_buttons
    )
    if env_successes(raw)["success"]:
        return {
            "spec": spec.to_json(),
            "success": True,
            "trivial": True,
            "env_steps": 0,
            "replans": 0,
            "start_match": start_match.to_json(),
            "final_match": env_successes(raw),
        }

    solver, _evaluator, _plan_config = make_solver(
        model, objective, settings, config, device, plan_seed
    )

    window = HistoryWindow(config.history_size, config.action_input_dim, device)
    first_frame = encoder(np.asarray(raw.render(camera=camera), dtype=np.uint8))
    window.reset(first_frame)
    if on_start is not None:
        on_start(first_frame)

    goal_tensor = goal_frame.unsqueeze(0).unsqueeze(0)  # (1, 1, C, H, W)
    steps = 0
    replans = 0
    success = False
    final_costs: list[float] = []

    while steps < eval_budget and not success:
        info = {
            "pixels": window.pixels(),
            "goal": goal_tensor,
            "action_history": window.action_history(),
            "action": torch.zeros(
                1, config.history_size, config.action_input_dim, device=device
            ),
        }
        if on_replan is not None:
            on_replan(info, window)

        outputs = solver.solve(info)
        plan = outputs["actions"][0]  # (horizon, action_block * action_dim)
        replans += 1
        # The mixture weight is only meaningful once the two terms are on a common
        # scale, and the scale has to be measured rather than guessed. Recording the
        # elite cost of every replan on the baseline arm is what fixes the divisor,
        # after which it is frozen and never refit per arm.
        if outputs.get("costs"):
            final_costs.append(float(np.asarray(outputs["costs"][-1]).min()))

        executed_blocks = min(settings.receding_horizon, settings.horizon)
        for block_index in range(executed_blocks):
            block = plan[block_index]
            actions = (
                block.reshape(config.action_block, config.action_dim).cpu().numpy()
            )
            actions = np.clip(actions, -1.0, 1.0).astype(np.float32)
            for action in actions:
                raw.step(action)
                steps += 1
                if env_successes(raw)["success"]:
                    success = True
                    break
            frame = encoder(np.asarray(raw.render(camera=camera), dtype=np.uint8))
            window.push(frame, block.to(device))
            if on_step is not None:
                on_step(frame, block.to(device))
            if success or steps >= eval_budget:
                break

    final = env_successes(raw)
    return {
        "spec": spec.to_json(),
        "success": bool(final["success"]),
        "trivial": False,
        "env_steps": int(steps),
        "replans": int(replans),
        "start_match": start_match.to_json(),
        "final_match": final,
        "elite_cost_mean": float(np.mean(final_costs)) if final_costs else None,
        "elite_cost_median": float(np.median(final_costs)) if final_costs else None,
    }


def load_cache(cache_dir: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    cache_dir = Path(cache_dir)
    cache = {
        name: np.load(cache_dir / f"{name}.npy", mmap_mode="r")
        for name in ("qpos", "qvel", "button_states", "actions")
    }
    track = np.load(cache_dir / "state.npy", mmap_mode="r")
    return cache, track


__all__ = [
    "PROTOCOL",
    "EpisodeSpec",
    "FrameEncoder",
    "HistoryWindow",
    "PlanSettings",
    "build_episode_specs",
    "load_cache",
    "make_solver",
    "run_episode",
    "state_row",
]
