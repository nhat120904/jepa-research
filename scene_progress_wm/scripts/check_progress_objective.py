"""Shape and invariance checks for the progress objective, on synthetic tensors.

The objective's indexing into ``predicted_emb`` is the one piece of Stage 3 that the
training smoke never exercises, and a silent off-by-one there would score the wrong
frames while still producing a plausible number. These checks run without a checkpoint,
so they can gate the pipeline before anything is trained.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scene_progress_wm.progress_head import (  # noqa: E402
    ARMS,
    ProgressConfig,
    ProgressHead,
    ProgressTracker,
    expectile_loss,
)
from scene_progress_wm.progress_objective import ProgressCost, build_mixture  # noqa: E402
from scene_progress_wm.scene_lewm import LeWMConfig  # noqa: E402


def main() -> None:
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("checks must run inside a Slurm compute job")

    lewm = LeWMConfig()
    device = torch.device("cpu")
    batch, samples, horizon = 1, 7, lewm.history_size + 2
    history, dim = lewm.history_size, lewm.embed_dim
    results = {}

    for arm, overrides in sorted(ARMS.items()):
        config = ProgressConfig(
            latent_dim=dim, action_input_dim=lewm.action_input_dim, **overrides
        )
        head = ProgressHead(config).eval()
        tracker = ProgressTracker(head, device)
        tracker.reset(torch.randn(dim))
        before = tracker.fingerprint()

        cost = ProgressCost(head, tracker, history_size=history)
        info = {
            "predicted_emb": torch.randn(batch, samples, history + horizon, dim),
            "action_candidates": torch.randn(
                batch, samples, horizon, lewm.action_input_dim
            ),
            "goal_emb": torch.randn(batch, 1, dim),
        }
        with torch.no_grad():
            value = cost(info)

        if value.shape != (batch, samples):
            raise RuntimeError(f"{arm}: cost shape {tuple(value.shape)} != {(batch, samples)}")
        if not torch.isfinite(value).all():
            raise RuntimeError(f"{arm}: non-finite cost")
        if tracker.fingerprint() != before:
            raise RuntimeError(f"{arm}: scoring advanced the executed-history state")

        # candidates must actually change the score, or the objective is inert
        alt = dict(info)
        alt["action_candidates"] = info["action_candidates"] + 3.0
        with torch.no_grad():
            other = cost(alt)
        action_sensitive = bool((other - value).abs().max() > 1e-6)
        if config.use_action and not action_sensitive:
            raise RuntimeError(f"{arm}: cost ignores the action candidates")
        if not config.use_action and action_sensitive:
            raise RuntimeError(f"{arm}: action-ablated head still reads actions")

        alt = dict(info)
        alt["predicted_emb"] = info["predicted_emb"] + 3.0
        with torch.no_grad():
            other = cost(alt)
        obs_sensitive = bool((other - value).abs().max() > 1e-6)
        if config.use_obs and not obs_sensitive:
            raise RuntimeError(f"{arm}: cost ignores the predicted latents")
        if not config.use_obs and obs_sensitive:
            raise RuntimeError(f"{arm}: observation-ablated head still reads latents")

        # Latching is only meaningful where there is memory to latch into. Severing the
        # recurrence severs the accumulator with it, so `prog_frame` is not monotone and
        # is not supposed to be -- it ablates history and latching jointly, which is why
        # `prog_nomono` exists to ablate latching alone.
        latents = torch.randn(4, 6, dim)
        actions = torch.randn(4, 6, lewm.action_input_dim)
        with torch.no_grad():
            _h, progress, _s = head.scan(latents, actions)
        deltas = (progress[:, 1:] - progress[:, :-1]).min().item()
        expect_monotone = config.monotone and config.use_history
        if expect_monotone and deltas < -1e-6:
            raise RuntimeError(f"{arm}: progress fell by {deltas}")
        if not config.monotone and deltas >= 0.0:
            raise RuntimeError(
                f"{arm}: the non-monotone control never fell, so it is not a control"
            )

        results[arm] = {
            "cost_shape": list(value.shape),
            "action_sensitive": action_sensitive,
            "obs_sensitive": obs_sensitive,
            "min_progress_delta": deltas,
            "expect_monotone": expect_monotone,
            "tracker_stable": True,
            "parameters": int(sum(p.numel() for p in head.parameters())),
        }

    # the mixture at weight 1.0 must be the goal term itself, not a scaled sum
    from stable_worldmodel.planning import GoalMSE

    goal_term = GoalMSE()
    config = ProgressConfig(latent_dim=dim, action_input_dim=lewm.action_input_dim)
    head = ProgressHead(config).eval()
    tracker = ProgressTracker(head, device)
    tracker.reset(torch.randn(dim))
    progress_term = ProgressCost(head, tracker, history_size=history)
    if build_mixture(goal_term, progress_term, 1.0) is not goal_term:
        raise RuntimeError("weight 1.0 must return the goal term unchanged")
    if build_mixture(goal_term, progress_term, 0.0) is not progress_term:
        raise RuntimeError("weight 0.0 must return the progress term unchanged")

    # expectile must lean the way it claims: tau < 0.5 penalises over-prediction more
    pred = torch.zeros(1024)
    low = expectile_loss(pred, torch.full((1024,), -1.0), tau=0.2)
    high = expectile_loss(pred, torch.full((1024,), 1.0), tau=0.2)
    if not high < low:
        raise RuntimeError("expectile tau=0.2 does not lean toward shorter targets")

    print(json.dumps({"verdict": "CHECKS_PASS", "arms": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
