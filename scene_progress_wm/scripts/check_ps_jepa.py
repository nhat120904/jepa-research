#!/usr/bin/env python3
"""Shape and invariance checks for the predictive-state JEPA, before any training.

Like ``check_progress_objective.py`` this needs no checkpoint, so it can gate the
pipeline before a single step has run.  Each check is an assertion about a claim
the module makes in prose; if the prose and the code disagree, this fails.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scene_progress_wm.ps_jepa import (  # noqa: E402
    ARMS,
    PSJEPA,
    PSJEPAConfig,
    SceneSegmentDataset,
    collapse_metrics,
    ps_jepa_losses,
)


class ZeroReg(torch.nn.Module):
    """Stands in for SIGReg so the prediction term can be read on its own."""

    def forward(self, projections: torch.Tensor) -> torch.Tensor:
        return projections.sum() * 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("checks must run inside a Slurm compute job")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    report: dict = {"device": str(device), "checks": {}}

    # -- 1. every arm runs, and they are parameter-matched -----------------
    counts = {}
    for arm, flags in ARMS.items():
        config = PSJEPAConfig(**flags)
        model = PSJEPA(config).to(device)
        counts[arm] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        latents = torch.randn(4, config.segment_blocks, config.latent_dim, device=device)
        blocks = torch.randn(4, config.segment_blocks, config.action_input_dim, device=device)
        out = ps_jepa_losses(model, ZeroReg(), latents, blocks, config)
        out["loss"].backward()
        flowing = sum(
            1
            for p in model.parameters()
            if p.requires_grad and p.grad is not None and float(p.grad.abs().sum()) > 0
        )
        trainable = sum(1 for p in model.parameters() if p.requires_grad)
        assert out["per_horizon"].numel() == config.predict_blocks
        assert flowing > 0, f"{arm}: no gradient reached any parameter"
        report["checks"][f"{arm}_pred_loss"] = float(out["pred_loss"])
        report["checks"][f"{arm}_grads"] = f"{flowing}/{trainable}"
    assert len(set(counts.values())) == 1, f"arms are not parameter-matched: {counts}"
    report["checks"]["parameters"] = counts

    # -- 2. the target must never carry gradient --------------------------
    config = PSJEPAConfig()
    model = PSJEPA(config).to(device)
    assert all(not p.requires_grad for p in model.target.parameters())

    # -- 3. candidate actions must move the prediction, except in the control
    horizon = config.predict_blocks
    deltas = {}
    for arm in ("ps_jepa", "ps_noaction_pred"):
        cfg = PSJEPAConfig(**ARMS[arm])
        m = PSJEPA(cfg).to(device).eval()
        belief = torch.randn(8, cfg.belief_dim, device=device)
        first = torch.randn(8, horizon, cfg.action_input_dim, device=device)
        second = torch.randn(8, horizon, cfg.action_input_dim, device=device)
        with torch.no_grad():
            gap = (m.predictor.rollout(belief, first) - m.predictor.rollout(belief, second)).abs().mean()
        deltas[arm] = float(gap)
    assert deltas["ps_jepa"] > 1e-4, "the predictor ignores the candidate chunk"
    assert deltas["ps_noaction_pred"] == 0.0, "the control still reads the chunk"
    report["checks"]["candidate_action_delta"] = deltas

    # -- 4. a different past must move the belief, except in the frame arm --
    history = {}
    for arm in ("ps_jepa", "ps_frame"):
        cfg = PSJEPAConfig(**ARMS[arm])
        m = PSJEPA(cfg).to(device).eval()
        latents = torch.randn(8, 12, cfg.latent_dim, device=device)
        blocks = torch.randn(8, 12, cfg.action_input_dim, device=device)
        other = latents.clone()
        other[:, :6] = torch.randn(8, 6, cfg.latent_dim, device=device)  # same present
        with torch.no_grad():
            a, _ = m.encoder.scan(latents, blocks)
            b, _ = m.encoder.scan(other, blocks)
        history[arm] = float((a[:, -1] - b[:, -1]).abs().mean())
    assert history["ps_jepa"] > 1e-4, "the filter ignores its own past"
    assert history["ps_frame"] == 0.0, "the frame control still carries history"
    report["checks"]["history_delta"] = history

    # -- 5. the action *prefix* is ignored by default ----------------------
    prefix = {}
    for arm in ("ps_jepa", "ps_action_history"):
        cfg = PSJEPAConfig(**ARMS[arm])
        m = PSJEPA(cfg).to(device).eval()
        latents = torch.randn(8, 12, cfg.latent_dim, device=device)
        one = torch.randn(8, 12, cfg.action_input_dim, device=device)
        two = torch.randn(8, 12, cfg.action_input_dim, device=device)
        with torch.no_grad():
            a, _ = m.encoder.scan(latents, one)
            b, _ = m.encoder.scan(latents, two)
        prefix[arm] = float((a[:, -1] - b[:, -1]).abs().mean())
    assert prefix["ps_jepa"] == 0.0, "the default filter is reading the action prefix"
    assert prefix["ps_action_history"] > 1e-4, "the ablation is not reading it either"
    report["checks"]["action_prefix_delta"] = prefix

    # -- 6. the collapse metric must separate a collapsed belief -----------
    varied = collapse_metrics(torch.randn(8, 20, 64, device=device), torch.tensor(0.01))
    flat = collapse_metrics(torch.ones(8, 20, 64, device=device) * 3.0, torch.tensor(0.01))
    assert varied["effective_rank"] > 10.0 > flat["effective_rank"]
    assert flat["normalised_error"] > varied["normalised_error"]
    report["checks"]["collapse_varied"] = varied
    report["checks"]["collapse_flat"] = flat

    # -- 7. segments never cross an episode boundary -----------------------
    if args.cache_dir is not None:
        data = SceneSegmentDataset(args.cache_dir, config)
        stride, episode_len = data.stride, data.episode_len
        starts = data.starts
        first_row = (starts - 1) * stride
        last_row = (starts + config.segment_blocks - 1) * stride
        assert (first_row >= 0).all()
        assert ((first_row // episode_len) == (last_row // episode_len)).all()
        sample = data[0]
        assert sample["latents"].shape == (config.segment_blocks, config.latent_dim)
        assert sample["blocks"].shape == (config.segment_blocks, config.action_input_dim)
        report["checks"]["segments"] = {
            "count": int(starts.size),
            "stride": stride,
            "episode_len": episode_len,
        }

    print(json.dumps(report, indent=2, sort_keys=True))
    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "ps_jepa_checks.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print("ALL CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
