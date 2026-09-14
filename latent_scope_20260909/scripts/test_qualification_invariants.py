#!/usr/bin/env python3
"""CPU sbatch regression tests. Synthetic fixtures are plumbing tests, not evidence."""
import argparse
import copy
import json
import os
import tempfile
from pathlib import Path

import torch

from stage_c.data import SegmentWindowDataset, StageCDataError
from stage_c.metrics import candidate_metrics
from stage_c.models import ModelDimensions, build_arm


def must_fail(fn, exception=ValueError):
    try:
        fn()
    except exception:
        return
    raise AssertionError("Malformed input was accepted")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Tests must run in a CPU sbatch job")
    torch.set_num_threads(2)
    torch.manual_seed(17)
    checks = []
    with tempfile.TemporaryDirectory(prefix="lscope-invariants-") as directory:
        root = Path(directory)
        episode = {"visual_features": torch.randn(16, 3, 16),
                   "proprio": torch.randn(16, 4), "actions": torch.randn(16, 3),
                   "eventual_success": torch.ones(16)}
        entries = []
        for i in range(2):
            torch.save(episode, root / f"{i}.pt")
            entries.append(dict(episode_uid=f"branch-{i}", source_episode_uid="source-0",
                                prefix_uid="prefix-0", split="candidate_eval", task_id=0,
                                episode_id=i, num_frames=16, path=f"{i}.pt",
                                candidate_group_id=0, candidate_index=i, candidate_count=2,
                                decision_step=2))
        manifest_path = root / "manifest.json"
        def dataset(rows):
            manifest_path.write_text(json.dumps({"schema_version": 1, "episodes": rows}))
            return SegmentWindowDataset(manifest_path, "candidate_eval", 3, 8)
        data = dataset(entries)
        assert data.windows == [(0, 2), (1, 2)]
        assert torch.equal(data[0]["visual_history"], data[1]["visual_history"])
        checks.append("one_causal_window_per_candidate")
        must_fail(lambda: dataset(entries[:1]), StageCDataError)
        bad = copy.deepcopy(entries)
        bad[1]["candidate_index"] = 0
        must_fail(lambda: dataset(bad), StageCDataError)
        bad = copy.deepcopy(entries)
        bad[1]["decision_step"] = 3
        must_fail(lambda: dataset(bad), StageCDataError)
        bad = copy.deepcopy(entries)
        bad[1]["split"] = "train"
        must_fail(lambda: dataset(bad), StageCDataError)
        episode["visual_features"][1] += 1
        torch.save(episode, root / "1.pt")
        must_fail(lambda: dataset(entries), StageCDataError)
        checks.append("reject_missing_duplicate_future_context_and_cross_split_sources")
    rows = [dict(task=0, group=0, candidate=i, candidate_count=2, label=i, score=0.) for i in range(2)]
    assert candidate_metrics(rows)["selected_success_rate"] == 0
    must_fail(lambda: candidate_metrics(rows + rows[:1]))
    must_fail(lambda: candidate_metrics(rows[:1]))
    must_fail(lambda: candidate_metrics([dict(rows[0], score=float("nan")), rows[1]]))
    checks.append("strict_candidate_metrics_and_locked_ties")
    dims = ModelDimensions(cameras=3, camera_feature_dim=16, proprio_dim=4, action_dim=3,
                           latent_dim=32, context_dim=16, summary_width=16, summary_tokens=2,
                           predictor_width=32, task_embed_dim=8, max_segment_steps=8)
    batch = dict(visual_history=torch.randn(2, 3, 3, 16), proprio_history=torch.randn(2, 3, 4),
                 actions=torch.randn(2, 8, 3), visual_future=torch.randn(2, 8, 3, 16),
                 proprio_future=torch.randn(2, 8, 4), eventual_success=torch.tensor([0., 1.]),
                 task_id=torch.tensor([0, 1]), progress_future=torch.randn(2, 8, 7),
                 has_progress=torch.ones(2, dtype=torch.bool))
    weights = dict(endpoint=1., frame=1., segment=1., reconstruction=.25,
                   composition=.5, split_consistency=.5, progress=1., value=1.)
    for arm in ("endpoint_only", "frame_rollout", "simple_progress", "direct_value",
                "unstructured_segment", "compositional_segment"):
        model = build_arm(arm, dims).eval()
        loss, terms, logits = model.loss(batch, weights)
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(logits).all()
        changed = dict(batch, visual_future=batch["visual_future"] + 100,
                       proprio_future=batch["proprio_future"] - 100)
        with torch.no_grad():
            # Compare the same inference kernel path (grad/no-grad may choose different kernels).
            assert torch.equal(model.loss(batch, weights)[2], model.loss(changed, weights)[2]), arm
        model.update_ema()
    checks.append("all_arms_finite_backward_and_no_future_input_in_scoring")
    direct = build_arm("direct_value", dims).eval()
    reversed_batch = dict(batch, actions=batch["actions"].flip(1))
    assert not torch.allclose(direct.loss(batch, weights)[2], direct.loss(reversed_batch, weights)[2], atol=1e-7, rtol=0)
    segment = build_arm("compositional_segment", dims).eval()
    segment.train()
    assert not segment.target_encoder.training and segment.online_target.training
    segment.eval()
    left, right = torch.randn(2, 2, 16), torch.randn(2, 2, 16)
    assert not torch.allclose(segment.composer(left, right), segment.composer(right, left), atol=1e-7, rtol=0)
    checks.append("action_and_composer_order_sensitivity")
    frame = build_arm("frame_rollout", dims).eval()
    sequence = torch.randn(2, 8, dims.latent_dim)
    assert not torch.allclose(frame.sequence_readout(sequence)[1], frame.sequence_readout(sequence.flip(1))[1])
    checks.append("ordered_full_frame_readout")
    other = build_arm("unstructured_segment", dims).eval()
    other.load_state_dict(segment.state_dict())
    total, terms, _ = segment.loss(batch, weights)
    control, control_terms, _ = other.loss(batch, weights)
    assert terms.keys() == control_terms.keys()
    intended = weights["composition"] * (terms["observed_composition"] + terms["predicted_composition"]) + weights["split_consistency"] * terms["partition_endpoint"]
    assert torch.allclose(total - control, intended, atol=1e-5)
    segment.zero_grad()
    terms["observed_composition"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in segment.online_target.parameters())
    checks.append("matched_subsegment_supervision_and_target_composition_gradient")
    for split in (1, 3, 5, 7):
        assert torch.isfinite(segment.predict_partition(batch, split)[2]).all()
    checks.append("heldout_partition_inference")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"verdict": "QUALIFICATION_INVARIANTS_PASS", "checks": checks,
                                       "job_id": os.environ["SLURM_JOB_ID"]}, indent=2) + "\n")
    print("QUALIFICATION_INVARIANTS_PASS", flush=True)


if __name__ == "__main__":
    main()
