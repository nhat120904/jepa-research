# Stage C implementation scaffold

> Review update, 13 September 2026: the numerical smoke pass does not qualify the scientific comparison. [FEASIBILITY_REASSESSMENT_20260913_EN.md](FEASIBILITY_REASSESSMENT_20260913_EN.md) identifies short-window memory, missing temporal/segment identity, restricted frame readout, additional-supervision confounding and unenforced one-window-per-candidate evaluation. These findings are not fixed in code by this documentation update.

Status: code prepared; no Stage-C encoding or training job launched. Stage C remains
blocked until the full Stage-B1 headroom/readout gate passes.

## Locked comparison

All arms consume the same three-camera frozen DINOv2 features, causal proprioceptive
history, eight native actions, task id and eventual-success supervision:

1. endpoint-only prediction;
2. full frame-latent rollout;
3. simple progress-sequence prediction;
4. learned unstructured segment target;
5. the same segment model with observed/predicted composition constraints;
6. direct history-plus-action value prediction.

The unstructured and compositional segment arms instantiate the same target encoder,
predictor and composer. Setting the composition/split losses to zero produces the matched
ablation. The target encoder sees future visual features but never candidate actions. An
EMA target copy and visual-feature reconstruction anchor protect against collapse.

The headline metric is fixed-candidate selected native success. Prediction MSE, value AUC
and target losses are diagnostics. The compositional arm must beat the strongest control
by at least five percentage points and must beat the unstructured segment arm; otherwise
the Stage-C screen stops.

## Data contract and leakage guards

`stage_c/data.py` reads episode-level feature tensors through a manifest. Training entries
must have `allow_training=true`, may not have source
`stage_b_validation_branches`, and the continuation-value training set must contain both
success and failure windows. Candidate evaluation entries are a separate
`candidate_eval` split with fixed group/candidate ids.

The currently downloaded RoboCasa sample is not enough to launch Stage C: metadata
declares 504 Scrub episodes and 509 Rinse episodes, but only five parquet episodes per task
and no videos are present. Full three-camera videos must be acquired and encoded. In addition, successful
human demonstrations alone are insufficient for the value readout; a separate
current-runtime training-rollout split with failures and native progress labels is still
required. Stage-B validation branches are never used for this purpose.

## Prepared entry points

- `scripts/encode_stage_c_offline.py`: pinned frozen DINOv2 episode encoder;
- `stage_c/data.py`: causal window loader and leakage/class-balance checks;
- `stage_c/models.py`: all six matched arms;
- `scripts/run_stage_c_train.py`: guarded one-arm trainer and fixed-candidate evaluation;
- `scripts/aggregate_stage_c.py`: locked strongest-control decision;
- `scripts/audit_stage_c_readiness.py`: compute-node data/gate audit;
- `scripts/smoke_stage_c_models.py`: forward/backward smoke test;
- `scripts/slurm_stage_c_*.sh`: batch entry points, prepared but not launched except a
  bounded CPU smoke test when recorded in the job ledger.
