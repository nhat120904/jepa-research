# Skill handoff world-model pilot

This directory implements the falsification-first Direction A proposed in
[`research/REPORT_VI.md`](../research/REPORT_VI.md): state-based OGBench Ant skills first, then a controlled test of
whether the handoff state and execution time leave useful planning headroom.

The code covers A0 and the privileged A1/A2 screen. The completed pilot stopped at A2, so no joint handoff model or
AntMaze-large transfer was started. See [`docs/RESULTS.md`](docs/RESULTS.md) for the evidence and decision,
[`docs/PROTOCOL.md`](docs/PROTOCOL.md) for frozen choices and stop rules, and [`docs/AUDIT.md`](docs/AUDIT.md) for
baseline/code readiness.

All MuJoCo, model training/loading, and result aggregation must run under Slurm. The smallest end-to-end smoke is:

```bash
sbatch skill_handoff_wm/scripts/slurm_a0_smoke.sh
```

For reproduction, if A0 passes, run the paired switching smoke with the produced checkpoint:

```bash
sbatch skill_handoff_wm/scripts/slurm_a1_smoke.sh \
  /home/nhatnc129/nhat.nc/jepa-research/skill_handoff_wm/outputs/a0_smoke/JOB_ID/train/best.pt
```

These smoke runs are plumbing checks, not publishable estimates. The required CompPlan reconstruction and HIQL/HSVL
system comparisons remain explicit blockers before any large campaign.
