# action_curvature_h0

Diagnostic program: **Action-Space Curvature Mismatch** in latent world models.

Stage 1 is a no-training measurement comparing the model's multi-step
action-to-outcome map `Phi_H(a) = F^H(E(o_t), a)` against the realized map
`Psi_H(a) = E(o_H^sim(s_t, a))` obtained by replaying the same action triplet
through the simulator from an exact state reset.  Stage 2 is a one-loss
regularizer used to test causality on whatever Stage 1 finds.

Stage 1 runs on **OGBench-Cube only** (frozen `quentinll/lewm-cube`); Push-T is
deferred until an exact full-state reset harness for it is verified.

`PROTOCOL.md` is locked and must not be edited to follow observed results;
amendments are appended with their justification and the state of evidence at
the time, per the convention used by the other pilots in this repo.

Reused artifacts:
- full-state reset / true endpoint: `diagnosis/scripts/76_ogb_true_endpoint_corrected.py`
- cached CEM populations: `physical_search_distillation/outputs/h0/populations`,
  `counterfactual_flow/outputs/ogbench_cube_phase0/locked_shards`
- contact stratification and boundary-blindness: `diagnosis/stratification/`,
  `diagnosis/metrics/boundary_blindness.py`
- candidate rank agreement: the CEM preselection audit (`diagnosis/scripts/53_*`)

Positioning note: the nearest prior art is **not** the 2026 LeWM line but
PCC (Prediction, Consistency, Curvature, ICLR 2020, arXiv:1909.01506), which
regularizes one-step latent transition curvature for linearization-based
control.  The object here is the **composed H-step action-sequence to terminal
latent map** queried by sampling-based CEM, and the reference is a measured
realized map rather than an assumed-good low-curvature prior.

The full statement of the idea -- motivation, theory, measurement design,
intervention, failure modes, and prior-art positioning -- is `METHOD.md`.
