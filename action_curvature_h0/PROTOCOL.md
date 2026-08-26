# Action-Space Curvature Mismatch — locked diagnostic protocol

Locked before any measurement is executed: 2026-08-24 UTC.

Stage 1 of this program is a **no-training diagnostic** and is written to
publication standard, not as a gate for a method.  Stage 2 (the regularizer)
is an intervention used to test causality on whatever Stage 1 finds.  A null
in Stage 2 does not invalidate Stage 1.

## Question

> Do latent world models distort the local action geometry of the environment,
> particularly in regions visited by planning, and does correcting that
> distortion improve control?

## Stage 1 scope (declared before any measurement, 2026-08-24)

- Environment: **OGBench-Cube only**.  Push-T is deferred until an exact
  full-state reset harness for it is verified; Stage 1 conclusions are not
  generalized beyond Cube.
- Checkpoint: released `quentinll/lewm-cube` (the same checkpoint used by the
  GFPR and PFCG pilots), frozen.
- Dataset: `ogbench/cube_single_expert.h5`.
- Reset and true rollout: `diagnosis/scripts/76_ogb_true_endpoint_corrected.py`,
  which restores `qpos`/`qvel` and realigns `_prev_qpos`/`_prev_qvel` before
  stepping, and renders under the controlled renderer path.

### Declared task-state subset for the state anchor

Two 3-vectors, both in metres, reported **separately** rather than concatenated:

- `object_pos`   = `raw_env._data.joint("object_joint_0").qpos[:3]`
                   (the accessor behind `cube_distance`,
                   `diagnosis/scripts/72_ogb_stage0_candidate_audit.py:189`)
- `effector_pos` = `raw_env._data.site_xpos[raw_env._pinch_site_id]`
                   (`pinch_position`, `diagnosis/scripts/84_ogb_matched_refit.py:124`;
                   OGBench reports this as `proprio/effector_pos`)

**No per-dimension std normalization.**  Both quantities are positions in
metres, so the units are already commensurable and raw metres are the natural
isotropic chart.  Std normalization would rescale x/y/z by their dataset spread,
which is itself an arbitrary change of chart — the exact confound this protocol
is built to avoid.  Std normalization is reinstated only if a future subset
mixes units.

Reporting the two separately removes any relative-weighting choice and supplies
a built-in pipeline control:

- `K_true_state_effector` is the curvature of the arm kinematics under action
  perturbation.  It is directly actuated and is expected to be low.  A high
  value indicates a defect in the measurement pipeline (reset, action scaling,
  triplet construction), not a property of the physics, and blocks the run.
- `K_true_state_object` is the contact-mediated component and is the quantity of
  scientific interest.

**Excluded by declaration:** gripper aperture (the finger DOF) is not part of
the anchor, because it is not a position and would reintroduce the unit-mixing
problem.  Grasp and release therefore enter the analysis only through
`object_pos` and through the contact-pattern mode label, never through the
anchor.  Aperture is logged as a covariate alongside the mode label.

## Objects

For a start observation `o_t` with exact simulator state `s_t`, an action chunk
`a = (a_t,...,a_{t+H-1})`, and a perturbation `delta`:

- Model map:      `Phi_H(a) = F^H(E(o_t), a)`            (predicted latent)
- Realized map:   `Psi_H(a) = E(o_H^sim(s_t, a))`        (encode of true rollout)
- Second differences: `D2Phi = Phi(a+d) - 2Phi(a) + Phi(a-d)`, likewise `D2Psi`
- First differences:  `v+ = Phi(a+d) - Phi(a)`, `v- = Phi(a) - Phi(a-d)`

Both maps live in the **same latent chart** (same frozen encoder `E`), so their
difference is type-correct.

## Metrics

1. `K_model  = ||D2Phi|| / (||Phi(a+d) - Phi(a-d)|| + eps)`
2. `K_true   = ||D2Psi|| / (||Psi(a+d) - Psi(a-d)|| + eps)`
3. `E_K      = ||D2Phi - D2Psi|| / (||Psi(a+d) - Psi(a-d)|| + eps)`   **primary**
4. `G_K      = K_model - K_true`                                       (signed gap)
5. `A_K      = cos(D2Phi, D2Psi)`                                      (alignment)
6. `S_sens_model = ||Phi(a+d) - Phi(a-d)|| / (2||d||)`, and `S_sens_true` likewise
7. Radial/angular decomposition of every curvature quantity, using the exact identity

       ||D2Phi||^2 = (||v+|| - ||v-||)^2 + 2||v+||*||v-||*(1 - cos(v-, v+))
                     \______radial______/   \_________angular_____________/

   Report `E_K` split into radial and angular parts.  If the mismatch is almost
   entirely radial, a cosine (angular) regularizer targets the wrong component
   and Stage 2 must be redesigned before any training run.

**Denominators use TRUE sensitivity, never model sensitivity.**  Normalizing by
the model's own response would reward an action-insensitive model.

### What `E_K` is and is not

`D2Phi - D2Psi = D2(eps)` where `eps(a) = Phi_H(a) - Psi_H(a)`.  `E_K` therefore
measures the **nonlinear action-dependence of the model's error**, not rollout
MSE: it removes constant and locally affine components of the prediction error
*along the probed action direction*.  This identity holds only for an exactly
symmetric, unclipped triplet (see the clipping rule below).

### Chart dependence — binding constraint on Stage 2

Curvature is **not** coordinate-invariant: under a smooth invertible
reparameterization `z = g(s)`, `d2(g o h) = g''(h')^2 + g' h''`.  Consequences,
both locked here:

- `K_true_state` and `K_true_latent` are **never** compared in absolute terms.
  The simulator-state chart is one arbitrary chart (and a heterogeneous one:
  positions, velocities, quaternions).
- Stage 1 is safe because it uses a **single frozen checkpoint** = a single chart.
- Stage 2 compares arms with **different encoders = different charts**.  A raw
  cross-arm `E_K` comparison is therefore not a meaningful statement on its own.
  The primary cross-arm metric MUST be chart-invariant: **rank agreement between
  the model-induced ordering and the true ordering over the same candidate set**
  (Spearman rho and top-k recall, reusing the CEM preselection audit).  `E_K` is
  reported as a secondary mechanistic quantity.

## Perturbation and validity rules

- Symmetric triplets only: `a-d`, `a`, `a+d`.
- **Clipping discard.**  Any triplet where either arm is clipped by the action
  bounds is discarded, not corrected: asymmetric spacing `h+ != h-` contaminates
  the second difference with a first-order term `f'(a)(h+ - h-)`, which is exactly
  the affine component the diagnostic relies on cancelling.
- **Discard rate is logged per stratum.**  Actions near the bounds are not
  uniformly distributed across contact strata, so a differential discard rate
  biases the same-mode vs cross-mode comparison.  If rates differ materially
  across strata, shrink sigma until they balance and report both settings.
- Scale sweep: `sigma in {0.025, 0.05, 0.10, 0.20} x normalized action range`.
- Horizon sweep: `H in {1, 3, 5, 10}`.
- All differences computed in **float64**.

## Numerical floor

With a deterministic simulator reset and a deterministic encoder, repeated
`Psi` evaluation gives a floor of exactly zero.  **That is a false all-clear.**
The binding small-`delta` limit is catastrophic cancellation: `D2` subtracts
`O(1)` quantities to obtain an `O(delta^2)` result.  The locked floor test is
therefore twofold:

1. Repeat-evaluation floor `||Psi^(1) - Psi^(2)||` at `delta = 0` (detects any
   nondeterminism in reset, physics, renderer, or encoder — must be reported
   even when zero).
2. Existence of a clean `delta^2` regime in `D2(delta)`.  Any scale at which
   `D2` flattens or becomes erratic at the low end is excluded from the fit.

## Boundary detection

Two independent detectors, reported against each other:

- **Counterfactual mode change** (simulator): contact patterns `m-`, `m0`, `m+`
  from the three true rollouts.  A perturbation is `cross-mode` if `m- != m0`
  or `m+ != m0`.  This is a property of the *perturbation*, not of the logged
  state.  Contact pattern is a documented **proxy** for dynamics mode; it misses
  non-contact mode changes (slip/roll transitions, joint limits).
- **Scaling exponent** (latent, label-free): fit `log D2(delta) = alpha log|delta| + c`
  on the **raw** second difference.  `alpha ~ 2` smooth, `alpha ~ 1` kink,
  `alpha ~ 0` jump.  Never estimate `alpha` from the normalized `K`: at a
  symmetric kink `Psi+ = Psi-`, the denominator vanishes while `D2` is large.
  Report fit residual/R^2; assign no `alpha` to samples not described by a
  single slope across the sweep.

Strata: `same-mode non-contact`, `same-mode contact`, `cross-mode`.

## Region comparison: offline vs planner-visited

Cached CEM populations (`physical_search_distillation/outputs/h0/populations`,
`counterfactual_flow/outputs/ogbench_cube_phase0/locked_shards`) supply the
planner-visited region.  Two variants, both required, because a single fixed
`delta` confounds location shift with exploration scale:

- `K_fixed`: same `sigma` as the offline measurement -> isolates **location** shift.
- `K_local`: `delta ~ N(0, alpha^2 Sigma_k)` from the CEM covariance at iteration
  `k` -> geometry at the scale the planner actually queries.  Late-iteration
  `Sigma_k` is narrow; report the realized `||delta||` against the numerical floor
  and exclude scales below it.

Claiming "the planner enters highly curved regions" requires `K_local` to move,
not only `K_fixed`.

## State-space anchor (mechanism localization only)

Computed on the declared subset above, in raw metres, separately for object and
effector:

    K_true_state_X = ||X_H^+ - 2 X_H^0 + X_H^-|| / (||X_H^+ - X_H^-|| + eps)

Used **only** to localize mechanism in Stage 2, never to adjudicate correctness:

- `K_true_latent` unchanged, `E_K` down  -> predictor repair.
- `K_true_latent` down, `K_true_state_object` unchanged -> representation reshaping.

Representation reshaping is **not** by itself a defect: a learned encoder is
entitled to choose a coordinate system in which the dynamics manifold is
straighter.  It is called harmful only if task-relevant fidelity degrades at the
same time — CRA_eff, boundary-blindness, physical probes, action sensitivity.

## Stage 1 readouts

1. `K_model` and `K_true_latent`, per stratum.
2. `E_K` (primary), radial/angular split.
3. `G_K` signed gap and `A_K` alignment.
4. `S_sens_model` and `S_sens_true` versus `H`.
5. `K_model(H)` and `K_true_latent(H)` on shared axes.
6. `delta`-sweep, log-log exponent `alpha`, numerical floor.
7. Counterfactual mode stratification.
8. Agreement between the contact-pattern and scaling-exponent detectors.
9. Offline vs CEM region, `K_fixed` and `K_local`.
10. State-space anchor, object and effector separately (effector doubles as
    the pipeline sanity control).

Trajectory-clustered bootstrap CIs on every reported statistic.

## Stage 1 decision rule

**Kill** if any of:
- same-mode `E_K` is small **and** does not increase in the planner-visited region;
- every large mismatch is accounted for by genuine mode boundaries (`alpha < 2`)
  or by the numerical floor;
- `K_model` is low everywhere while `S_sens_model` is also low (the model is
  action-deaf, a different defect that this program does not address).

**Green light** (strongest form): `E_K` large on same-mode samples, with
`E_K_CEM > E_K_offline`, while the true dynamics in those same samples have
`alpha ~ 2` — i.e. the planner queries a region that is physically smooth but
whose model-error geometry is nonlinear.

## Stage 2 arms (frozen now, executed only if Stage 1 greenlights)

1. `LeWM`
2. `+ open-loop multi-step prediction`   (matches the AS computational graph:
   backprop through `H` unrolled steps, no teacher forcing)
3. `+ open-loop multi-step prediction + AS`      `L_AS = 1 - cos(v-, v+)`
4. `+ open-loop multi-step prediction + norm-symmetry`  `L_norm = (||v+|| - ||v-||)^2`

Arms 3 and 4 are the **angular and radial halves of the same curvature identity**,
not treatment and placebo.  Their `lambda` values are matched by gradient norm.

Secondary ablation, not a control: independent-`delta` symmetry breaking
`(a+d1, a, a-d2)`.  It is *not* a null — forcing `cos(v+, v-) -> 1` across
independent directions pressures `J` toward a rank-1 response — so it is read as
a symmetry stress test only.

Guard metrics on every arm: CRA_eff and boundary-blindness, contact-stratified;
prediction loss; action sensitivity; `K_true_latent` and `K_true_state`.

Cross-arm primary metric is the chart-invariant rank agreement defined above.

## Claim boundary

Stage 1 licenses claims about the action geometry of the frozen
`quentinll/lewm-cube` checkpoint on OGBench-Cube at the tested perturbation
scales and horizons.  Nothing here is a claim about Push-T, about other
checkpoints, or about real-robot data.  It does not
establish that curvature mismatch causes planning failure; only Stage 2 can
address that, and only for the arms actually run.

## Pre-execution amendment (2026-08-24, before any measurement)

Two defects in the metric section above were found while implementing `core.py`
and are corrected here.  No measurement had been run, so no result could have
influenced these changes.

**1. Inconsistent denominators.**  Metric 1 defined `K_model` against the
model's own span while the note below the list required true sensitivity in all
denominators.  Both cannot hold, and `G_K = K_model - K_true` was meaningless as
written because it subtracted two differently-normalized quantities.  Locked
resolution — three quantities, all reported:

- `K_model_self = ||D2Phi|| / (||Phi+ - Phi-|| + eps)`  — intrinsic shape of the
  model map, the like-for-like counterpart of `K_true`.
- `K_model_true = ||D2Phi|| / (||Psi+ - Psi-|| + eps)`  — used for every
  cross-map comparison; immune to rewarding an action-deaf model.
- `G_K = K_model_true - K_true`, i.e. a common denominator on both terms.
- `E_K` keeps the true-span denominator, unchanged.

The three are exactly related by `K_model_self / K_model_true = S_true / S_model`,
so the sensitivity ratio is reported alongside them and none may be omitted.

**2. Radial/angular split is reported as fractions.**  The identity is stated on
the unnormalized `||D2||^2`; under any common denominator both components scale
together, so the split is reported as `radial_fraction + angular_fraction = 1`
plus the normalized magnitudes.  For `E_K` the same decomposition is applied to
the error map with `v+_eps = v+_Phi - v+_Psi` and `v-_eps = v-_Phi - v-_Psi`.

**3. R^2 convention in the jump regime.**  Found by the unit tests before any
measurement.  When `||D2||` is constant across scales the response IS the
`alpha = 0` jump regime, but the ordinary `R^2 = 1 - ss_res/ss_tot` is `0/0`
there.  Returning NaN would make the fit-quality filter discard exactly the
discontinuity samples the detector exists to find.  Locked convention: when the
total sum of squares vanishes, `R^2 = 1` if the residual also vanishes (a flat
line is an exact fit) and `0` otherwise.

## Stage 1 sample size and measurement conventions (preregistered 2026-08-24)

**n = 64 snapshots: orders 0-63 of the locked PERD manifest**
(`physical_search_distillation/outputs/h0/manifest.json`, 128 rows).  Taken by
order, no selection.  This manifest rather than a fresh one because readout 9
needs the cached CEM populations, which are keyed to it
(`.../outputs/h0/populations/snapshot_000..063`); a fresh manifest would turn a
free measurement into a new collection job.  Nothing is trained and no outcome
is consulted, so manifest reuse cannot leak.

Bootstrap is clustered on snapshot, which is the indivisible group.

**Perturbation scale.**  `--sigmas` are fractions of the **raw** action range
`high - low`.  The normalized space is unbounded and has no "range"; directions
are drawn isotropically in raw action units and converted through the
StandardScaler so the model and the simulator receive the same physical
perturbation.  For `cem_local` the same ratios (0.125, 0.25, 0.5, 1.0 of the top
scale) are expressed in units of the recorded CEM `proposal_std`.

**Clip validity is decided in raw action space**, where the bounds live, on every
one of the `horizon x action_block` primitive actions in the chunk.

**Contact pattern is the union over the whole rollout**, not the terminal
instant: a perturbation that makes contact mid-rollout and then separates has
still crossed a dynamics mode.  Pattern elements are unordered MuJoCo geom-id
pairs.

**Centre action.**  `dataset` source: the logged action chunk at the snapshot's
`storage_row`, read from the **unfiltered** action array (the finite-row mask
used to fit the scaler renumbers rows).  `cem_fixed` / `cem_local`: a uniformly
drawn elite of the recorded population, since elites are the candidates the
planner retains and refits around.

## Second pre-execution amendment (2026-08-24, still before any measurement)

An external review found further defects in the protocol and in the aggregation
code. Nothing had run, so no result could have influenced these changes. Job
`45646` was cancelled while `PENDING`; see `JOB_LEDGER.md`.

### A. Horizon sweep is `{1, 3, 5}`, not `{1, 3, 5, 10}`

The frozen planning configuration is `PlanConfig(horizon=5, action_block=5)` and
the cached CEM populations are `(2, 96, 5, 25)`. `H = 10` is unreachable without
changing the planning configuration the checkpoint was released with, which
would break comparability with every other result in this repository. The
protocol was wrong, not the array.

### B. Declared thresholds

These label rows of descriptive tables and gate fit quality. Every CI-backed
claim rests on continuous quantities, not on these cut-points.

| name | value | role |
|---|---:|---|
| `curvature_high` | 0.25 | taxonomy cross-tab cut-point |
| `mismatch_high` | 0.25 | taxonomy cross-tab cut-point |
| `effector_gate` | 0.25 | pipeline gate on `K_true_state_effector` |
| `smooth_alpha_low` | 1.5 | `alpha` at or above this counts as the smooth regime |
| `min_r2` | 0.90 | below this, no `alpha` is assigned |
| `min_sensitivity_quantile` | 0.10 | see C |

### C. Minimum true-sensitivity gate

`E_K`'s denominator `||Psi+ - Psi-||` vanishes not only at a symmetric kink but
at any stationary point or null direction of the realized Jacobian. There a
tiny second-order error yields an unbounded `E_K` while the map is perfectly
smooth (`alpha ~ 2`) — which would satisfy the previous strong green-light
condition. Two corrections:

1. A record enters the `E_K` statistics only if its realized sensitivity
   `S_true` is at least `min_sensitivity_quantile` times the median `S_true`
   within its own `(source, horizon, sigma)` cell. The rule is fixed in advance
   and computed from the data; the number of records it removes is reported.
2. Two denominator-free companions are reported alongside `E_K` always:
   `e_k_absolute = ||D2 eps|| / ||d||^2` and
   `e_k_pathlen = ||D2 eps|| / (||v-_Psi|| + ||v+_Psi||)`.

Normalized curvature depends on the perturbation scale even in a smooth regime,
so **sigmas are never pooled**.

### D. Fit-quality gate

`R^2` is now used, not merely reported: no `alpha` is assigned when
`R^2 < min_r2`, and `alpha` is counted **once per fit**, keyed by
`(snapshot, source, horizon, direction)`, not once per record in the sigma
sweep.

### E. Primary contrast

One preregistered cell carries the confirmatory claim. Everything else is a
response curve.

    Delta E_K = E_K[cem_fixed, H=5, sigma=0.10] - E_K[dataset, H=5, sigma=0.10]

Computed **paired by snapshot**: the per-snapshot median over directions within
the cell for each source, then the per-snapshot difference, then a
snapshot-clustered bootstrap CI over those differences. The previous rule
compared unpaired point estimates and took a `max` over CEM sources, which is a
selection bias; `cem_local` is a secondary, separately reported contrast.

### F. Two new readouts

- `E_J = ||(Phi+ - Phi-) - (Psi+ - Psi-)|| / (||Psi+ - Psi-|| + eps)`, the
  first-order mismatch. `E_K` annihilates every affine error component, which
  is its headline property and equally its blind spot: an affine Jacobian error
  can reorder the planner's candidates completely. Nothing may be attributed to
  curvature without `E_J` held as a covariate.
- The exact scalar-cost decomposition, on the quantity CEM actually ranks by:

      D2 C = 2<r0, v+ - v-> + ||v+||^2 + ||v-||^2
             \___residual___/  \____Gauss-Newton___/

  reported for the model cost and for the realized cost under the same goal
  embedding, with local concavity flagged by `ratio = residual / gn < -1`. This
  is an identity, not an `O(d^4)` approximation, and it replaces the Hessian
  argument as the analytical centrepiece. A linear map has `||D2 Phi|| = 0` yet
  a nonzero cost curvature, so the vector curvature of `Phi` cannot be the whole
  story for a planner that only ever sees the scalar.

### G. Contact mode, redefined for Cube

The previous rule — any geom-pair contact, unioned over the rollout — is void
here. The cube rests on the table at essentially every step, so it would put
every sample in the contact stratum and leave `same_mode_non_contact` empty.
Replaced by:

- Categories resolve from MuJoCo **body ids**, never geom names: the cube body
  is the body of `object_joint_0`, the static scene is the world body, anything
  else touching the cube is the robot. The run is refused if the cube body owns
  no geoms.
- Only **cube-robot** contact stratifies. Cube-table is a covariate.
- Traces are **per step**, so contact onset is part of the mode signature; two
  rollouts touching the same bodies at different moments are cross-mode when
  their onsets differ by more than `onset_tolerance` (default 1 step).

### H. Chart invariance of the Stage-2 cross-arm metric

The first amendment required "rank agreement between the model-induced ordering
and the true ordering" without naming the reference, which is ambiguous and only
one reading is chart-invariant. Ranking is invariant to monotone transforms of a
*scalar cost*, not to arbitrary nonlinear reparameterization of the latent
space. The reference is therefore the **physical** task cost — object-to-goal
distance in metres — which does not live in the latent chart at all. Ordering
against true-endpoint latent L2 is **not** chart-invariant and may not be used
for cross-arm comparison.

### I. Precision caveat

Model and encoder evaluations run in float32 inside the network; casting the
outputs to float64 before differencing does not recover precision lost in the
forward pass. The float64 rule therefore bounds the differencing error only.
The binding small-`delta` check remains the existence of a clean `delta^2`
regime in the raw second difference, and the reported `e_k_absolute` makes the
scale explicit.

### J. Stage-2 causal arm must match the measured estimand

Stage 1 measures `||D2 Phi - D2 Psi||`, while a straightening loss drives
`D2 Phi -> 0`. These coincide only where `D2 Psi` is small, and the previous
green-light condition required only `alpha ~ 2`, i.e. smoothness — a strongly
curved map is perfectly smooth and has `alpha = 2`. The estimand-matched
intervention is added as the primary causal arm:

    L_match = || D2 Phi - stopgrad(D2 Psi) ||^2

It queries the simulator during training, so it is a mechanistic oracle rather
than a deployable method — the same move the oracle-dynamics ladder made
earlier in this project. The straightening losses become simulator-free
surrogates, admissible only where Stage 1 has shown the realized curvature to be
small on the training support. Green-lighting the surrogates additionally
requires `K_model > K_true`, not merely a large `E_K`.

## Third pre-execution amendment (2026-08-25, after smoke job 45863)

Job `45863` completed the pipeline end to end and produced no scientific
result, by design: it is a smoke run on one snapshot. It exposed three defects,
all fixed here. No Stage-1 measurement has been consumed.

### K. Perturbation directions are feasible by construction

The all-or-nothing clip rule made `H=5` unmeasurable. A chunk spans
`horizon * action_block` primitive actions -- 25 at `H=5` -- and expert and
CEM-elite actions saturate their bounds often, so the probability that all 25
stay interior under a random perturbation is negligible. Job `45863` recorded:

| source | records | valid | valid horizons |
|---|---:|---:|---|
| dataset | 16 | 8 | `H=1` only |
| cem_fixed | 16 | 0 | none |
| cem_local | 16 | 0 | none |

Both planner-visited sources yielded nothing, so readout 9 was unobtainable.

Directions are now made feasible instead of drawn and rejected. For each
component the headroom to the bounds gives a cap on `|base[i]|` at the top
sigma; saturated components are masked out and the surviving vector is shrunk
by a **single global factor**, never clipped per component, because the
scaling-exponent fit requires the identical direction at every scale. The
masked fraction and shrink factor are recorded per direction under
`direction_feasibility`, and a fully saturated chunk is skipped and reported
rather than silently dropped. The clip check is retained as a self-check: with
feasible directions it must now pass, and a failure is a bug.

### L. The scale sweep reaches two decades lower

Job `45863` fitted `alpha = 0.30` and `0.44` with `R^2 = 0.42` and `0.23` over
`sigma in {0.025 ... 0.20}`, and `e_k_absolute` fell as roughly `delta^-1.2`
rather than staying flat. The probe was therefore **not in a local regime at
any tested scale**: at those perturbations the realized endpoints are close to
decorrelated. The sweep becomes

    sigma in {0.00125, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.10, 0.20}

spanning 2.2 decades. If no sub-window admits a `delta^2` regime at
`R^2 >= min_r2`, then curvature is not well posed on this environment at any
scale, and **that is the Stage-1 result to report** rather than a defect to
work around.

The `R^2` gate behaved correctly on first contact with real data: it rejected
every fit, so `same_mode_alpha_is_smooth` returned `null` and the verdict was
`INCONCLUSIVE_NO_USABLE_ALPHA`. Under the pre-amendment vacuous `all()` the
same input would have green-lit on no evidence.

### M. Contact bodies are recorded, not just the booleans

Job `45863` resolved the cube cleanly -- body 24 `object_0`, one geom, 5 world
geoms of 64 -- but reported `table_contact = False` on every record, while the
cube is expected to rest on a table. Two readings fit that identically: the cube
is grasped at this snapshot, or the table is not the world body and its contact
is being labelled robot. Only body identity separates them, so the classifier
now returns the set of bodies touching the cube, the per-record union is stored,
and the body-id-to-name table is written into `contact_resolution`.

### N. Preliminary observation, explicitly not a result

On the 8 valid records of one snapshot at `H=1`, at scales now known to be
outside any local regime, `K_true ~ 1.10` against `K_model ~ 0.024`: the
realized map was some 46x more curved than the model map. That is the
"model erases real nonlinearity" row of the taxonomy, not the "spurious model
curvature" row the surrogate losses would target, and the verdict branch added
in amendment J (`MISMATCH_BUT_MODEL_UNDER_TRUE`) fired on first contact with
data. One snapshot, one horizon, eight records, invalid scales: this licenses
no claim and is recorded only because it will look like confirmation if the
same sign appears at `n=64` and must not be treated as a prediction that was
made in advance.

## Fourth pre-execution amendment (2026-08-25, resolution-floor investigation)

Job `46142` (the corrected probe, all three prior bugs fixed, 256/256 valid)
ran the primary contrast through the real `aggregate.py` pipeline and returned
`KILL_NO_SAME_MODE_MISMATCH`, but on a thin evidence base: only 9/32 alpha fits
cleared the `R^2 >= 0.90` gate, and same-mode `E_K`'s CI lower bound (0.226) sat
just under `mismatch_high` (0.25). Before spending compute on `n=64` to sharpen
that boundary, the resolution floor itself is investigated directly, per the
user's framing: **compare the rendered frames at the same tiny action
perturbations to see whether the change is lost at the rasterizer or at the
encoder/patch tokenization.** If the image itself barely moves, more snapshots
cannot rescue the measurement; the sigma range or the reference construction
would need to change instead.

**Method.** `measure_one` already renders the exact triplet the latent
comparison uses; the pixel-space second difference is computed on those same
frames at zero extra render cost, so the pixel and latent alphas are fit on
literally the same data, not a separately-drawn comparison:

    pixel_d2_norm = ||Triplet(frame(a-d), frame(a), frame(a+d)).d2||   (raw pixel space, flattened, float64)

fit against `||d||` the same way as the latent alpha, with one difference: the
floor is not the repeat-evaluation floor (pixel values are integer-quantized,
typically 8-bit) but one raw grey-level step, `floor=1.0`, since that is the
natural unit below which no render difference can be expressed at all.
`pixel_max_abs_diff` and `pixel_nonzero_diff_fraction` are recorded per record
as a direct, alpha-independent check for "did the render change at all."

**Reading rule**, per the user's framing:

- pixel span/`max_abs_diff` near zero even at the largest sigma -> the floor is
  the **rasterizer**: sub-pixel motion at these action scales, and no snapshot
  count fixes it. The sigma range or the observation itself needs to change.
- pixel alpha close to 2 (image moves smoothly) while latent alpha stays flat
  -> the floor is the **encoder / patch tokenization** (ViT-tiny, patch size
  14, so an image displacement smaller than roughly one 14px patch may not move
  a patch's embedding until a token boundary is crossed). This would be a
  reportable finding in its own right, independent of the curvature-mismatch
  question, and reframes what "local" means for this observation.

Snapshot selection, sigma range, and horizon are unchanged (probe snapshots
0-7, `H=5`, the same 8-point sweep); only the pixel-space fit is added.

## Fifth pre-execution amendment (2026-08-25): the realized reference is not
## well posed on rasterized observations

Jobs `46172` (H=5) and `46200` (H=1) ran the identical snapshot set, sigma
sweep, feasible-direction logic and centre, differing only in the measured
horizon. Because base directions are drawn over the full chunk shape and then
truncated, the H=1 perturbation is exactly the first row of the H=5
perturbation: a clean horizon contrast.

### The open-loop compounding hypothesis is refuted

Pixel alpha is unchanged between horizons (median 0.47 at both H=5 and H=1),
with `R^2` typically 0.97-1.00. Non-smoothness is fully present at a single
action block; it does not accumulate over 25 open-loop primitive steps.

### The mechanism, and why no sigma window can rescue it

The stable value `alpha ~ 0.5` is not noise. A rasterized render is
piecewise-constant in scene geometry: each pixel is a step function of object
position. A translating sharp edge sweeps roughly `L * delta` newly-covered
pixels, each flipping by an O(1) grey-level amount, so

    ||Psi(a+d) - Psi(a-d)||_2  ~  sqrt(delta)          -> alpha ~ 0.5

and because the pixel sets swept on the two sides are near-disjoint,
`<v+, v-> ~ 0`, giving the second prediction

    ||D2 Psi||  ~  sqrt(||v+||^2 + ||v-||^2)  ~  ||Psi+ - Psi-||   -> d2/span ~ 1

Both were tested on all 32 configurations (2 horizons x 2 sources x 8
snapshots): measured `alpha_span` median **0.52**, measured `d2/span` median
**1.03**, tightly concentrated. The consequence is that the normalized latent
curvature `K = ||D2|| / span` is approximately **1 regardless of delta** -- it
measures edge-sweep geometry, not curvature.

The latent inherits this: `alpha_latent` tracks `alpha_pixel` per snapshot, and
the encoder is not the origin of the non-smoothness. Simulator **state** remains
smooth (`alpha_object` reaches 2.03, 2.37, 3.09, 2.50 at H=1), confirming the
physics is differentiable and only the observation path is not.

**No delta window exists.** A clean `delta^2` regime requires the displacement
to be much larger than the antialiasing width (~1 px) while remaining local
relative to the dynamics nonlinearity. Measured object displacement spans
4.5e-5 m to 7.1e-3 m across the whole sweep, i.e. roughly 0.01 to 1.6 pixels at
224x224. The upper end of the entire admissible action range barely reaches one
pixel of motion, so the window is empty. This is not fixable by sigma range,
snapshot count, horizon, or encoder choice.

### Consequence for the program

The realized reference `Psi = E o render o sim`, which is the novel core of this
diagnostic, is **not a valid curvature reference for pixel-based world models**.
Stage 1 as specified cannot be run to a meaningful conclusion, and the `n=64`
array is not submitted.

Scope of the claim: 8 snapshots, OGBench-Cube, one renderer configuration at
224x224, one frozen checkpoint. The argument for why no sigma window exists is
geometric rather than statistical, but the pixel-scale numbers backing it are
from this configuration only.

What survives: curvature is well defined in simulator state space, where
`alpha ~ 2` is measured directly. Any reformulation must either define the
reference in a space where the observation map is differentiable, or abandon
second-order quantities on rasterized observations.

## Sixth pre-execution amendment (2026-08-25): the model map is exactly smooth;
## the floor was numerical, the reference is not

Job `46382` ran the identical snapshot, sweep, directions and centre at
`--model-dtype float32` and `float64`, differing only in forward precision.

**Result.** In float64, `||D2 Phi|| / ||d||^2 = 0.1875` at every one of the eight
scales, constant to four significant figures across 2.2 decades:
`alpha_model = 2.000`, `R^2 = 1.000`. In float32 the same quantity is floored at
`~2e-3` and fits `alpha = 0.582`, `R^2 = 0.734`. The two agree only at the two
largest scales, where float32 still has signal.

The model map `Phi_H` is therefore **exactly smooth in the action sequence**, as
a neural rollout should be, and `0.1875` is its curvature magnitude -- a
well-defined quantity. The apparent non-smoothness of the model side was
entirely an artifact of float32, i.e. the caveat recorded in amendment I and
left unresolved until now.

**Enabling the test required patching upstream.** `stable_worldmodel/wm/lewm/
module.py:204` hardcodes `x = x.float()` in `Embedder.forward`, the only such
downcast in that module, immediately before the action `Conv1d`. Left alone it
silently defeats a float64 forward and raises at the Conv1d bias. It is patched
at runtime, only under `--model-dtype float64`, guarded by an `inspect.getsource`
check, and recorded per run as `embedder_float32_patch_applied`; the vendored
checkout is untouched so every other pilot in this repo keeps its provenance,
and float32 runs stay bit-identical to all earlier results.

**Consequence: the diagnosis is now surgical.**

| map | smooth? | measured |
|---|---|---|
| `Phi_H` (model rollout) | yes | `alpha = 2.000`, `R^2 = 1.000` (float64) |
| simulator state | yes | `alpha ~ 2` |
| `Psi_H = E o render o sim` | **no** | `alpha ~ 0.5`, `d2/span ~ 1`, mechanism confirmed |

Only the reference is broken, and it is broken geometrically rather than
numerically: no precision, sigma range, horizon, snapshot count or encoder
choice repairs a rasterizer. This also explains the earlier `E_K` numbers
mechanically: with `D2 Phi` clean and small and `D2 Psi` dominated by edge
sweep, `E_K` reduces to `||D2 Psi|| / span ~ 1`, which is exactly the measured
`d2/span` median of 1.03.

**Every float32 measurement in this program is affected.** All earlier
`k_model_*`, `g_k` and `e_k` values were computed with `D2 Phi` pinned at its
numerical floor below `||d|| ~ 0.27`, so the model side was *over*estimated at
small scales. The `KILL_NO_SAME_MODE_MISMATCH` verdict from job `46142` was
computed on that data and is not a valid reading of the model. It is superseded,
not confirmed, by this amendment.

## Seventh amendment (2026-08-25): the linear physical-state bridge fails
## Gates B and C; the reformulation is closed, not weakened

Reformulation tested: replace the broken latent reference with a physical-state
one. Train an affine probe `P(z) = ((z - mu)/sigma) W + b` on real observations,
decode both the model rollout and the simulator truth to metres, and compare
curvature there. Affine by construction, so `D2 (P o Phi) = W' . D2 Phi` exactly
and the probe cannot manufacture curvature.

**Gate A passed** (job `46383`, 480 states from 120 episodes disjoint from all
128 manifest episodes, held out by episode): held-out `R^2` 0.977-0.988 for
object XYZ, 0.946-0.977 for effector. But the held-out median error is
**13.7 mm** against a median triplet displacement of **0.33 mm**, so Gate A only
establishes that the latent carries cube position at workspace scale. It says
nothing about local resolution, which is why Gates B and C decide.

**Gates B and C both fail** (job `46386`, float64, 512 records, 345
non-degenerate across 12 snapshot-source cells):

| quantity | object | effector |
|---|---:|---:|
| `median ||D2 e||` | 6.03e-4 m | 5.93e-4 m |
| `median ||D2 s_true||` | 1.43e-4 m | 1.09e-4 m |
| **Gate B ratio** (threshold 0.25) | **4.20** | **5.44** |
| **Gate C** | **0/12** | **0/12** |

The effector control fails too, and it is the easiest possible case: the
end-effector is directly actuated and nearly linear in the action.

**Mechanism, confirmed rather than inferred.** The bridge *adds* curvature
rather than losing it: `median ||D2 bridge|| = 5.05e-4 m` against
`median ||D2 s_true|| = 1.43e-4 m`, a factor of 3.5. That is exactly the
predicted inheritance -- `D2 (P o E o sim) = W . D2 Psi`, and `D2 Psi` is the
rasterization edge-sweep term measured in the fifth amendment. Projecting
192 dimensions down to 3 does **not** annihilate it.

**The signal is hopeless against this floor.** The model's decoded curvature is
`median ||D2 P(Phi)|| = 1.12e-6 m`, i.e. **128x smaller** than the true physical
curvature and **540x smaller** than the bridge's own curvature error. Any
mismatch computed here measures the bridge, not the model.

**Consequence.** Every reference that reaches the model through
`E o render` inherits the rasterizer, whether compared in latent space (fifth
amendment) or decoded to metres (this one). Per the constraint fixed in advance
and repeated in `train_probe.py`, the response is **not** a nonlinear probe: an
MLP would manufacture curvature of its own and destroy the one property that
made this comparison well posed. The linear-bridge reformulation is closed.

What still stands, unchanged: `Phi_H` is exactly smooth (`alpha = 2.000`,
`R^2 = 1.000`, sixth amendment) and simulator state is smooth. Both sides are
individually well defined; what has no valid construction, so far, is a
*comparison* between them that does not pass through a rasterizing renderer.

## Eighth amendment (2026-08-25): ordinal endpoint runs clean; curvature does
## not predict misranking on the pilot, with one borderline exception

The ordinal reformulation removes the rasterizer from **both** sides for the
first time in this program. The model cost `||Phi_H(a) - z_g||^2` depends on the
action only through `F`: the encoder touches the initial observation and the
goal, both shared across the triplet, so the renderer contributes constants that
cancel from any comparison among the three. The physical cost is object-to-goal
distance read straight from simulator state. Comparing only the *ordering* of
three costs is invariant under any strictly increasing transform, so no shared
units and no chart alignment are needed.

**A design defect in the first run of the kill test, found and corrected before
reporting.** The predictor was `model_ratio = residual / gn`, but
`D2 C < 0 <=> ratio < -1` while `VALLEY` is defined as `D2 C > 0`, so small
`|ratio|` implies a valley *by construction*. Predictor and outcome were
partially circular, and that circular pairing produced the run's only CI-clean
effect (`false_valley`, -0.25, in the direction opposite the hypothesis). Re-run
with `k_model_self`, the curvature of the action-to-outcome **map**, which is
goal-free and independent of the shape definition.

**Result** (job `46404`, float64, 512 records, 193 usable after the
physical-spread filter, 7 snapshots):

| outcome | top-minus-bottom quantile | 95% CI | excludes 0 |
|---|---:|---|---|
| `shape_disagree` (primary) | -0.117 | [-0.291, +0.103] | no |
| `argmin_wrong` | -0.198 | [-0.383, +0.027] | no |
| `false_valley` | +0.163 | [+0.000, +0.382] | no (touches) |

Spearman between curvature and disagreement: **-0.036**. The preregistered
verdict keys on `shape_disagree` and returns
`NO_CURVATURE_MISRANKING_LINK_KILL_AS`.

**The one thing that is not simply null.** `false_valley` rises monotonically
across all four curvature quantiles -- 0.021, 0.042, 0.104, 0.184, a factor of
nine -- and it is the operationally meaningful outcome: the model reports a
local minimum where physics has none, which is precisely what a planner acts on
and settles into. Its CI touches zero, so on this evidence it is suggestive and
nothing more.

**Power.** This pilot has 7 snapshots against the preregistered Stage-1 `n = 64`,
roughly a ninefold shortfall, and the spread filter discards 62% of triplets.

**Disposition, and its limits.** `false_valley` was *not* the preregistered
primary; promoting it now, after seeing these numbers, would not be a test but a
restatement of them. The clean way to give it a fair hearing is a
generation/confirmation split: this pilot generated the hypothesis on snapshots
0-7, and it is confirmed or rejected on snapshots 8-63 of the same locked
manifest, with `false_valley` fixed as primary **before** those shards are read.
If it fails there, the curvature route is closed on its own terms rather than on
an underpowered null.

## Ninth amendment: CONFIRMATORY PREREGISTRATION, locked 2026-08-25 before any
## snapshot 8-63 shard is generated or read

The pilot (snapshots 0-7) generated a specific hypothesis it was not designed to
test: **high model-side curvature produces false local minima** -- the model
reports a valley where physics has none. Across the four curvature quantiles the
rate rose 0.021, 0.042, 0.104, 0.184, monotone, but the CI touched zero on 7
snapshots. Promoting it post hoc would restate the pilot rather than test it, so
it is fixed here as the primary outcome of an independent confirmatory run on
the 56 untouched snapshots.

**Confirmatory sample.** Orders **8-63** of the locked manifest
(`physical_search_distillation/outputs/h0/manifest.json`), 56 snapshots, none of
which has been measured, inspected, or aggregated at any point in this program.
Snapshots 0-7 are the generation set and are excluded from the confirmatory
analysis entirely; they are not pooled in.

**Primary outcome.** `false_valley` rate: the model's cost over the triplet has
its minimum at the centre while the simulator's physical cost does not.

**Predictor.** `k_model_self`, the normalized curvature of the action-to-outcome
map, computed in float64. Goal-free and independent of the shape definition --
chosen because the pilot's first predictor (`model_ratio`) was partially
circular with this very outcome, as recorded in the eighth amendment.

**Test statistic.** Top-minus-bottom contrast across four curvature quantiles,
with a snapshot-clustered bootstrap 95% CI (10,000 resamples).

**Decision rule, fixed now.**
- **CONFIRMED** if the contrast is positive and its 95% CI excludes zero.
- **NOT CONFIRMED** otherwise. AS loses its empirical justification and the
  curvature route closes on its own terms.
- Secondary, descriptive only, never decisive: monotonicity of the rate across
  the four quantiles; `shape_disagree` and `argmin_wrong` contrasts.

**Everything else is held identical to the pilot** and may not be retuned:
sigmas `{0.00125, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.10, 0.20}`, horizons
`{1, 5}`, 2 directions, both action sources, `--model-dtype float64` with the
Embedder patch, physical-spread filter `1e-4` m, degeneracy rule as locked.

**Power.** The pilot yielded 193 usable triplets from 7 snapshot clusters and a
CI half-width of ~0.19. The confirmatory run has 56 clusters, roughly eight
times more, so the CI should narrow by about a factor of 2.8. At the pilot's
observed effect size (+0.163) that is comfortably enough to exclude zero. The
test is therefore capable of detecting the effect it was built for, and a null
here is informative rather than merely underpowered.

## Tenth amendment (2026-08-25): the confirmatory test CONFIRMS the false-valley
## hypothesis

Job `46416`, snapshots 8-63, run and analysed under the rule locked in commit
`733d77a` before any of these shards existed. 3584 records, 1177 usable after
the unchanged spread filter, 39 snapshot clusters.

**Primary outcome, as preregistered.**

| curvature quantile | median `k_model_self` | `false_valley` rate | n |
|---|---:|---:|---:|
| 0 | 0.0014 | 0.0034 | 294 |
| 1 | 0.0063 | 0.0170 | 294 |
| 2 | 0.0196 | 0.0544 | 294 |
| 3 | 0.0601 | 0.1763 | 295 |

Top minus bottom: **+0.1729, 95% CI [+0.1213, +0.2347]**, excludes zero. By the
locked decision rule this is **CONFIRMED**.

The rate rises monotonically across all four quantiles, a factor of 52 from
lowest to highest, and the effect size reproduces the pilot's (+0.163) almost
exactly on a disjoint sample -- an independent replication, not a re-reading.

**Secondaries behave as in the pilot and remain null**: `shape_disagree`
-0.006 (CI [-0.081, +0.078]), `argmin_wrong` +0.076 (CI [-0.012, +0.167]),
Spearman 0.047. Curvature does not predict ordinal disagreement in general. It
predicts one specific, operationally important failure: the model reporting a
local minimum that physics does not have.

**A stale automated field, flagged rather than quoted.** `curvature_vs_misranking.py`
still computes its `verdict` string from `shape_disagree`, the pilot's primary,
so it prints `NO_CURVATURE_MISRANKING_LINK_KILL_AS` on this run. That string does
**not** implement the preregistered rule for the confirmatory analysis and must
not be reported as the result; the primary is `false_valley` per the ninth
amendment. The field is left as-is rather than edited after seeing the data, and
this note records the discrepancy.

**What this does and does not license.** It establishes an association between
model-side action-space curvature and false local minima, on OGBench-Cube, one
frozen checkpoint, at the tested scales. It does not establish that reducing
curvature removes those minima, nor that removing them improves planning. Those
are the AS intervention's claims and require the causal chain: AS lowers
curvature -> false-valley rate falls -> CEM elite quality rises -> planning
success rises. The first link is now worth testing; none of the rest is
established.

## Eleventh amendment (2026-08-25): the false-valley effect is ANGULAR, so the
## cosine loss is on target -- and snapshots 64-127 are now reserved

`k_model_self` is total curvature; the AS cosine loss only touches the angular
half of the identity, so confirming the total does not license the loss. Tested
on the confirmatory shards, no new compute: the components are recovered from
the energy fractions already recorded, `k_rad = k sqrt(f_r)` and
`k_ang = k sqrt(f_a)`.

Radial and angular are parts of one total and are mutually correlated, so a
univariate test on each would show an effect for both. The design is
conditional: bin on one component, read the contrast along the other within
those bins.

**False-valley rate, radial (rows) x angular (columns), 1161 records, 39 snapshots:**

|  | ang0 | ang1 | ang2 |
|---|---|---|---|
| **rad0** | 0.007 (n=276) | 0.034 (n=87) | 0.167 (n=24) |
| **rad1** | 0.000 (n=100) | 0.046 (n=197) | 0.167 (n=90) |
| **rad2** | 0.000 (n=11) | 0.000 (n=103) | 0.150 (n=273) |

The rate rises monotonically along every row and is flat or falling down every
column.

| conditional contrast | top-minus-bottom | 95% CI | excludes 0 |
|---|---:|---|---|
| angular, holding radial | **+0.158** | [+0.102, +0.212] | **yes** |
| radial, holding angular | -0.017 | [-0.069, +0.035] | no |

**Verdict: `ANGULAR_DOMINATES_COSINE_AS_IS_ON_TARGET`.**

**Why, stated plainly rather than dressed up.** The median energy split is
**3% radial, 97% angular**: this model's action-map curvature is almost entirely
directional, which is a large part of why the angular component carries the
effect. The conditional analysis still earns its place -- it shows radial adds
nothing once angular is held -- but the honest summary is that cosine AS is on
target mainly because there is little else to aim at. A caveat with it: the two
components are positively correlated through the total, so the off-diagonal
cells are thin (n=11 and n=24 at the corners) and the conditioning is weaker
there than the well-populated diagonal suggests.

**Reservation, binding from now.** This decomposition used snapshots 8-63, so
those snapshots have now informed the *design* of the intervention and can no
longer serve as its evaluation. Orders **64-127** of the manifest, 64 snapshots,
have never been measured, inspected or aggregated at any point in this program.
They are reserved as the held-out causal-evaluation set and must not be touched
until the intervention is trained and its evaluation preregistered.

**What is licensed now.** Training cosine AS on the predictor with the encoder
frozen, so the latent chart is identical before and after and curvature is
comparable across arms. Nothing about planning is licensed yet; the chain
AS -> angular curvature down -> false-valley down -> CEM ordering better ->
planning success up still has three untested links.

## Twelfth amendment (2026-08-25): intervention design, locked before training

Three arms, encoder frozen throughout so the latent chart is identical across
arms and curvature is comparable.

| arm | description |
|---|---|
| 1 | original `quentinll/lewm-cube`, no further training |
| 2 | frozen encoder, predictor trained on an **open-loop H-step rollout** prediction loss |
| 3 | arm 2 exactly, plus `lambda * (1 - cos(v-, v+))` on the same triplet rollouts |

### A. Arm 2 must share arm 3's computational graph, not merely its data

"Same data, steps and optimizer" is not sufficient. The AS term requires three
`H`-step rollouts with gradients flowing through the unrolled predictor, while
the upstream LeWM objective is teacher-forced single-step
(`pred_loss = (pred_emb - tgt_emb)^2`, `scripts/train/lewm.py:80`). If arm 2 kept
the single-step objective, arm 3 would differ by *two* things -- open-loop
multi-step training and AS -- and any gain would be unattributable. This is the
confound identified at the very start of this program and it is binding here.

Both arms therefore train on the identical open-loop `H=5` rollout loss over the
identical sampled triplets; arm 3 adds only the AS term. Setting `lambda = 0`
would **not** be a valid arm 2 if the rollouts existed solely to serve the AS
branch, because no gradient would flow through them.

With the encoder frozen, SIGReg contributes no gradient (it acts on encoder
outputs), so the continuation objective reduces to the rollout prediction loss.

### B. What "held out" means for orders 64-127, stated rather than assumed

The original checkpoint was trained on the full OGBench-Cube dataset, which
includes the episodes behind snapshots 64-127. Excluding those episodes from
continuation training would leave arms 2 and 3 *less* exposed to them than arm
1, an asymmetry that would bias the very comparison the design exists to make.

So: **all three arms share identical training data**, and 64-127 is held out
from *our design decisions* -- which snapshots we looked at when choosing the
component to target, the threshold, `lambda`, and the checkpoint rule -- not
from the model's training distribution. That is the notion that matters here,
because the contamination risk being managed is analyst choice, not memorisation.

### C. Seeds, so the AS effect can be separated from continuation noise

Arm 1 is a single deterministic checkpoint. Arms 2 and 3 run **3 seeds each**,
fixed now. Every link is evaluated seed-wise, and arm 3 must beat arm 2 by more
than the seed spread of arm 2. A single-seed comparison cannot distinguish AS
from where continuation training happened to land.

### D. Guards during development (orders 0-63 only)

Tracked jointly, since curvature falling is not by itself good:
angular curvature down; rollout prediction quality not collapsed; action
sensitivity not collapsed. The cosine loss is scale-invariant, so it does not
*directly* reward shrinking the displacement to zero, but joint optimisation can
still reach an action-deaf predictor by another route, and that is the failure
this program has documented more than any other.

`lambda`, the checkpoint-selection rule, and the seed count are chosen on orders
0-63 and committed **before** any shard from 64-127 is generated. Trying several
`lambda` on the held-out set and keeping the best is explicitly excluded.

### E. Preregistered claim order for the held-out evaluation

1. **Manipulation check** -- does arm 3 lower angular curvature versus arm 2?
   Fail: the method does not do what it claims; stop.
2. **Mechanism** -- does the false-valley rate fall? Fail: curvature is not the
   causal lever, only a correlate.
3. **Planner interaction** -- CEM elite quality / candidate ordering / false
   elites. Fail: the local minima were fixed but search barely notices.
4. **Endpoint** -- planning success and physical regret.

A failure at any link is reported at that link; later links are not consulted to
rescue an earlier failure.

### F. Final three details, locked 2026-08-25 before any training run

1. **One script, one flag.** Arms 2 and 3 are the same training script with
   `lambda_as = 0` versus `> 0`. Arm 2 also builds the symmetric triplet and
   runs the same three `H=5` rollouts; it simply carries no AS gradient. Matched
   in graph and compute, not merely in objective family. The base loss is the
   open-loop `H`-step rollout prediction loss on the centre chunk, so both arms
   receive multi-step gradient; the `+-delta` rollouts serve AS alone, and the
   only difference between arms is the AS gradient itself.
2. **Paired seeds.** Seed `k` of arm 2 and seed `k` of arm 3 start from the same
   checkpoint and share batch order and perturbation RNG. The primary contrast
   is the **within-seed paired difference** `AS - continuation`, which removes
   the between-run variance that a two-sample comparison would carry.
3. **`lambda` grid, `lambda`-selection rule and checkpoint rule are committed
   before any 64-127 shard exists.** Selecting the AS checkpoint by best
   false-valley while selecting arm 2's by best prediction loss is specifically
   excluded; the rule must be identical across arms.

**Primary criterion.** AS must improve on **arm 2** consistently across paired
seeds. Arm 1 is context -- it says what continuation and multi-step training do
on their own -- not the comparison. The causal contrast for AS is arm 3 vs arm 2.

### G. Development protocol, locked 2026-08-26 before the lambda sweep

**Grid.** `lambda in {0, 0.01, 0.03, 0.1, 0.3}`, 3 paired seeds each = 15 runs.
`lambda = 0` is arm 2 and is the paired control for every other lambda at the
same seed.

**Training budget, fixed now for every arm.** 1000 steps, batch 16, final
checkpoint. No early stopping, and specifically no checkpoint selection by
false-valley -- the rule is identical across arms, which is what makes the
comparison a comparison. The budget is set by compute, not by any observed
result.

**Guards are measured on a FIXED diagnostic manifest, never from the training
log.** Each training log point sits at a different sampled sigma and direction,
so `action_sensitivity` there varies by more than an order of magnitude for
reasons that have nothing to do with the arm. Every arm is therefore re-measured
on identical snapshots, identical directions and identical sigmas: dev orders
`{0, 5, 10, ..., 55}` (12 snapshots), `dataset` source, `H = 5`, the standard
8-point sigma sweep, float64. Only then is `lambda = 0` versus AS a clean
comparison.

**Selection rule, applied in this order on orders 0-63 only.**
1. Must lower angular curvature versus the paired `lambda = 0`.
2. Reject if the base rollout loss is worse than the paired `lambda = 0` by
   more than 10%.
3. Reject if action sensitivity falls more than 20% below the paired
   `lambda = 0`.
4. Among survivors, take the lowest false-valley rate; ties go to the smaller
   `lambda`.

Orders 64-127 open only after `lambda`, the step count and the checkpoint rule
are committed.

**On the smoke result.** At `lambda = 0.1` both the AS term and the base loss
fell relative to the paired `lambda = 0`. That shows the gradient reaches the
predictor and moves it in the intended direction; over 20 steps it is not
evidence that the method works, and is not recorded as such.

### H. What a Gate-1 failure means, locked before the dev eval is read

If no `lambda` in the committed grid lowers angular curvature against its paired
`lambda = 0` on the fixed manifest, the conclusion is stated as:

> Cosine AS, at this formulation and this scale, does not have the authority to
> control angular curvature in this model.

It is **not** licensed to extend the grid to `lambda in {1, 3, 10}` and continue
calling the result the same preregistered sweep. Extending a grid because its
committed range failed is hyperparameter selection on the outcome, and it
converts a preregistered test into an exploratory one without saying so.

The admissible response is a **new development amendment on orders 0-63**,
declared as exploratory, investigating why the gradient is too weak: relative
gradient scale of the AS term against the base loss, loss normalisation, or a
different weighting. Orders 64-127 stay closed throughout, so the held-out
evaluation remains uncontaminated no matter how many development iterations
this takes.

**Reading order for the dev eval, not to be shortcut.** Gate 1 on the fixed
manifest first; only lambdas passing it are examined for base loss and
sensitivity; only survivors of those have their `false_valley` looked at. The
training logs already hint that AS may be too weak, but each of their points
sits at a different sampled sigma and direction, so they are a warning and not
evidence, and they are not consulted for the verdict.

## Thirteenth amendment (2026-08-26): cosine AS closed with a mechanism;
## continuation preregistered for held-out

### Cosine AS fails Gate 1, and the decomposition says why

Dev eval `46598`, 16 arms x 12 fixed snapshots, identical directions and sigmas.
`NO_LAMBDA_SURVIVES_GATES`: angular curvature is *higher* than the paired
`lambda = 0` at every lambda, monotonically (`+0.000125, +0.000110, +0.000765,
+0.001679`), and `lambda = 0.3` also fails the base-loss gate at 1.199.

Decomposed in closed form from the stored quantities (self-consistency residual
`1.5e-9`, so this is algebra, not reconstruction error): the failure is **not**
an objective mismatch. `1 - cos`, the training objective itself, rises with
lambda -- ratios `1.015, 1.013, 1.092, 1.209`. The gradient does not steer its
own target.

Against the original checkpoint the picture is sharper:

| arm | `1 - cos` vs original | `k_angular` vs original |
|---|---|---|
| continuation (`lambda = 0`), 3 seeds | x0.791, x0.751, x0.710 | x0.890, x0.867, x0.843 |
| `lambda = 0.3`, 3 seeds | x1.005, x0.856, x0.862 | x0.996, x0.926, x0.929 |

**Open-loop multi-step continuation alone lowers angular curvature (~13% in
`k_angular`, ~25% in `1 - cos`), 3/3 seeds. Adding the AS term cancels part of
that, monotonically in lambda.** Consistent with the base-loss degradation:
AS damages the multi-step fit, and the multi-step fit is what straightens the
map. Physically coherent -- a better multi-step predictor is a more faithful
one, and the physics is smooth (`alpha ~ 2`, measured).

Recorded as a negative result with a mechanism: cosine AS not only fails to
reduce `1 - cos`, it obstructs the straightening that ordinary multi-step
fitting produces on its own.

`false_valley` fell at `lambda = 0.03` and `0.1` on dev. Gate 1 failed, so under
the locked reading order those numbers are exploratory clues and are not
evidence. They are not used.

### What is and is not established about continuation

Established on dev: **continuation lowers angular curvature.**
NOT established: **continuation lowers false valleys.** Calling continuation
"the real lever" would be an overstatement; orders 64-127 exist to answer
exactly that question and have not been opened.

### PREREGISTRATION for the held-out test, locked before any 64-127 shard exists

- **Intervention**: frozen encoder + `H=5` open-loop continuation, `lambda_AS = 0`,
  final checkpoint at 1000 steps, the **three seeds already trained**. No further
  training and no tuning.
- **Control**: the original checkpoint.
- **Held-out**: orders 64-127, never used to alter training or evaluation.
- **Gate 1, manipulation check**: continuation's angular curvature must be below
  the original's.
- **Gate 2, primary, mechanism**: per snapshot, compute the `false_valley` rate
  for the original and the mean rate across the three continuation seeds on the
  identical triplets; bootstrap clustered on snapshot. **Confirmed** if
  `continuation - original < 0` and the 95% CI excludes zero.
- Only if Gates 1 and 2 both pass is CEM elite quality or planning success run
  or claimed.
- **No seed or checkpoint selection by dev false-valley.** All three seeds enter
  the held-out analysis under the fixed rule.

### Consequence for the paper's framing

If continuation confirms, the method contribution is weak on its own: multi-step
and open-loop training already exist in the literature -- RC-aux uses
multi-horizon open-loop prediction against train/test mismatch, and "Closing the
Train-Test Gap in World Models for Gradient-Based Planning" (2512.09929) targets
the same gap between prediction training and action-sequence optimisation. The
defensible claim is therefore not the training scheme but the mechanism:

> We identify angular action-space curvature as a predictor of false local
> minima, confirm that relationship on a held-out sample, and show that ordinary
> multi-step world-model training mitigates the failure by straightening the
> planner-facing action map.

That moves the paper from a method contribution to a mechanism/diagnostic
contribution with a simple intervention, which is what the evidence supports.

## Fourteenth amendment (2026-08-26): held-out result, orders 64-127

The preregistration in the thirteenth amendment was locked in commit `972852d`
before any shard under `outputs/heldout/` existed. Job `46650` (4 arms x 4
chunks, orders 64-127) produced 4096 valid records, 1744 of which survive the
`ordinal_physical_cost_spread >= 1e-4` filter across 47 snapshots shared by all
four arms. Both gates pass.

**Verdict: `CONFIRMED_CONTINUATION_LOWERS_FALSE_VALLEYS`.**

### Numerical correction applied before the result was accepted

The first run of `heldout_test.py` reported `GATE1_FAILED`. That was not a
scientific result. `model_angular_fraction = angular / (radial + angular)` is
`0/0` when the total second-difference energy falls to or below `EPS = 1e-12`,
i.e. where the map is locally straight to numerical precision; 3 of the 1744
analysed records (1 in each continuation seed, 0 in the original) were
undefined, and a single one propagated NaN through a per-snapshot median and
failed the gate arithmetically.

The metric was given a well-defined zero-curvature treatment rather than a
patch. Angular curvature per record is defined directly as

    k_angular = sqrt(angular_energy) / (span + EPS)

which equals `k_model_self * sqrt(model_angular_fraction)` wherever the fraction
is defined and does not require it. Because `angular_energy <= ||D2||^2 <= EPS`
in the degenerate case, the zero-curvature limit is `k_angular = 0`. That is the
locked default (`--degenerate-policy zero`). Two sensitivity runs were made: the
worst-case upper bound (`upper`, all the vanishing energy attributed to the
angular part) and the earlier drop behaviour (`drop`). **All three return
bit-identical gate numbers**, because each affected snapshot contributes one
degenerate record out of roughly nine to a median. Results are in
`outputs/heldout/heldout_result_{zero,upper,drop}.json`; the invalid first run
is kept at `heldout_result_v1_drop_policy.json`.

### Gate 1, manipulation check: PASS

Continuation lowers angular curvature, on 33 of 47 snapshots. Three summaries
of the same comparison, which must not be quoted interchangeably:

| summary | value |
|---|---|
| A. difference of medians | `0.021511 -> 0.018675`, `-13.2%` of the median |
| B. median of paired per-snapshot deltas | `-0.000824` (absolute, units of `k_angular`) |
| C. median of paired *relative* deltas | `-5.6%` |

The gate is defined on B, which is `< 0`. A and C differ because the median of
differences is not the difference of medians. The dev-sweep figure of `~13%`
was a ratio of arm-level aggregates, i.e. an A-type summary, so the held-out
A-type value of `-13.2%` reproduces it; the paired C-type summary is smaller at
`-5.6%`. **Held-out effect stated as A reproduces dev; stated as a paired
statistic it is roughly half that.** Quoting `-3.8%` (the B numerator over the
A denominator) mixes the two and is wrong.

### Gate 2, primary: PASS

| | mean `false_valley` |
|---|---:|
| original | `0.1505` |
| continuation, mean of 3 seeds | `0.0585` |

Paired difference `-0.0920`, snapshot-clustered bootstrap 95% CI
`[-0.1518, -0.0375]`, excludes zero. A `61%` reduction in the mean rate.
Per-seed means `0.0608 / 0.0598 / 0.0551` -- the seed spread is far smaller
than the effect, so this is not a continuation lottery.

### The effect is heterogeneous, and the claim is limited accordingly

Marginally the split is `19` lower, `14` tied, `14` higher out of 47, which
understates the effect because `22/47` snapshots already have
`false_valley = 0` at the original and cannot improve. Conditioning on whether
there is a failure to fix:

| group | n | mean paired diff | lower | higher |
|---|---:|---:|---:|---:|
| original `false_valley > 0` | 25 | `-0.2084` | 19 | 3 |
| original `false_valley = 0` | 22 | `+0.0402` | 0 | 11 |

Decomposition of the `-0.0920`: `-0.1108` from the first group, `+0.0188` from
the second. Best improvement `-0.7143`, worst regression `+0.1515`.

**Continuation removes false valleys where they exist (19 of 25, mean `-0.21`)
and introduces small ones in about half the states that had none.** The claim
is a large mean reduction concentrated on failing states, *not* that
continuation improves nearly every state. The regression on clean states is a
real cost and is reported as one.

### Status of the causal chain

On mutually disjoint samples:

1. high angular curvature <-> high false-valley rate -- confirmed, orders 8-63.
2. multi-step continuation -> angular curvature down -- Gate 1, orders 64-127.
3. multi-step continuation -> false valleys down -- Gate 2, orders 64-127.

This is stronger than correlation: a preregistered intervention moved the
predicted mediator and the predicted downstream failure on untouched data.
It is **not** a proven mediation -- that curvature reduction is what *causes*
the false-valley reduction remains unshown, and the heterogeneity above is
consistent with a partly separate pathway.

Gates 1 and 2 having passed, the thirteenth amendment permits the CEM/planning
step. Its outcomes and decision rule must be locked before those numbers are
read. Orders 64-127 are now used; the planner evaluation runs on the same
snapshots but its metrics are preregistered before being computed. If CEM and
planning improve, the chain closes. If they do not, the conclusion is that
false valleys are a local pathology that is not yet shown to be the planning
bottleneck.

## Fifteenth amendment (2026-08-26): CEM-interaction preregistration

Locked before any population at the deployed budget exists and before any arm
scores any candidate. Gates 1 and 2 of the thirteenth amendment having passed
(fourteenth amendment), this is the next link in the chain.

### Why the cached populations are not reused

`physical_search_distillation/outputs/h0/populations` was collected at a reduced
budget -- `N=96`, `K=10`, `T=12` (`physical_search_distillation/PROTOCOL.md:30`)
-- not the deployed cube configuration. Verified in the released code:
`scripts/plan/config/solver/cem.yaml` is `num_samples: 300`, `n_steps: 30`,
`topk: 30`, and `scripts/plan/config/cube.yaml` is `horizon: 5`,
`action_block: 5`. Scoring at `K=30` on a 96-candidate population would be a
31% elite fraction against the deployed 10%, i.e. a planner surrogate rather
than the planner. Populations are therefore regenerated at the deployed budget.

### What is regenerated

- Original frozen checkpoint only, once, shared byte-identically by every arm.
  Orders 64-127.
- Deployed configuration: `N=300`, `K=30`, `T=30`, `H=5`, `action_block=5`.
- Cached iterations: `0` and `T-1 = 29`, together with the simulator start
  state, so any arm can be scored on the identical candidate tensors and any
  action sequence can be executed from the identical state.

### The two arenas have different standing and are not pooled

**Primary arena: iteration 0.** Candidates are drawn from the initial proposal
(`mean = 0`, `std = var_scale = 1`) before any model has refit anything, so the
population is not conditioned on either arm. Verified in `cem.py` that the
callback receives `candidates` sampled *before* the update alongside the
`prev_mean`/`prev_var` they were drawn from; confirmed empirically on the old
cache (`proposal_mean[0]` all zero, `proposal_std[0]` all one, candidate 0
equal to the mean, while the final iterate has `max|mean| = 2.70`). The smoke
re-asserts all three at the deployed budget before any array is submitted.

**Secondary arena: iteration 29, the original model's own final population.**
This deliberately places continuation inside the basin the original CEM already
selected. Passing here is strong evidence. Failing here while iteration 0 passes
does **not** license "continuation does not help CEM"; the only supported
reading is: *continuation improves refitting on an unbiased common proposal, but
not after conditioning on the basin already selected by the original model.*
The arenas are never aggregated into one test, and the arena is never chosen
after seeing the data.

The final population is not made primary, because doing so conditions the
primary on the control arm before scoring begins.

### Primary outcome

Per snapshot, per arm, on the iteration-0 candidate tensor:

1. score all 300 candidates with the arm's model cost;
2. take the 30 lowest-cost elites;
3. form the CEM update `elite_mean = topk_candidates.mean(dim=1)` -- this is the
   deployed operation (`cem.py`: `batch_mean = topk_candidates.mean(dim=1)`,
   `outputs['actions'] = mean`), i.e. the plan that is actually executed, not
   the top-1 candidate;
4. restore the simulator to the recorded start state and execute `elite_mean`;
5. outcome = physical goal distance in metres at the end of execution.

**Contrast**: `continuation - original`, continuation being the mean over the
three already-trained seeds, paired by snapshot, snapshot-clustered bootstrap.
**PASS** if the point estimate is `< 0` and the 95% CI excludes zero.

### Secondaries, computed in both arenas

- physical cost of the top-1 candidate by model cost (top-1 ranking);
- mean physical cost of the 30 elites (selection quality);
- rank correlation between model cost and physical cost over all 300
  candidates (whole-population ordering).

These answer three different questions and none of them substitutes for the
primary: in a non-convex landscape the mean of 30 individually good actions
need not itself be good, which is precisely the property under test.

### Pre-registered risks and scope limits

- Iteration 0 is the **first** CEM update from an isotropic proposal, not
  converged CEM. The claim is scoped to one update on an unbiased population.
- Averaging 30 quasi-random elites shrinks the action magnitude, so absolute
  physical outcomes may be poor for **both** arms and a floor could compress the
  contrast. Absolute per-arm distances are recorded so a floor is diagnosable;
  a null accompanied by both arms at the floor is reported as uninformative,
  not as a refutation.
- The elite mean may leave the action bounds and is clipped exactly as the
  environment clips it; the clipping rate is recorded per arm.

### Guards

- No seed or checkpoint selection by any planner metric. All three continuation
  seeds enter under the fixed rule.
- Every arm scores the identical candidate tensors; a hash of the candidate
  array is recorded per snapshot and checked equal across arms.
- Only if the iteration-0 primary passes is the full per-arm closed-loop CEM
  endpoint run, where each arm generates its own population over 30 iterations.
  That endpoint, not this test, is what may be called a planning improvement.

## Sixteenth amendment (2026-08-26): viability filter, locked before arm scoring

The fifteenth amendment's population regeneration completed (job `46676`, orders
64-127, 64/64 shards, all 64 index-0 pre-refit gates PASS). Before any arm other
than the scorer-validation smoke was scored, the validation run exposed a
measurement-validity problem that the preregistration did not cover.

### The problem

`cube.yaml` sets `terminate_at_goal: True`. On a start state that already
satisfies the success predicate the episode terminates on the first primitive
action, so a 25-action plan never executes and no arm's choice can change the
outcome. On the regenerated populations this affects **16 of 64** snapshots,
where *every* candidate terminates after one action. Such snapshots contribute
an exactly-zero paired difference, diluting the primary and inflating `n`.

### The filter, defined on the pre-action start state

A snapshot enters the analysis **iff its start state is not already successful**
under the environment's own predicate: `cube_env.py:_compute_successes` marks a
cube successful iff `||obj_pos - tar_pos|| <= 0.04`, and
`72_ogb_stage0_candidate_audit.py:cube_distance` computes that same norm. The
filter is therefore

    keep snapshot iff cube_distance(start_state, goal_block_0_pos) > 0.04 m

evaluated on the restored start state, depending only on the state and the goal
and on no model, so it cannot favour an arm.

`max(executed_steps) > 1` over the shared population is **not** the definition,
only a corroborating classification, because it depends additionally on the 300
sampled actions: a state could be unsolved yet have every candidate terminate
early for an unrelated reason. Agreement between the two classifications is
reported. The `spread >= 1 mm` condition is dropped from the definition; on the
old-budget cache it excluded nothing beyond the frozen states and is reported as
corroboration only.

### Snapshot 064 is retained, with the contamination disclosed

Scorer validation necessarily ran a real arm, so the contrast on snapshot 064 is
partially unblinded: `original 0.02681 m` versus `lam0_seed0 0.02906 m`, i.e.
continuation **worse** on that snapshot. Snapshot 064 is viable under the filter
above and is **kept in the primary**:

> Snapshot 064 was partially unblinded during scorer validation. No decision
> rule was changed as a consequence; therefore it remained in the confirmatory
> analysis.

Removing an observed unfavourable point would be indefensible regardless of the
purity rationale. The one contrast seen runs *against* the hypothesis, so
retaining it makes the test more conservative, not less. A sensitivity analysis
`primary excluding snapshot 064` is reported alongside; if the conclusions agree
the contamination is immaterial.

### Unchanged

Primary outcome, test statistic, `K = 30`, both arenas, and the arena roles are
exactly as locked in the fifteenth amendment. Only the analysis population is
narrowed, by an arm-independent validity rule fixed before scoring.

## Seventeenth amendment (2026-08-26): CEM-interaction result -- NOT CONFIRMED

Jobs `46676` (population regeneration, 64/64, all index-0 gates PASS), `46755`
(viability), `46756` (arm scoring, 64/64). All four arms scored byte-identical
candidate tensors; the per-arena hash equality check passed on every snapshot.

**Verdict: `NOT_CONFIRMED`. The primary is a null.**

### Viability: the two classifications disagree, and the locked one is stricter

| filter | viable | excluded |
|---|---:|---:|
| start-state predicate (locked, sixteenth amendment) | **39** | **25** |
| `max(executed_steps) > 1` (corroboration only) | 48 | 16 |

Nine snapshots -- `64, 70, 72, 89, 96, 100, 101, 114, 121` -- are kept by the
executed-steps classification but excluded by the locked predicate, and the
disagreement is one-sided. Mechanism: those start states are *already*
successful, but the first action knocks the cube out of the goal region, so the
episode does not terminate and more than one step executes. The executed-steps
criterion therefore under-excludes exactly the already-solved states. Choosing
the start-state predicate, as locked, was the correct call and it changed `n`
from 48 to 39.

The filter is not a knob: excluded snapshots have start distance at most
`0.0327 m`, viable ones at least `0.0416 m`, a clean gap around the `0.04`
threshold. Snapshot 064 is excluded by this general rule, so the partial
unblinding recorded in the sixteenth amendment is moot; the sensitivity
excluding it is identical to the primary by construction and is reported as such.

### Primary, arena 0: null

| | mean elite-mean physical distance (m) |
|---|---:|
| original | `0.15848` |
| continuation, mean of 3 seeds | `0.15995` |

Paired difference `+0.00147 m` (continuation nominally *worse*), 95% CI
`[-0.00228, +0.00454]`, does not exclude zero. Lower on 9 of 39, tied on 8.
Per-seed means `0.1613 / 0.1597 / 0.1589`: no seed lottery in either direction.

### This is an informative null, not the pre-registered floor case

The fifteenth amendment reserved the reading "both arms at the floor, therefore
uninformative". That escape does not apply and is not used:

- arena-0 population spread, median `0.157 m` -- ample room to discriminate;
- elite-mean out-of-bounds fraction `0.000` for every arm -- no clipping;
- executed steps median `25` for every arm -- every plan runs to completion.

The CI is tight relative to the effect it would need to detect: `+/- ~4.5 mm`
around a mean of `158 mm`. A continuation benefit of the size the mechanism
would predict is excluded, not merely unresolved.

### Secondary arena and all secondaries: also null

Arena 1 (the original CEM's own final population): `+0.00050 m`, CI
`[-0.00096, +0.00182]`. Secondaries, none excluding zero:

| arena | metric | difference | 95% CI |
|---:|---|---:|---|
| 0 | top-1 physical | `+0.00351` | `[-0.00478, +0.01185]` |
| 0 | mean of elite physical | `+0.00004` | `[-0.00090, +0.00098]` |
| 0 | rank correlation | `+0.00782` | `[-0.00286, +0.01943]` |
| 1 | top-1 physical | `-0.00723` | `[-0.01874, +0.00223]` |
| 1 | mean of elite physical | `+0.00112` | `[-0.00096, +0.00344]` |
| 1 | rank correlation | `+0.00400` | `[-0.03470, +0.04110]` |

Arena-1 rank correlation drops 2 of 39 snapshots where every candidate has an
identical physical distance, so the statistic is undefined; the count is
reported rather than letting a nan propagate through a mean.

### Incidental, arm-independent: the CEM update is itself the lossy step

On arena 0 for the **original** model, averaged over the 39 viable snapshots:

| quantity | mean distance (m) |
|---|---:|
| executed top-30 elite **mean** | `0.1585` |
| mean of the 30 elites' own distances | `0.1279` |
| top-1 candidate | `0.1219` |
| population best | `0.0767` |
| population **median** | `0.1281` |

The action CEM actually executes is worse than the elites individually on 24 of
39 snapshots, and worse than a *typical random candidate* on 20 of 39. This is
the non-convexity the primary was chosen to probe: averaging thirty individually
good action sequences produces an action that is not good. It holds for the
original checkpoint, so it is a property of the deployed CEM update on this
landscape, not an artifact of the contrast and not an explanation of the null in
either arm's favour.

### Consequence for the chain and for the paper

The locked chain stops here. Per the fifteenth amendment, the full per-arm
closed-loop CEM endpoint is run only if this primary passes; it does not, so it
is **not** run and no planning claim is made.

What survives, on mutually disjoint samples:

1. high angular curvature <-> high false-valley rate -- confirmed (orders 8-63);
2. continuation -> angular curvature down -- confirmed (orders 64-127);
3. continuation -> false valleys down -- confirmed (orders 64-127);
4. continuation -> better CEM refit on a shared population -- **refuted** at the
   deployed budget, on an unbiased proposal, with a tight CI.

The supported conclusion is the one the preregistration named in advance:
**false valleys are a local pathology of the cost landscape that is not shown to
be the planning bottleneck.** Link 4 failing after links 1-3 succeeded is a
sharper result than never having tested it -- the mechanism is real and
measurable, it responds to an intervention, and it still does not move the
planner. The incidental finding above suggests where the remaining loss sits:
the elite-averaging update, which no change to the cost landscape's local
curvature can repair.

## Eighteenth amendment (2026-08-26): TD-JEPA novelty gate, preregistered

A gate on whether a new method direction is worth developing, not a claim about
any model. Locked before any TD-JEPA checkpoint exists on this cluster.

### Why this gate exists

The seventeenth amendment closed the geometry branch. Post-hoc measurement on
the spent orders 64-127 then located the remaining loss, and none of it is where
the previous program looked:

| branch | available headroom |
|---|---:|
| false valleys (refuted intervention) | `0 mm` |
| CEM update operator, oracle gate over always-top-1 | `4 mm` |
| candidate **ranking** (model top-1 vs oracle best) | `55 mm` |

The model's cost recovers about `12%` of the achievable gap (Spearman `0.112`),
and blending its ranking toward the true ranking converts smoothly into physical
quality with no floor. That makes ranking the only branch with headroom.

**All of these numbers are exploratory and post-hoc on spent data. They motivate
a hypothesis; none of them may appear in a paper as evidence.**

A curvature-gated aggregation operator was screened and **rejected before being
proposed**: angular curvature does predict the elite-averaging penalty
(Spearman `+0.520`, while the false-valley rate does not, `-0.070`), but the
oracle per-snapshot gate beats always-top-1 by only `4.2 mm`, and a curvature
threshold fit on that same data (`0.1256 m`) is *worse* than always-top-1
(`0.1243 m`). Best-action execution is also prior art -- iCEM's `return_mean`
flag, present in the vendored `planning/solver/icem.py`. Branch closed.

### The question

> Can TD-JEPA's temporally trained LeWM representation already solve same-state
> counterfactual candidate ranking on OGB-Cube, despite never receiving
> same-state counterfactual supervision?

TD-JEPA (arXiv 2607.25337) shares the LeWM backbone and targets the same
diagnosis. Its supervision is same-trajectory temporal order as positives and
cross-trajectory pairs as negatives. CEM must rank 300 counterfactual action
sequences from **one** start state, which is a different negative distribution.
Whether temporal supervision transfers to it is an empirical question.

### Arms, fixed

1. `lewm_original` + latent L2 -- control.
2. `td_jepa` + latent L2 -- **primary baseline**; the paper's own deployed
   OGB-Cube mode is latent L2 on the temporally trained checkpoint.
3. `td_jepa` + mined temporal cost `d_psi` -- secondary, exploratory only; not
   the paper's Cube configuration.

### Arena

Fresh snapshot manifest. Iteration-0 population, `N = 300`, `H = 5`,
`action_block = 5`, `var_scale = 1`. Every arm scores the identical candidate
tensor, hash-checked equal as in the fifteenth amendment.

Iteration 0 is **provably model-independent here**, not merely shared: with no
actor warm-start the initial mean is zero and the initial std is `var_scale`, so
`candidates = randn(generator) * var_scale` depends only on the RNG seed. Both
facts are asserted by `check_population_index0.py`. No arm runs its own CEM,
which would confound ranking quality with search trajectory.

### Primary and secondary

**Primary**: physical goal distance of each arm's **top-1 candidate by model
cost**, paired by snapshot, snapshot-clustered bootstrap. The hypothesis is
about ranking, so the test reads the ranking directly. The CEM elite-mean is
excluded from this gate: the seventeenth amendment showed it is a large
arm-independent downstream distortion and it would only add noise here.

**Mandatory secondary**: Spearman(model cost, physical cost) over all 300
candidates. Reported always, and explicitly **not** a kill criterion on its own,
because rank correlation can rise while the top of the ranking stays poor.

### Decision statistic and rule, locked

Fraction of oracle headroom recovered, on paired per-snapshot means:

    R = (J_lewm - J_td_jepa) / (J_lewm - J_oracle)

where `J_oracle` is the best candidate in the shared population. Bootstrap CI on
both `R` and on `J_lewm - J_td_jepa`.

| outcome | verdict | consequence |
|---|---|---|
| `R >= 0.50` and CI on `J_lewm - J_td_jepa` excludes 0 | `TD_JEPA_CLOSES_RANKING_GAP` | Counterfactual Action-Ranking is at high risk of being incremental; pivot or rethink. |
| `R < 0.25` | `RANKING_GAP_OPEN` | The gap survives temporal supervision; the method is worth developing. |
| `0.25 <= R < 0.50` | `GRAY_ZONE` | Method may proceed but must later beat TD-JEPA head to head. |

`w ~ 0.6` from the exploratory blend is **not** used as a threshold; `w` has no
fixed physical meaning outside that simulation.

### Reproduction-fidelity gate, mandatory, before the comparison

No public TD-JEPA checkpoint was found: the README points to LeWM's Hugging Face
release, and neither the cluster nor the HF cache holds one. The baseline must
therefore be reproduction-trained, which creates the single largest threat to
this gate -- **an undertrained baseline would manufacture a favourable gap for
us**.

Guards, locked:

- Train per the paper protocol: `--config-name=ogb_train data=ogb
  variant=td_jepa`, 10 epochs, `ogbench/cube_single_expert.h5`.
- Train **three seeds**, not the single published seed.
- Admit a checkpoint to the arena only if the authors' own eval path
  approximately reproduces their reported OGB-Cube gain over LeWM. If no seed
  reproduces it, the gate is **inconclusive** and no novelty claim is made in
  either direction.
- Among reproducing seeds, carry the **best** TD-JEPA seed by the authors' own
  metric into the arena. Selecting the baseline favourably to itself makes any
  gap we report conservative.

### Data discipline

Orders 64-127 are spent. They may be used for triage only, and **no number
derived from them enters the paper**. Any claim requires a fresh manifest
generated before scoring.

### Nineteenth amendment (2026-08-26): reproduction-fidelity criterion made numeric

The eighteenth amendment said "approximately reproduces", which would let us
decide after seeing three seeds which one counts. Replaced with a fixed number,
locked before the repository is cloned.

**Official metric**: OGB-Cube **success rate (%)** from the authors' own
`eval.py --config-name=cube`. Their reported result is `+14.2` points over LeWM;
LeWM's published OGB-Cube success rate is about `74%`.

**Control**: the released `quentinll/lewm-cube` checkpoint evaluated through the
**same** official path, on this cluster. The criterion is expressed against our
own control rather than against their absolute number, so any environment or
version difference on this cluster cancels.

**Why not an absolute tolerance.** `cube.yaml` sets `num_eval: 50`. At
`p ~ 0.74` the binomial standard deviation of a 50-episode gain is `7.7 pp`, so
a `+/-5 pp` absolute window sits *inside* the noise and would fail by chance
rather than by fidelity. Eval noise is instead reduced by pooling.

**Locked criterion.**

- Train three seeds with the official recipe: `--config-name=ogb_train data=ogb
  variant=td_jepa`, 10 epochs, `ogbench/cube_single_expert.h5`.
- Evaluate every checkpoint, and the LeWM control, on **three eval seeds pooled
  (150 episodes)**. Pooled `SD` of the gain is then `4.4 pp`.
- **PASS** iff at least one of the three training seeds reaches a pooled gain of
  **`>= 11.4` percentage points** over our LeWM control, i.e. `>= 80%` of the
  reported `14.2`.
- If `0/3` pass: **`TD_JEPA_REPRODUCTION_INCONCLUSIVE`**. Stop. No novelty claim
  is made in either direction, and a weak TD-JEPA is never used to argue that a
  ranking gap exists.

Power of this rule if the reported effect is real: `0.74` per seed, `0.98` for
at least one of three.

**Selection**: among passing seeds, carry forward the highest **official success
rate**. *Seed selection uses only the authors' reproduction metric, never our
same-state ranking metric.* No quantity from the Phase-1 arena may influence
which checkpoint enters it.

### Two different measurements, not to be conflated

The fidelity gate reads **closed-loop success rate** under the authors' full
receding-horizon planning. The Phase-1 arena reads **open-loop top-1 physical
distance** on a shared one-shot population. A method can move one without moving
the other -- that is precisely the gate's question -- so neither number may be
quoted as evidence about the other.

### Phase order, locked

**Phase 0, fidelity only.** Clone the repository at a pinned commit; record the
commit hash, the fully resolved config, the dataset path and hash, and the
checkpoint metadata. Train three seeds on the official recipe. Evaluate through
the authors' own eval path, never our scorer. Apply the rule above.

**Phase 1, ranking arena.** Opened only on a Phase-0 PASS. The fresh ranking
manifest is not generated before then. `TD-JEPA + d_psi` stays secondary and
exploratory; if it happens to be stronger that is worth knowing, but it may not
redefine the baseline after the fact.

### Nineteenth amendment, addendum (2026-08-26): protocol read from the authors' own locked artifacts

The repository was cloned at pinned commit
`b4c17ca4649c9bf47272fa66c38da7a684f2a020` (2026-07-29) to
`diagnosis/external/td-jepa`. It ships `results/paper_locked/`, which specifies
their evaluation far more precisely than the abstract does. The criterion is
tightened to match it **before any training**.

Their locked Cube cost matrix, training seed 3072, epoch 10:

| cost | mean success | std |
|---|---:|---:|
| latent L2 | **82.2%** | 2.9 |
| blend `alpha=0.10` | 81.2% | 3.0 |
| `d_psi` | 77.0% | 1.7 |

`d_psi` is worse than latent L2 by `5.2` points, paired `p = 0.0015`. This
independently confirms the arm assignment of the eighteenth amendment: latent L2
is the primary baseline on Cube and `d_psi` is secondary. Note also that
`config/train/ogb_train.yaml` sets `planning_cost_mse_blend: 0.10`, which is the
in-training monitor, not the cost of the reported result; every arm's cost is
set explicitly at evaluation.

**Their evaluation protocol, adopted verbatim:**

- Ten locked plan seeds `[20260714, 7, 11, 13, 17, 19, 23, 29, 31, 37]`.
- A locked 50-episode validation manifest
  (`eval_manifests/ogbench_cube_single_expert_val_seed20260714_n50.json`), so
  the episode set is **identical** across plan seeds and across arms. Their
  across-seed spread is planner stochasticity on a fixed episode set, not
  episode sampling, and the comparison against our control is therefore paired
  on episodes.
- Solver for Cube: `num_samples 300`, `n_steps 10`, `topk 30` -- "CEM-10", not
  the `stable_worldmodel` default of 30 iterations.
- Training seeds `3072, 3073, 3074`, matching their own multi-seed set.

**Consequences for the locked criterion.** The eighteenth/nineteenth amendment's
"three eval seeds pooled" is replaced by their ten locked plan seeds, and the
three training seeds are theirs rather than ours. Both changes make the test
more faithful and lower its noise: the standard error of a per-checkpoint mean
is about `2.9/sqrt(10) = 0.9` points, and of the paired gain about `1.2` points,
against which the locked `>= 11.4` point bar has power near `0.99` if the
reported `14.2` is real. **The bar itself is unchanged.**

The control is still our own run of the released `quentinll/lewm-cube` through
this same path; their LeWM number is not assumed.

### Nineteenth amendment, second addendum (2026-08-26): which LeWM is the control

The repository ships `variant=lewm` and `variant=rc_aux` alongside `td_jepa`,
all trainable with the same recipe. That creates two distinct LeWM references,
and using the wrong one would invalidate the gate.

- **Fidelity-gate control: `variant=lewm`, trained by us, same recipe and same
  three seeds.** Their reported `+14.2` is a within-codebase delta against their
  own LeWM baseline, so only a within-codebase LeWM reproduces it. Comparing
  against the released checkpoint instead would fold any difference between the
  release and their baseline into the measured gain, in an unknown direction.
- **Phase-1 arena control: the released `quentinll/lewm-cube`.** That is the
  deployed model this whole program has studied, and every earlier amendment's
  numbers refer to it.

Both are evaluated and reported; neither substitutes for the other. Cost is six
training runs rather than three, which is the price of a gate that can actually
fail for the right reason.

Should our within-codebase LeWM and the released checkpoint differ materially
under the identical eval path, that difference is recorded as a finding about
reproduction, and the Phase-1 arena still uses the release.
