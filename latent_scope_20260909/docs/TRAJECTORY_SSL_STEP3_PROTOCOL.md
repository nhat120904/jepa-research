# Step 3 protocol: matched self-supervised trajectory-target pilot

Locked: 2026-09-20, before profile or training. This is a one-seed development pilot.

## Question and exclusions

Does a fixed algebraic trajectory target provide a prediction/composition advantage
over an ordered future-frame target and freely learned summary tokens, given identical
causal observation history, future actions, data and predictor budget?

This does **not** train native success, contact, coverage, reward, value or language-goal
heads. It does not fine-tune/evaluate GR00T, establish counterfactual candidate headroom,
or measure closed-loop control. It cannot establish that trajectory JEPA beats a memory
VLA. The old test split, old grounded checkpoints and old label files are not loaded.

## Data and shared observation representation

- Cached 20 Hz DINOv2-S patch grids, proprio and logged actions from released Scrub demos.
- Manifest `train` episodes for optimization. Manifest `val` episodes for development
  evaluation. Manifest `test` entries are removed before payload loading.
- First train a shared per-frame autoencoder without task labels: spatial camera/patch
  attention plus standardized proprio to a 16-dimensional latent; decode normalized
  DINO patch features and proprio. Freeze this encoder for every trajectory arm.
- History: 24 observations at stride four, through current `o_t`, each paired with its
  incoming executed action. Candidate action `a_t` is excluded from history and supplied
  only to the predictor. Windows begin late enough that no reset padding is needed.

The 16-dimensional bottleneck is a pilot choice, not an information ceiling. Every arm
uses the same bottleneck, so this test attributes differences among targets conditional
on it. Failure of all arms triggers an observation-target diagnosis, not a global method
verdict.

## Targets and matched predictors

Train lengths are 32 and 64 transitions. All arms use the same width/depth action
Transformer, history GRU, endpoint target, optimizer, batches and step count.

1. `ordered_frames`: 20 uniformly sampled future frame latents, flattened. This is a
   strong direct ordered-sequence target, not a claimed reproduction of autoregressive
   DINO-WM.
2. `learned_summary`: four 64-dimensional tokens from a trajectory autoencoder trained
   only to reconstruct the same 20 future latents. Freeze the target encoder/decoder
   before action-conditioned prediction.
3. `signature_d2`: raw degree-2 signature of the shared integrated feature path
   `(physical time, integral frame latent)`, plus endpoint. The 17 path channels yield
   17 + 289 = 306 nonconstant coefficients. Fit per-coordinate prediction-loss
   normalization on train windows only; invert it before composition. Composition is
   the fixed truncated Chen product. A self-supervised decoder maps true/predicted
   signatures plus endpoint and duration to the same 20 future latents.

Each arm has its own learned memory update `U(h, representation, endpoint)`, trained to
match the history encoder at the observed segment end. At deployment-style split
prediction, the right segment receives only the predicted update/endpoint and remaining
actions. This update is part of each matched arm; it is not called the composer.

At 64 steps, additionally train a 32+32 rolled path. No target future observations enter
the rolled right prediction. Summary and frame baselines decode both predicted parts and
resample their ordered concatenation. Signature reports both this same sequential decode
and a whole-path decode after fixed Chen composition.

## Evaluation

No checkpoint selection on evaluation metrics in this one-seed pilot. Report:

- frame-autoencoder train/validation reconstruction;
- observed-target decoding ceiling for lengths 32/64 and unseen 128;
- direct action-conditioned prediction at 32/64 and unseen 128;
- split prediction at 64=32+32, 64=24+40 (unseen lengths/split), and
  128=64+64 (unseen total);
- common MSE on the same 20 frozen frame latents, endpoint MSE, normalized target MSE;
- signature sequential versus fixed-composed decoding;
- parameter counts and run time.

All windows from one episode remain dependent. This pilot reports aggregate development
means, not confidence intervals or a confirmatory statistical claim. A later positive
claim requires fresh episodes and multiple training seeds.

## Decision rule

`SIGNATURE_MECHANISM_LEAD` only if all hold:

1. true signature targets decode unseen length 128 no worse than 1.20 times the learned
   summary ceiling;
2. fixed-composed signature common MSE is no worse than the best matched baseline on both
   64=24+40 and 128=64+64;
3. it improves at least one of those two by 10% or more;
4. fixed composition is no worse than decoding its own two segments sequentially by more
   than 5% on either split.

If the target ceiling fails, stop this lift/bottleneck before more predictor training.
If the ceiling passes but prediction fails, localize the bottleneck to action-conditioned
prediction/update. If signature prediction is accurate but matched controls select the
same future latents equally well, there is no representation-level reason yet to carry
the algebra into planning. Passing this rule only earns a candidate-ranking experiment;
it does not establish control improvement or paper-level novelty.

## Budget and execution

- Profile first: 12 train + 8 val episodes, 100–200 steps per stage, GPU hard cap 30 min.
- Full pilot only after profile passes: all non-test episodes, one GPU, hard cap 3 h;
  expected substantially below the cap. Internal wall-clock guard saves partial result.
- Source/config are snapshotted per job. Never overwrite old runs. Record every job in
  `JOB_LEDGER.md`. No simulator/rendering/model download occurs; only cached tensors load.
