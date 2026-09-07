# ORDER-JEPA execution status

Last updated: 2026-09-07 UTC.

## Current verdict

**`STOP_OR_FIX_STAGE_A`: do not train ORDER on either qualified setup.**  The physical
premise passes: swapping two five-control action blocks while preserving the
action multiset frequently changes the outcome in both PushT and Reacher. Neither
the original-paper DINO-WM PushT predictor nor the official LeWM Reacher predictor
shows the hypothesized order-specific decision gap. Stage B training remains closed.

## Official LeWM Reacher qualification

The second Stage-A audit used the official `quentinll/lewm-reacher` checkpoint,
not any `facebookresearch/jepa-wms` checkpoint. It used the released
`stable-worldmodel` Reacher environment and the paper's random-trajectory data
collection distribution, with fresh deterministic trajectories rather than
downloading the 23.8 GB archive. The fixed design contains 200 anchors clustered
in 25 trajectories, four swapped-order pairs per anchor, three context frames,
five-control action blocks, and a five-block prediction horizon.

- Slurm preparation job `50457`, smoke job `50459`, eight-shard full job `50460`,
  and aggregation job `50468` all completed with exit code zero.
- Checkpoint revision: `62adae4b71dc474ddf8f794c476ebfe737a743ca`;
  weights SHA-256: `eb70b1fd5409f8f81875d62f5ee5a20dd220a3128a477de66b5760f475f0f469`.
- Exact reset, pixels and repeated endpoints all had maximum error zero.
- 723/800 pairs passed the fixed physical relevance filters. Mean normalized
  qpos order effect was 2.1874, CI [2.1115, 2.2640].
- True-latent physical order-ranking accuracy was 0.7261, CI [0.6871, 0.7614];
  predicted accuracy was 0.7151, CI [0.6827, 0.7465].
- The order-specific accuracy gap was only 0.0111, CI [-0.0234, 0.0439], far
  below the preregistered 0.10 lower-bound requirement.
- The predicted order vector was already strong: cosine 0.8936 and amplitude
  ratio 1.0115. There is little missing order signal for ORDER to restore.
- A generic dynamics problem is nevertheless present: on the witness-free,
  diagnostic-pair-free ordinary panel, true-latent selection succeeded on every
  anchor while predicted-latent selection succeeded on 0.415, CI [0.350, 0.485],
  with dynamics excess physical regret 1.3007, CI [1.1287, 1.4859]. This does
  not rescue ORDER because the swapped-order diagnostic does not localize that
  generic planning error to action order.

Machine-readable result:
`/mnt/data/nhatnc129/jepa/order_jepa/reacher_stage_a/analysis/summary.json`.

## Completed checks

- Original author source: `gaoyuezhou/dino_wm`, pinned at
  `0a9492fa12044b852ae9e001cc74604b79c8bb0c`.
- Analytic/unit sanity job `50377`: completed; five tests passed, the cost-gap
  and paired-error identities agree to numerical precision, and the constructed
  unicycle error has the expected fourth-order squared-error scaling.
- Four-anchor simulator smoke collection: completed with exact reset, rendered
  pixel, and repeated-endpoint agreement.
- Full simulator collection job `50391`: all eight shards completed with exit
  code zero, producing 200 anchors and 400 swapped-order pairs.
- Episode-clustered physical analysis jobs `50399`/`50400`: completed with exit
  code zero.
- Original-paper checkpoint scoring job `50415`: all four GPU shards completed
  with exit code zero; decision aggregation jobs `50419` and corrected
  witness-free aggregation `50445` completed with exit code zero.

## Full physical qualification result

The qualification set contains 400 pairs from 200 anchors across 21 source
episodes.  Confidence intervals are 95% episode-clustered bootstrap intervals.

| Measurement | Estimate | 95% CI |
|---|---:|---:|
| Any contact in either order | 0.6675 | [0.5995, 0.7267] |
| Normalized object order effect, mean | 0.6162 | [0.4958, 0.7511] |
| Object effect at least 0.25 | 0.4825 | [0.4146, 0.5480] |
| Effect at least 0.25, both orders contact | 0.7513 | [0.6847, 0.8230] |
| Effect at least 0.25, exactly one order contacts | 0.6429 | [0.5167, 0.7632] |

The normalized effect is object-position displacement divided by 20 pixels plus
object-angle displacement divided by `pi/9`, matching the physical success
tolerances.  Among the 133 no-contact pairs the measured object effect is
exactly zero.  Reset physics error, pixel error, and repeated-endpoint error are
all exactly zero over the full collection.  This pattern localizes the signal
to interaction with the object rather than reset noise or pusher-only motion.

Machine-readable result:
`/mnt/data/nhatnc129/jepa/order_jepa/stage_a_original/physical_smoke.json`.

## Original-checkpoint predictor audit

The audit loaded the authors' OSF PushT release through the pinned original
source.  No `facebookresearch/jepa-wms` or third-party retrained checkpoint was
used.

- Checkpoint SHA-256:
  `e909f0cec958fc0b49a79f2e85730ae6b5f83dee61a224705f91883eb3bb74c5`.
- Resolved-config SHA-256:
  `e4d7f21b85c778ab8b83c0bb83bc7625fdfb193a6a0619fef25f4a2afd7f8049`.
- 171/400 pairs passed the pre-registered physical relevance filters.
- True-latent physical order-ranking accuracy: 0.7135, CI [0.6524, 0.7633].
- Predicted-latent physical order-ranking accuracy: 0.7310, CI [0.6645, 0.7950].
- Accuracy gap, true minus predicted: -0.0175, CI [-0.0765, 0.0409].
- Relevant order-vector normalized error: 0.7728; cosine: 0.6210; predicted/true
  amplitude ratio: 0.7339.

The high vector error confirms that the predictor imperfectly represents the
swapped-order difference.  It does not create the required decision deficit:
its physical order ranking is statistically indistinguishable from, and
numerically slightly better than, ranking encoded true outcomes.

## Witness-free candidate selection

The initial aggregate included the labeled feasible witness.  Because that
candidate makes goal coverage tautological, job `50445` recomputed selection on
the other 31 candidates per anchor:

- Ordinary-panel goal coverage: 0.4400, CI [0.3799, 0.4974].
- True-latent selected success: 0.3650, CI [0.2974, 0.4293].
- Predicted-latent selected success: 0.3650, CI [0.2967, 0.4262].
- True-latent physical regret: 0.1885, CI [0.1307, 0.2528].
- Predicted-latent physical regret: 0.2008, CI [0.1502, 0.2570].
- Dynamics excess regret: 0.0123, CI [-0.0395, 0.0663].

Thus both predictor-specific gates fail.  Training ORDER here could plausibly
improve the order-vector metric, but the audit provides no evidence that doing
so would improve physical decisions.  The machine-readable final decision is
`/mnt/data/nhatnc129/jepa/order_jepa/stage_a_original/analysis/summary.json`.
