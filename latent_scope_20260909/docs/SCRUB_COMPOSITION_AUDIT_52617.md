# Scrub label and composition audit — job 52617

Date: 2026-09-15. CPU only on a compute node (8 CPU, 64 GB), COMPLETED in 00:49:02, exit 0.
No policy, rendering or training. Self-test on synthetic trajectories passed before any
download.

Result: `/mnt/data/nhatnc129/jepa/latent_scope_baseline/outputs/scrub_composition_audit_52617/composition_audit_result.json`.
Per-episode labels: `.../labels/episode_XXXXXX.npz` (count, contact, grasped, far, success,
sponge_xy, board_pos, board_quat, recorded_reward). Data mirror:
`/mnt/data/nhatnc129/jepa/latent_scope_data/hf_mirror`.

## Data and labels (all 504 episodes)

| Check | Result |
|---|---|
| Download | All meta, parquet, extras and videos (3536 files), ~30 min |
| Videos | 504/504 episodes: all three cameras decode, frame count = parquet rows |
| State load | Max error 0.0 |
| Playback success vs recorded | **442/504 agree (87.7%)**, always on the exact recorded frame. All 62 disagreements are recorded successes that playback never marks successful (too few accepted contacts). None were excluded. |
| Offline filter = native counter | 504/504 |

## Composition of segment summaries

Scrub-phase windows containing at least one grasped contact. Numbers below are label-agree
episodes, overlapping windows (stride 4), 2-part midpoint split, unless stated. The full
table (all splits, non-overlapping blocks, disagree group) is in the JSON.

| L | Native count: sum of parts ≠ whole | Native: re-filter per-part accepted points ≠ whole | Coverage area: sum ≠ whole | Coverage: union of part masks ≠ whole | History changes increment (native / coverage) |
|---:|---:|---:|---:|---:|---:|
| 16 | 22.3% | 4.6% | 29.7% | 0% | 26.1% / 33.9% |
| 32 | 16.7% | 3.4% | 22.1% | 0% | 20.9% / 27.0% |
| 64 | 13.2% | 2.7% | 17.3% | 0% | 17.3% / 21.8% |
| 128 | 12.1% | 2.5% | 15.5% | 0% | 13.9% / 16.9% |

- **Three-part splits roughly double the rate:** native sum wrong in 40/30/23/21% of windows
  for L = 16/32/64/128.
- **Direction:** across more than 100k windows, sum-of-parts was never below the whole and
  the empty-history value was never below the increment, for either target. `<` is
  possible in principle for the greedy filter but did not occur. Re-filtering per-part
  accepted points gives the opposite sign (below the whole) in 2–6%: points rejected inside
  a part occasionally matter for the merge.
- **Magnitude when wrong:** about 1 contact for the native count (median episode total 7);
  about 2.3 cm² for coverage (median final 29.5 cm²).
- **Spread:** non-additive windows occur in almost every episode (≥428 of 442 for 2-part
  splits). With non-overlapping blocks, 169–284 distinct episodes contribute at least one
  non-additive L32–L64 block.
- **Disagree group** (62 episodes) has roughly half the non-additivity at long lengths,
  consistent with missed contacts.

## Frame-rate robustness (every second frame)

| Target | Episodes whose final value changes | Median relative change | p90 |
|---|---:|---:|---:|
| Native count | 377/504 | 18% | 40% |
| Coverage area (1 cm discs) | — | 23% | 40% |

**The coverage target is not more robust than the native count.** Both depend on sampling of
transient contacts. The earlier hypothesis that coverage would be cadence-stable is refuted.

## What this means for the method

1. **A real but moderate composition problem exists.** Adding segment counts or areas is
   wrong in 12–30% of two-part merges and 21–46% of three-part merges, by roughly one
   contact. A trivial additive composer is therefore beatable, but the headroom is bounded:
   the error it makes is small in absolute terms.
2. **A set of contact positions is almost a sufficient summary.** Union of disc masks
   composes exactly; re-filtering per-part accepted points is wrong in only 2–6%.
   The natural segment summary is a spatial contact map, not a scalar. It also defines a
   strong non-learned control: predict the contact map, then take the union or re-filter.
   A learned composer must beat that, not only the additive sum.
3. **History matters.** In 14–34% of windows, the new contribution depends on where the
   sponge already scrubbed. A predictor with a three-frame context cannot represent this;
   the model needs a spatial memory of previous contacts (or an equivalent history summary),
   and every baseline needs the same history access.
4. **Targets must be computed at 20 Hz.** A model observing fewer frames loses 18–40% of the
   label signal. That is a ceiling on any cadence-reduced model, not a method failure.
5. **Label noise is about 12% of episodes.** Keep all episodes and report both groups. A
   short diagnosis of the grasp check in playback would reduce this before image probes.

No image readability, prediction benefit or composition benefit is established by this audit.
