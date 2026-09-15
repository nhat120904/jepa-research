# Scrub demonstration data check — job 52600

Date: 2026-09-14. CPU only (4 CPU, 16 GB), COMPLETED in 00:15:00. No policy, rendering
or training. Purpose: decide whether released ScrubCuttingBoard demonstrations can
support an offline test of in-segment accumulated-effect prediction and composition,
replacing GR00T end-of-episode success as the first test.

Method: load each recorded simulator state in the RoboCasa 1.0.1 runtime, call the
native `update_state` (same call as `Kitchen._post_action`) and `_check_success`.
50 episodes: 0–4 plus 45 sampled with seed 20260914 from 5–503.

Result: `/mnt/data/nhatnc129/jepa/latent_scope_baseline/outputs/scrub_data_check_52600/scrub_data_check_result.json`
(per-frame arrays in `scrub_monitor_arrays.npz`). Code and config hashes are in `code/`.

## Verdict: usable with explicit caveats

| Question | Result |
|---|---|
| Full data available? | Yes. 504 episodes with states, parquet, XML, meta; three 256×256 videos each. **0.85 GB total**, 0.28 GB without video. 45 episodes of non-video files downloaded in 320 s. |
| Videos decodable and aligned? | Episode 0: all three cameras decode, frame count = parquet rows (423). One episode only. |
| Recorded state loads exactly? | Yes, max error 0.0 on all 50 episodes and all frames. |
| Playback monitor reproduces recorded success? | **44/50.** In those 44, first success frame matches the recorded reward frame exactly (offset 0). The other 6 end with 3–4 accepted contacts instead of ≥5, with sweep and release satisfied: playback **undercounts** contacts. Cause not diagnosed (grasp/contact evaluated at the stored state rather than inside `mj_step`, or 0.5.1 vs 1.0.1 task code). |
| Offline filter reimplementation = native counter? | 50/50 exact, so window statistics below use the native rule. |

The exact-frame agreement mainly validates the release moment, since count and sweep are
usually satisfied earlier. It does not validate every intermediate count.

## Event density (accepted distinct contacts)

Episodes: median 445 frames at 20 Hz; median 7 accepted events (range 3–23). The first event
occurs at a median 36% of the episode; first-to-fifth event spans median 91 frames
(range 34–328), first-to-last 173.5 frames. Median 4.2 grasped-contact frames per
accepted event.

Windows with stride 4, restricted to the scrub phase (from L frames before the first event
to the last event):

| Length L | ≥1 event | ≥2 events | History changes increment* | Midpoint split double-counts** |
|---:|---:|---:|---:|---:|
| 8 | 30% | 2% | 35% | n/a |
| 16 | 50% | 10% | 26% | 21% |
| 32 | 72% | 30% | 21% | 16% |
| 64 | 89% | 54% | 19% | 13% |
| 128 | 98% | 76% | 15% | 12% |

\* Among windows with events: the count from an empty history differs from the native
increment given the real history.
\** Among windows with events: left-half count plus right-half count (each from an empty
history) exceeds the whole-window count.

## Label sensitivity

Re-running the native filter on every second frame changes the final count in **39/50**
episodes, usually lower (for example 23→16, 8→5). The accepted count depends strongly on
temporal sampling of transient contact frames.

## Consequences for the mechanism-test design

1. **Segment lengths must be 32–128 native steps.** At 8 steps only 2% of scrub-phase
   windows contain two events. The Stage-C lengths 2/4/8 cannot test accumulation or
   composition on this task.
2. **At least 12–21% of midpoint splits are non-additive.** The statistic only counts
   `left + right > whole`; it does not count `left + right < whole`, which the greedy
   ordered filter also allows (a history point can reject an early right-segment contact
   and thereby admit a later one). So it is a lower bound, and the remaining splits are
   **not** shown to be additive. Exact merging generally needs the left segment's accepted
   positions plus the right segment's ordered contact sequence, not counts or right-only
   accepted points. Correction pending: report `<`, `=`, `>` separately, with episode-level
   counts. Evaluation must stratify additive and non-additive splits.
3. **Labels must be computed at the native 20 Hz cadence**, whatever cadence the model uses.
   Define the target as this playback monitor and report its 44/50 agreement with recorded
   success. Do not call it the exact native counter.
4. **Demonstrations are success-only human data.** They support prediction and composition
   under held-out partitions and lengths, not counterfactual action ranking.
5. **Not yet tested:** whether accepted contacts are inferable from the permitted images.
   That needs features, which is the next step.

No model, planning result or composition benefit is claimed.
