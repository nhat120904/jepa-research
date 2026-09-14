# Qualification result verified 2026-09-14

The requested carrier repair and bounded pilot already ran in the shared workspace.
The earlier conversation status referencing only 52226 was stale. No duplicate GPU
job was submitted during this verification.

Both squeue and sacct were checked: 52418 and 52419 are absent from the queue and
COMPLETED with exit 0:0. CPU proof 52418 took 00:02:22; GPU pilot 52419 took 04:57:16.

The compiled-model/full-MjData restore replaces the nonidentity XML round trip.
See RESTORE_REPAIR_20260913.md for saved-action validation and remaining scope limits.
Prefixes are chosen at observed partial contact without inspecting future success.

Pilot artifact:
`/mnt/data/nhatnc129/jepa/latent_scope_baseline/outputs/qualification_52419/qualification_result.json`

- Verdict: QUALIFICATION_PILOT_COMPLETE.
- Eight completed Scrub prefixes, eight candidates per prefix.
- Two prefixes were recoverable against predeclared candidate 0 in screening:
  empirical single-seed oracle availability gain was 25 percentage points.
- Four complete independent-seed confirmation pairs had zero gain. For attempt 4,
  selected candidate 7 and baseline both succeeded at seed 143040 and both failed
  at 143041. For attempt 5, selected candidate 3 and baseline both failed at seeds
  143050 and 143051.

This establishes exploratory candidate availability on a conditional partial-contact
panel, but not a reproducible selection advantage. The 25-point screen result must
always be reported alongside the zero confirmation gain. Four confirmation pairs
cannot establish a tight null or attribute the result uniquely to continuation noise.
It does not establish a composition benefit or authorize Stage-C training.

The requested repair and pilot are complete. The next scientific question is whether
candidate ranking persists across continuation seeds; any further experiment should
lock a multi-seed evaluation and budget before collecting more candidate branches.
