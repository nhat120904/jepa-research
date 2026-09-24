# Gate C2-confirm result (jobs 54268, 54269, 54270)

Protocol: `GATE_C2_CONFIRM_PROTOCOL.md` (pinned before any confirmation data).
Fresh roots 1300–1499, n = 200 closed-loop episodes per arm. Reader r0 = `c2_train_53938` (frozen),
reader r1 = `c2_train_54268` (`--seed 1`, held-out Spearman 0.856, so r1 training OK).

## Verdict

| Rule | Result | Verdict |
|---|---|---|
| Primary: PROG8_r0 − P0, CI lower bound > 0 | +4.5 pp, CI [−4.0, +13.0], McNemar 43 vs 34, p = 0.36 | **NOT_CONFIRMED** |
| Seed robustness: PROG8_r1 − P0 | +2.5 pp, CI [−5.5, +10.5], McNemar 37 vs 32, p = 0.63 | NOT_ROBUST |

| Arm | Success (n = 200) |
|---|---|
| P0 | 0.650 |
| PROG8_r0 | 0.695 |
| PROG8_r1 | 0.675 |

Pooled C2 + confirm for r0 (300 roots, labelled in the protocol as pooled, not a test): +7.7 pp [+1.0, +14.3].

## Interpretation

- Per rule 4, the +14 pp of C2 (roots 1200–1299) is treated as **not established**. The fresh-root
  estimate is about a third of it; the C2 point estimate was most likely an optimistic draw.
- Both readers point in the same direction (+2.5 to +4.5 pp), which is compatible with a small true effect
  that 200 roots cannot resolve, and also with no effect. This protocol cannot distinguish the two.
- Selection among the released PushT policy's proposals by a label-free hindsight progress reader on ACTUAL
  futures therefore does not qualify as a scorer for this arena under the contract's gate C criterion.
  Any world model that predicts that reader's input inherits this ceiling.
- No retuning of reader, goals or K on these roots. A new reader requires a new protocol and fresh roots.

## Compute

54268 29 min, 54269 4 × ~39 min, 54270 CPU. Programme GPU usage ≈ 5.6 + 0.5 + 2.6 ≈ 8.7 GPU-h, which
exceeds the 8 GPU-h cap by ≈ 0.7 h (the per-root cost was higher than the 1.8 GPU-h estimate).

## Implementation-consistency check (CPU job 54422, after the verdict)

Question: is NOT_CONFIRMED an artifact of a code or run difference between C2 and C2-confirm?

| Check | C2 PROG8 | Confirm PROG8_r0 |
|---|---|---|
| reader sha256 | 751a342d… | 751a342d… (identical) |
| P0 success / mean steps | 0.65 / 219.5 | 0.65 / 219.9 |
| progress score mean / sd | −4.480 / 0.527 | −4.478 / 0.522 |
| share of decisions choosing candidate 0 | 0.114 | 0.128 |
| within-bank Spearman(progress, 8-step coverage) | 0.049 | 0.036 |
| mean 8-step coverage gain of the chosen candidate | +0.0005 | +0.0010 (PHYS8: +0.0068) |

No sign of an implementation difference: same reader, same score distribution, same selection behaviour.
The two PROG8 − P0 estimates are not statistically different from each other: +14.0 vs +4.5, difference +9.5 pp, CI [−4, +23].
Across the four 50-root blocks of the confirm run, PROG8 − P0 ranged from −12 to +18 pp, so 50–100 roots carry a lot of noise.

Corrected reading: NOT_CONFIRMED means the effect is not established at the pre-registered bar. It does **not** mean the effect is zero.
The most plausible size is a small positive effect (pooled over 300 roots +7.7 pp [+1, +14]; not a test, and it includes the
data that motivated the confirmation). Per decision, the reader's choice takes about 10% of the 8-step coverage gain that PHYS8 takes.
Either way, this is well short of the pre-registered goal of retaining most of the oracle headroom.
