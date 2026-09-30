# CTA Round 6: training for the 16-candidate bank (2026-09-27)

Round 5 (JOB_LEDGER): with the deployment bank = 8 policy samples + their 8 perturbed copies, the oracle GEOM16 rose to
43/50 (P0 31) but every learned scorer choosing among 16 fell below P0 (CTA4 19, DIRECT4 7). Verified cause in the
training code: Round 4 ranked the policy bank and the perturbed bank of the same context as two separate 8-banks
(`scripts/cta_round4.py`, K = 8), so no loss compared a policy candidate with a perturbed one — the comparison the
16-bank arm makes at every decision. How much of the failure this explains is what Round 6 measures.

Changes (one recipe, no tuning on closed-loop roots):
1. Training example = the paired 16-bank (both halves share context and order in the Round-4 cache; asserted).
2. Losses over all 16: rank16 (Round-4 weighted pairwise ranking) + anchor (the same, restricted to pairs with
   candidate 0, the default the arm must beat).
3. Stage A: the code reader is fine-tuned on source codes of the actual futures (frozen source encoder, cache reused).
   Stage B: CTA world model (Round-4 task recipe with the new reader as teacher) and direct scorer, same banks/losses.
4. Selection per network by its own objective on dev 16-banks = collection roots 32200-32249 (excluded from training).
5. Offline table on dev 16-banks (`offline16.json`): retained gap, Spearman, and the rate of choices clearly worse
   than the default (> 0.005), for the parent reader on actual codes, CODE6, CTA4, CTA6, DIRECT4, DIRECT6.
6. Closed loop: 200 dev roots 2100-2299, arms P0, GEOM16, CTA6, CTA6@8, DIRECT6, CODE6 (privileged: reader on the
   actual future's code). Logged scorers CTA6, DIRECT6, CODE6, CTA4, FULL on every visited state.

Not changed: source encoder, FULL reader (retraining it needs re-encoding end frames; later if needed). Known data
limit: collection episodes only ever advanced through policy candidates, so states reached after repeated perturbed
choices are not covered; if errors concentrate there, the next step is data from the new planner's rollouts on
train roots. Old policy-only 8-banks (source 0) are not used in Round 6.

## Result of the Round-6 training (55304) and the next test (2026-09-27)

Offline on the dev 16-banks (`offline16.json`), Round 6 changed little: retained gap over 16 — parent reader on actual
codes .77, CODE6 .79, CTA4 .80, CTA6 .78, DIRECT4 .54; choices clearly worse than the default — CTA4 2.4%, CTA6 2.7%,
DIRECT4 6.5%. CTA4 already ranks the 16-bank well on collection states, so within-bank-only training was not the main
cause. The Round-6 closed loop (55305) and its aggregate/demo (55306/55307) were cancelled before producing results.

What the Round-5 logs show instead (states visited in closed loop, roots 2100-2149): choosing among 16 turns small
mistakes into large ones. With CTA4, losses vs the default have p90 .023 and max .08 (≈ 12 and 42 px) over 16,
against p90 .002 over the policy half; the scorer error rate is also higher on the states CTA4 itself visits (11%)
than on P0's (6%). Next test = selection that makes larger departures from the policy's own samples pay for
themselves: score_k − λ·σ_k (σ_k = 0 for policy samples, the perturbation scale for perturbed copies). Offline on P0
states, λ = .08 cuts losses > .01 from 3.8% to 1.2% while keeping 7.1 of 8.0 gain (policy half only: 1.1). Closed-loop
test on 50 dev roots (2100-2149), arms P0, CTA4~0.04, CTA4~0.08, CTA4~0.16, R4 networks, same bank; λ grid fixed now.
This is a dev-root selection among three values; any winner must be confirmed on sealed roots.

## Scope note (2026-09-28, user review)
Deploying perturbed copies (Round 5, Round-6 closed loop, penalty test) departs from the method design, which ranks
only the frozen policy's own samples so that the planner cannot pick implausible actions that exploit model error.
These runs are an out-of-design experiment; their failure is evidence for that design choice, not a test of CTA.
Perturbed banks remain legitimate as TRAINING data (Round 4). Penalty test 55330 (50 roots): P0 31, CTA4~0.04 25,
~0.08 21, ~0.16 28. In-design PushT evidence so far: Round 4, K = 8 policy samples, 100 roots: CTA4 68, P0 64,
DIRECT4 60 (n.s.). Priority moved to OGBench (user decision 09-28).
