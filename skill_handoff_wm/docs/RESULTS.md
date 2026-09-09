# Direction A pilot results

Decision date: 2026-09-07. Status: **STOP after A2 on OGBench AntMaze-medium**. Do not open A3 joint-model
training or AntMaze-large transfer under this protocol.

This is a development-screen decision from one low-level-policy training seed, not a general rejection of joint
exit-state/time models.

## Audit outcome

- OGBench 1.2.1 and the requested medium/large state datasets are available on the cluster.
- Official OGBench reference-agent source and public HSVL source are available. No pretrained OGBench agent checkpoint
  is distributed in the inspected source, so this pilot trained and froze a transparent XY-conditioned GCBC skill.
- No official CompPlan implementation was found. The pilot therefore tests component headroom only; it is not a
  CompPlan reproduction or a complete system-baseline table.

## A0: low-level skill competence — PASS

Job 50487 completed in 5m09s on the MIG partition. The frozen checkpoint was evaluated on 200 held-out validation
trajectory states (50 per horizon), using complete simulator-state restoration and the OGBench 0.5 XY success radius.

| Future-goal horizon | Success | Mean minimum XY distance |
|---:|---:|---:|
| 10 | 92% | 0.402 |
| 20 | 78% | 0.468 |
| 40 | 64% | 0.655 |
| 80 | 74% | 1.340 |
| **Pooled** | **77%** | — |

This passes the frozen engineering gate (at least 70% at 10/20 and at least 50% pooled). Action validation MSE alone was
not used for the decision.

Artifacts: [`summary.json`](../outputs/a0_screen/50487/eval/summary.json),
[`metrics.jsonl`](../outputs/a0_screen/50487/train/metrics.jsonl), and
[`ledger.json`](../outputs/a0_screen/50487/train/ledger.json).

## A1: privileged handoff headroom — borderline PASS on first screen

After a smoke-run audit, all arms were allowed to replan to the same 1,000-step deployment cap and the final-waypoint
oracle no longer borrowed an undeployed future probe. Job 50511 evaluated 100 paired task/seed cases.

| Arm | Success | Mean deployed steps |
|---|---:|---:|
| fixed 10 | 87% | 444.0 |
| fixed 20 | **90%** | **391.4** |
| fixed 40 | 88% | 489.1 |
| fixed 80 | 82% | 731.5 |
| reach-radius heuristic | 86% | 551.2 |
| physics oracle | **100%** | 346.3 |

Oracle minus the strongest non-oracle arm was exactly +10 success points, with a paired episode bootstrap interval
[+5,+16]. This met the pre-set point-estimate screen at its boundary. The oracle spent 250,231 additional privileged
simulator-query steps, separately from deployed actions.

Artifacts: [`summary.json`](../outputs/a1_screen/50511/eval/summary.json) and
[`gate.json`](../outputs/a1_screen/50511/gate.json).

## A2: simple-baseline kill — STOP

Job 50512 used 50 disjoint oracle-collection episodes to train a state-dependent termination classifier. Its input was
the full exit state, current and next waypoints, elapsed duration, and final-waypoint flag. The classifier's grouped
validation exact-duration accuracy was only 29.7% (chance is 25%), so it is not evidence that learned termination is
solved. It is retained as a deliberately simple baseline.

On a fresh 100 paired task/seed screen:

| Arm | Success | Mean deployed steps |
|---|---:|---:|
| fixed 10 | 86% | 448.2 |
| fixed 20 | 91% | 405.3 |
| fixed 40 | 84% | 524.3 |
| fixed 80 | 74% | 764.6 |
| reach-radius heuristic | **92%** | 520.5 |
| learned termination | 89% | 548.3 |
| physics oracle | **100%** | 350.0 |

Oracle minus the strongest deployable baseline was +8 points, paired bootstrap interval [+3,+14], below the pre-set
10-point headroom gate. The oracle again required 250,638 privileged simulator-query steps. Because the oracle is at
100%, improving the simple termination baseline can only shrink this residual; it cannot restore the required headroom.

Artifacts: [`gate.json`](../outputs/a2_screen/50512/gate.json),
[`evaluation summary`](../outputs/a2_screen/50512/eval/summary.json), and
[`termination summary`](../outputs/a2_screen/50512/termination/summary.json).

## Interpretation and next action

The proposed task/controller pair is too recoverable under the fair 1,000-step budget: fixed duration or a reach-radius
rule already solves roughly nine out of ten cases. There is measurable privileged headroom, especially in deployment
steps, but the success gap is not robustly above the pre-registered screen threshold and does not justify a joint
`p(s_exit, tau, outcome | s,u,b)` campaign.

Do not describe this as a negative result for joint handoff modeling, CompPlan, HIQL, or HSVL: none of those full systems
was tested. The supported conclusion is narrower: **Direction A, instantiated as state-based AntMaze-medium with the
frozen XY-GCBC skill, BFS waypoints, and 1,000-step cap, fails the cheap-headroom/simple-baseline gate.**

If the research question is retained, the next task must make handoff quality consequential under a frozen, justified
budget (for example a tight deadline or an irreversible/dynamic transition), and its A0--A2 protocol must be registered
before looking at method results. Under the current report's decision tree, the clean next project is the Direction B
TAWM consistency pilot rather than AntMaze-large or A3.
