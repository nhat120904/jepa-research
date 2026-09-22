# Does a memory VLA make trajectory JEPA unnecessary?

Date: 2026-09-20. Literature/source review plus implementation scope decision; no new
policy/control experiment. The user's method-learning constraint is self-supervision.

## Recommendation

For improving a fixed task with demonstration supervision available, a history-aware
VLA is a more direct baseline to qualify than building a trajectory world model plus
an unqualified value/scoring system. We have not established that Scrub failures are
caused by missing memory, or that history fine-tuning solves this specific task.

Do NOT motivate trajectory JEPA solely by “GR00T cannot remember which places were
scrubbed.” Memory can be learned directly in a policy. A recurrent policy can also
implicitly learn action consequences; explicit prediction has no information monopoly.

History-VLA answers “what action should I take given this history and goal?” A world
model explicitly answers “what observations/effects would follow this proposed action
sequence?” The latter could help with new queries, objectives, action alternatives,
or data efficiency, but introduces prediction error, scoring error and inference cost.
These are hypotheses to test, not guaranteed benefits of a world-model architecture.

## Primary-source evidence

- [HAMLET](https://arxiv.org/abs/2510.00695) introduces compact moment tokens and a
  memory module for VLA action conditioning. The authors report a RoboCasa Kitchen
  100-demo aggregate of 64.1% to 66.4%, plus larger gains on history-dependent real
  tasks. These are not RoboCasa365 Scrub task results and do not predict our gain.
- [Official HAMLET GR00T implementation](https://github.com/myungkyuKoo/HAMLET-Isaac-GR00T)
  explicitly points to its `n1.5` branch for RoboCasa-Kitchen. Availability is verified
  at source level; compatibility with our checkpoint/runtime and cost are untested.
- [MemoryVLA](https://arxiv.org/abs/2508.19236) is another primary example of perceptual
  and cognitive history informing a diffusion action expert. Neither paper proves that
  adding raw history frames alone is sufficient for all long-horizon failures.

A method described as “fully self-supervised world-model learning” may use logged
actions as conditioning variables without task reward labels. History-VLA fine-tuning
against demonstration actions is imitation learning, not the same learning claim.
It remains a legitimate, strong downstream engineering baseline. A frozen GR00T proposal
also inherits supervised pretraining: do not call the entire stack supervision-free.

## Minimal fair eventual control comparison

1. The same GR00T checkpoint fine-tuned with current observation only.
2. The same data/action-loss budget, but history-enabled GR00T.
3. History-enabled GR00T plus a trajectory JEPA selector.
4. History-enabled GR00T plus a matched frame-world-model selector.

Comparing (2) against an untouched checkpoint confounds history with ordinary task
adaptation. Comparing (3) only against a memoryless VLA confounds explicit future
prediction with access to the past. Cases (2)-(4) should share permitted history,
goals, proposal distribution and execution cadence. Evaluate selection on common
candidate banks and evaluate complete closed-loop systems separately. Account for
extra robot interaction, labeled demonstrations, training and inference compute.

For the incremental JEPA claim, keep a strong proposal and ask whether its own
candidate alternatives have independently confirmed exploitable headroom. Memory
could remove that headroom; it could also improve proposals while leaving useful
choices. Either possibility must be measured. If memory alone achieves comparable
success at lower cost, prefer it operationally and do not defend JEPA on that setting.

## Why implement step 2 now anyway?

The existing transformer composer did not implement signature algebra. A small CPU
check of a corrected causal interface and a fixed signature target settles whether
the proposed new experiment is technically well-defined, cheaply and without task
labels. It does not justify a GPU training campaign or certify research novelty.

Implement this in a separate versioned package, keeping the historical experiments
reproducible. No history-VLA training is started: the current user authorized step 2,
not an additional expensive VLA adaptation project. No new reference-tracking arena
or modified native objective is silently introduced.
