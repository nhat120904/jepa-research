# 53750: stop the current Wall implementation

Queue and accounting verified: COMPLETED exit 0, 2m05s. All 41 tests passed. Four
arms trained for 2,000 updates with the same RGB-derived affinity/query supervision.
The operational verdict locked before execution is:
`STOP_CURRENT_WALL_METHOD_IMPLEMENTATION`.

## Main validation result

| Arm | H48 ordered MSE / regret | H64 ordered MSE / regret | Gate |
|---|---:|---:|---|
| Summary16 | .048759 / .116939 | .058699 / .179482 | fail |
| Frame tokens | .049238 / .116939 | .056418 / .204003 | fail |
| Direct query | .077995 / .200266 | .065636 / .200417 | fail |
| No action | .034812 / .023694* | .037174 / .006154* | control |

`*` No-action produces ties; first-argmax selects the default. Its uniform-tie regrets
are .156294/.159988, so the low first-argmax values are not predictive skill.
Nevertheless, action-aware models also fail the independently locked MSE, shuffled-
action and actual-default-regret criteria. `usable_by_arm` is false for summary,
frame and direct.

Summary fits training much better than validation: H48 ordered MSE .013502 train
versus .048759 validation; regret .045990 versus .116939. Direct overfits more strongly
(.005963 -> .077995). This is not evidence that compression is the unique bottleneck.
The data/model/objective combination fails to generalize counterfactual ranking.

At H48, summary and matched frame have essentially identical validation behavior;
paired ordered-MSE difference has exploratory 95% prefix-bootstrap interval
[-.01232, .01221]. Summary therefore has no demonstrated accuracy advantage over
the frame-token control. Cross-query results are also not positive evidence: only
3--5 positive ordered branches occur at each horizon and every action-aware arm has
the same selection regret at H48/H64.

## Scope of the negative result

Stop:

- this Wall dataset/proposal bank and 24-prefix training setup;
- the current summary16, frame-token and direct architectures;
- further tuning of readouts, latent losses, unroll length, epochs or larger variants;
- claiming a planning or predictive-abstraction improvement from these pilots.

Not proven impossible:

- all predictive abstractions, all trajectory JEPA variants, or every future arena;
- compute-only advantages not measured here;
- methods trained on a substantially different dataset/objective and evaluated with
  independent queries and a policy proposal distribution.

Any restart must be a new research design, not another hyperparameter continuation:
pre-register a task with nontrivial oracle-selection support, substantially more
independent prefixes, a capable matched frame/direct baseline, queries independent
of proposal construction, and an untouched test split. It should first establish
that an action-aware baseline generalizes candidate ranking; only then compare a
compact predictive abstraction. No automatic job is authorized from this result.

Limitations: one seed, reused development validation, three layouts, RGB-derived
task-designed labels rather than generic self-supervised JEPA, and no physical success
or closed-loop control measurement. These limitations reduce generality; they do not
invalidate the predeclared decision to stop spending on this implementation.

Artifacts: `/mnt/data/nhatnc129/jepa/predictive_abstraction/query_native_53750/output/`.
