# Technical Note Supporting the August 2026 Performance Report

**Researcher:** Nguyễn Cường Nhật  
**Reporting period:** 01–31 August 2026  
**Topic:** Planning misalignment and action-space curvature in latent world models

## 1. Objective

This work studied why a latent world model can predict plausible future observations but still guide a planner toward poor actions. The main hypothesis was that the model may distort the local relationship between an action sequence and its predicted outcome. This distortion can create a **false valley**: an action looks locally optimal according to the model, although replaying the same action in the simulator shows that it is not.

The main experiments used the released LeWorldModel (LeWM) checkpoint on OGBench-Cube. A related contact-gating experiment used DINO-WM on the MetaWorld pushing task.

## 2. Action-space curvature straightening

### Simple idea

Let `Phi(a)` be the final latent state predicted by the model after executing action sequence `a`. For a small action change `d`, three predictions are compared:

`Phi(a-d)`, `Phi(a)`, and `Phi(a+d)`.

If the change from the first point to the second is similar to the change from the second to the third, the local action-to-outcome map is nearly straight. If their directions or lengths change sharply, it has high curvature. The curvature was separated into:

- **Angular curvature:** the predicted direction of motion changes.
- **Radial curvature:** the predicted amount of motion changes.

The same three action sequences were replayed from exactly the same simulator state. This same-state comparison distinguishes distortion introduced by the model from genuine nonlinearity in the physical system.

### Implementation

Three model variants were evaluated:

1. **Original:** the released LeWM checkpoint without modification.
2. **Multi-step continuation:** the predictor was fine-tuned to remain accurate over several rollout steps.
3. **Continuation + cosine straightening:** the same continuation training plus a cosine loss encouraging consecutive predicted changes to point in the same direction.

The encoder was frozen. The cosine-loss strength was selected on development data, while the main results were measured on held-out states. All comparisons used the same states, candidate actions, and evaluation procedure.

## 3. Main results

### Curvature was associated with false planning minima

| Angular-curvature group | False-valley rate |
|---|---:|
| Lowest 25% | 0.34% |
| Second 25% | 1.70% |
| Third 25% | 5.44% |
| Highest 25% | 17.63% |

The highest-curvature group had a false-valley rate **17.3 percentage points** above the lowest group, with a 95% confidence interval of **[12.1, 23.5] points**. In simple terms, false local minima became much more frequent as angular curvature increased.

Approximately **97%** of the relevant curvature effect was angular and **3%** was radial. When the other component was held fixed, angular curvature increased the false-valley rate by **0.158** (95% CI **[0.102, 0.212]**), while the radial effect was **−0.017** (95% CI **[−0.069, 0.035]**). The radial interval includes zero, so no reliable radial effect was detected.

### What the interventions changed

The explicit cosine straightening loss did not pass its own manipulation check: every tested loss weight produced higher angular curvature than its paired continuation-only run. Therefore, this loss did not provide evidence that directly penalizing curvature improved the model.

Multi-step continuation alone was more effective. It reduced angular curvature by about **13% in aggregate** (about **6%** under the strict paired-state summary). It also reduced the false-valley rate from **15.05% to 5.85%**, a paired decrease of **9.2 percentage points** (95% CI **[−15.2, −3.8] points**), equivalent to a **61% relative reduction**.

However, the geometric improvement did not improve planning performance:

- In candidate-level evaluation, continuation selected an action whose final outcome was **1.5 mm farther** from the goal; the 95% CI was **[−2.3, 4.5] mm**, which includes zero.
- In closed-loop evaluation over 10 planning seeds × 50 episodes, success was **62.6%** for the original model and **62.4%** for continuation. The paired difference was **−0.20 percentage points**, with 95% CI **[−2.47, 1.80] points**.

Thus, continuation clearly reduced the measured curvature pathology, but this reduction did not translate into better action selection or higher task success.

## 4. Contact-aware gating versus matched random gating

Contact can create genuine sharp changes in motion, so forcing a representation to be straight across every contact transition may be inappropriate. The contact-aware variant therefore **disabled the straightening loss when contact mode changed**. Its control disabled the loss on the same fraction of transitions—about **6.6%**—but selected those transitions randomly. This matched random control tests whether contact information itself adds value.

The physical premise was supported: object trajectories curved approximately **2.4× more** when contact started or stopped than at other times. However, the experiment did not provide evidence that contact-aware gating outperformed the matched random control.

| Model form | Contact-aware | Matched random | Unmodified reference |
|---|---:|---:|---:|
| Frozen projector | +0.011 | **+0.051** | −0.085 |
| Fine-tuned encoder | −0.084 | −0.080 | −0.085 |

The reported value is Spearman rank correlation between the model cost and the true task outcome after CEM refinement. A positive value means the model tends to rank better actions correctly; zero means little ranking information; a negative value means the ranking is reversed. These means are all close to zero, so they should not be interpreted as a strong absolute improvement by any gated method. With the frozen projector, the matched-random arm was numerically higher on all three training seeds (+0.051 versus +0.011), but the experiment was not designed to establish a reliable aggregate improvement from this small difference. With encoder fine-tuning, the two gating methods were effectively indistinguishable (−0.084 versus −0.080).

The defensible conclusion is narrower: although contact transitions are physically special, this experiment found no reliable evidence that identifying them provides an advantage over omitting an equal number of terms at random. It is a negative result about the proposed contact-specific mechanism, not evidence that random gating improves planning.

## 5. Exploratory CEM-ranking audit on LeWM/OGBench-Cube

The LeWM experiment also compared the model's candidate ranking with the physical outcome ranking. The pattern was a loss of ranking signal after CEM refinement, but it was weaker than the separate DINO-WM/MetaWorld result:

| Candidate set | Mean rank correlation between model cost and physical outcome |
|---|---:|
| Initial CEM proposal | +0.112 (39 states) |
| Original CEM final population (iteration 29) | +0.020 (37 states) |

Thus, on LeWM the initially weak positive ranking signal was almost absent in the final candidate population; it did **not** become reliably negative. Two final-population states had tied physical outcomes, for which rank correlation is undefined.

This is a secondary, exploratory comparison because the final population was selected by the original model's own CEM search and no pre-specified confidence interval was computed for the initial-to-final change. It supports the cautious statement that model ranking is weak in the final CEM region, but it does not establish a general or causal claim that CEM reverses the ranking.

## 6. Conclusion

Angular action-space curvature is a measurable local failure mode and strongly predicts false planning minima. Multi-step continuation reduced both curvature and false valleys, but did not improve candidate quality or closed-loop success. Contact-aware gating showed no reliable advantage over a matched random control. The LeWM CEM audit is consistent with weak cost ranking in the final CEM region, but does not identify that ranking error as the definitive remaining planning bottleneck.
