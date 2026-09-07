# Technical Note — Action-Space Curvature Straightening in Latent World Models

---

## 1. The idea

A latent world model is **trained to predict** the next observation, then
**deployed inside a planner** that searches over action sequences. Only the
first job is ever optimised for.

What the planner actually queries is one map: *given an action sequence `a`,
what latent state do I end up in, and how far is that from the goal?* Call it
`Phi(a)`. The planner samples hundreds of candidate `a`, scores each with
`||Phi(a) - z_goal||`, and keeps the best. Nothing else enters the decision.

Existing "straightening" work makes latent trajectories straight **in time**.
But a planner does not optimise over time — it optimises over **actions**. So
the property that should matter is straightness of `Phi` **in action space**:
nudge the action slightly, and the predicted outcome should move slightly, in a
predictable direction.

**Hypothesis.** If `Phi` is curved in action space, the cost landscape develops
dips the real physics does not have. The planner walks into one, reports a good
action, and the robot does something else. We call such a dip a **false valley**.

*Analogy:* a road map that draws straight roads as curves. Distances still look
about right, so the map seems fine — but the shortest route computed on it is
the wrong route.

---

## 2. Implementation

### Measuring curvature

No training required.

1. Take a real state and a real action sequence `a`.
2. Perturb symmetrically: `a - d`, `a`, `a + d`.
3. The **second difference** `Phi(a+d) - 2*Phi(a) + Phi(a-d)` is the curvature
   along `d`. Zero if the map is straight.
4. Replay the same three sequences **in the simulator** from an exact state
   reset — the realised outcome the model should have matched.

Curvature splits into **radial** (wrong *magnitude* of effect) and **angular**
(wrong *direction* of effect). A **false valley** is recorded when the model's
cost has a local minimum over the triplet and the real physics does not.

### The three arms

| arm | what it does |
|---|---|
| **original** | the released checkpoint, unchanged — the control |
| **continuation** | fine-tune the predictor to stay accurate over *several* rolled-out steps, not just one |
| **continuation + cosine AS** | additionally penalise the *direction change* between consecutive predicted displacements — the explicit curvature-straightening loss |

Same data, same training graph, paired seeds, encoder frozen. The strength
`lambda` of the cosine loss was chosen on a development split; every number
below comes from **held-out states never used for that selection**, under
decision rules fixed in writing beforehand.

**What "original" is.** The publicly released world-model checkpoint for this
task, used unmodified — the same baseline other papers report against, quoted
near **68%** task success with claimed gains of **+8.1**, **+9.67** and **+14.2**
points over it. Our own measurement of that identical checkpoint is **62.6%**,
lower because our evaluator draws 50 episodes at random while those papers use a
fixed episode list. Published gains are quoted only as a sense of scale; the
comparisons that carry weight here are arm-versus-original on identical episodes.

---

## 3. Results

### 3.1 Curvature predicts planning failures

| curvature group | false-valley rate |
|---|---:|
| lowest 25% | 0.34% |
| second | 1.70% |
| third | 5.44% |
| highest 25% | 17.63% |

Top minus bottom: **+17.3 percentage points, 95% CI [+12.1, +23.5]**.

*The numbers:* false-valley rate is the share of tested action triplets where
the model's cost shows a local minimum that physics does not have. The CI not
containing 0 means the gap is not sampling noise.

### 3.2 The curvature is directional, not a scaling error

**97% angular, 3% radial.**

| component | effect on false valleys | 95% CI |
|---|---:|---|
| angular, radial held fixed | **+0.158** | [+0.102, +0.212] |
| radial, angular held fixed | −0.017 | [−0.069, +0.035] |

*The numbers:* each is the change in false-valley rate (scale 0 to 1) caused by
that component alone. An interval straddling 0 means no detectable effect.

### 3.3 The explicit cosine loss failed its own check

At **every** `lambda` tested, angular curvature came out **higher** than the
paired run with the loss off.

*The numbers:* a sign test, not a size test — the intervention moved its own
target the wrong way, so nothing downstream is worth measuring.

### 3.4 Multi-step continuation reduced curvature and false valleys

- **angular curvature down ~13%** (aggregate) / **~6%** (strictly paired
  per-state median);
- **false-valley rate 0.1505 → 0.0585**, paired difference **−0.092, 95% CI
  [−0.152, −0.038]** — a **61% reduction**, same sign on all three seeds.

*The numbers:* 0.1505 means 15.05% of tested triplets had a false valley. −0.092
is the drop in that rate; 61% is the same drop expressed relative to the start.

The effect is **not uniform**:

| starting condition | n | mean change |
|---|---:|---:|
| states that had false valleys | 25 | **−0.208** (fixed 19 of 25) |
| states already clean | 22 | +0.040 (small new valleys in 11) |

*The numbers:* change in false-valley rate per state; negative is an improvement.

### 3.5 None of it improved planning

**Candidate-level test** — all arms score the *same* candidate actions.
Continuation's chosen action landed **+1.5 mm further** from the goal, 95% CI
**[−2.3 mm, +4.5 mm]**.

*The numbers:* distance from the object to the goal at the end of the executed
plan. An interval containing 0 is a null.

Not a measurement floor: candidates spanned 157 mm of outcome, nothing hit its
action bounds, every plan ran to completion.

**Closed-loop test**, official success metric, 10 planning seeds × 50 episodes,
all arms on identical episodes:

| arm | success rate |
|---|---:|
| original | **62.6%** |
| continuation (mean of 3 seeds) | **62.4%** |

Paired difference **−0.20 points, 95% CI [−2.47, +1.80]**, better on 5 of 10 seeds.

*The numbers:* success rate is the percentage of episodes solved; "points" are
percentage points of that rate.

The **same** arm ranges 54% to 78% across planning seeds (sd 6.7 points).

### 3.6 Where the remaining error sits

**The search moves into the region where the cost is wrong.**

| stage | rank correlation |
|---|---:|
| initial candidate population | **+0.247**, CI [+0.163, +0.321] |
| after the search converges | **−0.085**, CI [−0.135, −0.033] |

*The numbers:* rank correlation between the model's cost ordering of candidate
actions and their true outcome ordering. Scale −1 to +1: **+1** = ordered
exactly right, **0** = the ordering carries no information, **−1** = ordered
exactly backwards. Negative means the candidates the model scores better are the
ones that actually do worse.

**The final averaging step is itself lossy.**

| action | distance to goal |
|---|---:|
| what the planner executes (mean of top 30) | 0.159 m |
| those top 30 candidates, individually | 0.128 m |
| a typical random candidate | 0.128 m |
| the best candidate in the population | 0.077 m |

*The numbers:* distance from the object to the goal after execution, in metres;
lower is better.

Executing the single best-seen candidate instead of the average was also a null
(−1.8 points, CI [−3.6, +0.4]).

---

## 4. Contact-aware gating versus matched random gating

A second, related question, tested on a contact-rich pushing task with a
different frozen model.

**The premise.** Physical trajectories do not bend uniformly — they kink where
contact starts or stops. Measured: object trajectories bend **2.4× more** at
contact-mode switches than elsewhere. So rather than straightening everywhere,
apply it only at those contact transitions.

**The control that makes it a test.** A matched **random** arm drops exactly the
same *fraction* of steps, chosen at random instead of by contact. If contact
alignment carries the information, the contact arm must beat it. Both forms were
run: a frozen projector on top of the encoder, and a fine-tuned encoder.

| form | contact-aligned | matched random | reference: do nothing |
|---|---:|---:|---:|
| frozen projector | +0.011 | **+0.051** | −0.085 |
| fine-tuned encoder | −0.084 | −0.080 | −0.085 |

*The numbers:* the same rank correlation as §3.6, measured after search, with
the seed-to-seed spread around 0.13.

**Reading:** the premise is true — contact really is where trajectories bend —
but contact alignment contributes nothing its random control does not. In the
frozen form the random arm scores higher on all three seeds; in the fine-tuned
form the two arms do not separate. Refuted in both forms.

---

## 5. Conclusion

1. **The diagnosis holds and replicates.** Angular action-space curvature
   predicts false local minima, with a large, monotone, CI-clean effect on
   disjoint samples.
2. **The mechanism responds to intervention.** Multi-step continuation lowers
   angular curvature and cuts false valleys by 61% on held-out states.
3. **It does not convert into planning performance.** Null on candidate quality
   and null on closed-loop success, with intervals tight enough to exclude the
   published effect sizes.
4. **Contact-aware gating is refuted** against its own matched random control,
   in both the frozen and fine-tuned forms.
5. **Therefore:** false valleys are a genuine local pathology of the cost
   landscape but **not the planning bottleneck**. The remaining loss sits in the
   narrow region the search itself creates, and in the planner's averaging step
   — not in a global defect of the latent geometry.

Reaching a null *after* the first three links held is a sharper result than not
having tested it: the mechanism is real, measurable and movable, and it still
does not move the planner.
