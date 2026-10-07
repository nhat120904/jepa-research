# Scene STATE progress and experiment report

The latest scene STATE pipeline scores **92/100 on fresh environment seed5**, after adding a learned event-support prior and continuing the skill on genuine multi-object event segments. A matched frozen-parent-skill control on seed5 scores87/100. The support-only variant scored85/100 on seed3 and87/100 on seed4; the initial scene pipeline scored34/100 on seed3 and32/100 on a matched CPU seed4 control. All are learned closed-loop results, with simulator scoring and raw episode logs retained. These variants have different skill weights; do not pool their scores as one checkpoint.

| Latest seed5 task | Parent skill | New skill |
|---|---:|---:|
| open |20/20|20/20|
| unlock and lock |20/20|20/20|
| rearrange medium |20/20|20/20|
| put in drawer |17/20|18/20|
| rearrange hard |10/20|14/20|
| Total |87/100|92/100|


| Task | Initial seed3 GPU | Base seed4 CPU | Support seed3 CPU | Support seed4 CPU |
|---|---:|---:|---:|---:|
| open | 20/20 | 20/20 | 20/20 | 20/20 |
| unlock and lock | 0/20 | 0/20 | 20/20 | 20/20 |
| rearrange medium | 14/20 | 12/20 | 18/20 | 20/20 |
| put in drawer | 0/20 | 0/20 | 18/20 | 18/20 |
| rearrange hard | 0/20 | 0/20 | 9/20 | 9/20 |
| Total | 34/100 | 32/100 | 85/100 | 87/100 |

Seed3 paired outcome counts:33both successful,52previous failures now successful,1previous success now failed,14both failed. Original seed3 inference used GPU5workers; repaired evaluation uses CPU4workers and bfloat16 autocast. The same-hardware seed4 base control in57810 scores32/100:32both successful,55fixed,0regressions,13both failed. Logged initial states match exactly; logged goals differ by at most0.001canvas units (three-decimal output, tolerance1.597). Execution settings match. The seed4gain is55percentage points; extra6000support training updates and head inference are the intentional cost difference. This control is the previous method configuration, not a published benchmark baseline. All arms share the parent WM/h/skill, candidates, horizon750simulator steps, max20000expansions, timeout250, execution4actions per chunk, rest5frames and recovery8steps. Repair adds a6000update support MLP; extra training/model cost must be disclosed. Separate scene model training and adapter calibration also remain disclosed; this is not a single zero-shot cube/puzzle/scene checkpoint.

## Concrete failure and bounded repair

Initial plans are found in100% of episodes but miss preconditions. For task2, planner attempts a locked drawer/window directly; the WM predicts a handle moving while the observed handle remains still. For tasks4/5 it attempts cube placement inside a closed drawer. High accuracy on observed validation events (99.84%acted,84.43%side effects) did not establish feasibility of arbitrary requests.

The added MLP consumes public object state, acted identity and intended target. Positives are genuine TRAIN event tuples; negatives randomly corrupt other-object context while preserving the acted object and target. Thresholds are the2ndpercentile of genuine play-validation scores per actor, retaining97.76–97.99% of those events. Unsupported requests return the input state instead of creating imagined progress. Corruptions are density-estimation negatives, not simulator-confirmed failures. No task IDs, lock relations, prescribed order, distance solver or scripted controller is supplied. WM/h/skill weights remain frozen.

After repair, first-plan depths are2,6,4,5,8 across tasks1..5; median initial planning time0.615s on CPU in seed4. A task5 plan can contain8events, but success often requires additional replanning: median10completed event records for successful task5 seed4 episodes,475simulator steps. The final successful event may be absent from the records because the environment terminates before the rest detector fires.

## Remaining bottleneck

In seed4 all13failures are2task4 and11task5 episodes, exhausting750steps. Robot placement and drawer motion are coupled: closing/reopening the drawer transports the cube. In the40task4/5 episodes,61cube-event records move the drawer beyond tolerance; median drawer prediction error4.338canvas units against position tolerance1.597. These traces include off-data execution and early completion, so this is an execution-distribution diagnostic, not an offline WM validation score or proof that the WM alone causes every failure. Cube skill misses also occur.

The loop's original termination rule can stop on a settled changed side effect while the acted object remains in transit. The opt-in acted-rest-only correction is isolated in source_boundary_v3 and tested with an explicit transit/rest semantic case. Job57810 completed this40episode diagnostic:task4=17/20,task5=9/20 (26/40 vs27/40);2fixed and3regressed. The change does not solve the remaining bottleneck and is not adopted as the reported method.

## Data, resources and scope

Official public state play data:1000TRAIN and100play-validation episodes,1,001,000/100,100frames;17,858/1,851extracted events. Public40D observations map to5object tokens with6fields. Buttons/handle contact geometry is calibrated from TRAIN public effector/joint trajectories. This is benchmark-specific preprocessing. Simulator state appears only in diagnostics and benchmark scoring. These runs do not establish pixel reader quality, new-layout generalization, multi-training-seed robustness or a matched published-baseline win. NoLHBL is included.

57772 trains WM30000steps,h150000steps(width2048,absdiff,imagined-walk pool300000), BCskill40000steps. Completed2:35:29,exit0:0. 57774initial100episode evaluation completed47s. 57808support training+105episode evaluation completed2:31,CPU4/12GB, noGPU. 57810matched comparison and boundary diagnostic uses CPU4/12GB/15min. Each run has distinct output directories and source SHA manifests. Account use is below90% of the fifth-ranked monthly user in GPU/CPU/memory hours including requested limits; no idle GPU or duplicate array.

Remote output root: `/mnt/data/nhatnc129/jepa/event_wm/scene_state_20261006`.
Source snapshots:source(initial),source_support_v2(support),source_boundary_v3(opt-in boundary).
Parent checkpoint:job_57772/wm/u_model.pt; skill:job_57772/skill/u_skill.pt.
Support checkpoint:support_57808/u_model.pt; compare results:compare_57810.
Full source/config/job history and hashes are recorded inJOB_LEDGER.md and per-run ledger files.

## Completed skill continuation,57812

948/17,858TRAIN events involve multiple moved objects; the prior BC filter excluded them. The generic inclusion experiment resumes the parent skill for10000updates atlr3e-5, with fresh optimizer and identical architecture/normalization/chunk8/release10. Frame/event pairs increase from1,079,795 to1,135,223 using the same raw play episodes. WM/h/support and the original event termination stay frozen. Source_skill_v4 is isolated; no task-specific control sequence or simulator label was added.

Development seed4task4/5 improves from27/40 to31/40 (18/20 and13/20). Predeclared final checkpoint on fresh seed5 yields92/100 versus frozen parent87/100 on the sameGPU5workers, state starts, candidates and horizons.9fixed,4regressed,83both successful,4both failed. Goals differ by at most0.001canvas units at logged precision against tolerance1.597. Median initial planning time0.145s onGPU; this cannot be compared directly with CPU timings. Eight failures remain,2task4 and6task5, all exhausting750steps.

The skill experiment adds10000training updates and changes segment inclusion together; a compute-matched continuation-only control was not run, so inclusion alone is not causally isolated. One training seed and100evaluation episodes remain preliminary; episode-wise Wilson95 interval is85.0–95.9%, and the paired +5point gain has exact two-sided p≈0.267. These are development results, not evidence of a statistically established or published-baseline win. Multi-training-seed replication, pixel performance and new-layout transfer remain unmeasured.

57812 completed2:19,exit0:0, verified with bothsqueue andsacct. All listed scene jobs have completed. Current checkpoint pair and evaluation path are recorded inLATEST_RUN.json.
