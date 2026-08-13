# MetaWorld + OGBench paper evidence revision

Date: 2026-08-11. Target: anonymous TMLR submission. Status: accept-quality
revision round in progress; original evidence revision complete.

## Revision objective

Turn the current MetaWorld mechanism paper into a coherent cross-substrate
diagnostic paper without weakening claim discipline. MetaWorld remains the
causal mechanism study; the corrected OGBench-Cube audit is added as an
independent same-population replication of residual terminal-cost misranking.

## Evidence hierarchy

1. **MetaWorld mechanism evidence.** Exact simulator candidate dynamics,
   physical positive controls, same-population ranking, a residual-matched
   null, and a cost-only CEM refitting intervention identify how the fixed
   stateprobe composition fails.
2. **MetaWorld latent-L2 evidence.** Exact-dynamics endpoint results and
   same-population ranking show weak/task-dependent ordering. A targeted
   DINO-push cost-only branch identifies a refitting effect in one
   representative cell, without supporting a cross-task claim.
3. **MetaWorld task breadth.** A preregistered extension adds button-press and
   window-close failures under 16/16 reference controls and a drawer-close
   success boundary; two failed reference controls remain reported and
   excluded from interpretation.
4. **OGBench-Cube replication.** On fixed final CEM populations from one
   released LeWM checkpoint, reproducibly rendered true endpoints remove
   learned endpoint-prediction error yet leave physical selection regret. This
   is cross-model and cross-benchmark support, not an OGBench causal refitting
   result or a deployable simulator-assisted method.

## Required disclosure and claim boundaries

- The released `dino_wm_metaworld` and `jepa_wm_metaworld` stacks share the
  same frozen `dinov2_vits14` encoder. Because exact dynamics bypass the
  predictors, they are two released-stack checks, not two independent
  representation replications.
- The OGBench claim is about the released `quentinll/lewm-cube` encoder plus
  terminal squared-L2 on one task and checkpoint.
- Say “exact dynamics are insufficient” and “learned-dynamics error is not
  necessary for failure in these audits,” not that dynamics never matter.
- The physical/reference selectors are diagnostic upper bounds, not fair
  deployable competitors.
- Do not attribute an encoder-only failure, claim universality across
  representations/planners, or claim that scaling cannot fix the problem.

## Manuscript changes

- Broaden the title and abstract around exact dynamics and
  representation-derived cost ordering.
- Reframe the introduction around four separable failure sites and state the
  evidence hierarchy explicitly.
- Add OGBench/LeWM/stable-world-model citations and protocol details.
- Add a compact OGBench result table with success, physical distance, regret,
  and rank agreement.
- Add the completed preregistered task-breadth table, including its positive-
  control exclusions and successful boundary condition.
- Separate cross-benchmark commonality from benchmark-specific mechanisms in
  Discussion.
- Update scope, conclusion, reproducibility appendix, paper README, and claim
  ledger.
- Audit duplicated/stale prose, labels, citations, table widths, undefined
  references, anonymity, and LLM-use disclosure.

## Verification and job ledger

The PDF must be built with `diagnosis/scripts/slurm_paper_build.sh`; no LaTeX
build is run on the login node. Record the job ID, terminal state, page count,
and warning audit here after submission.

| Job | Command | State | Output |
|---|---|---|---|
| 38381 | `sbatch diagnosis/scripts/slurm_paper_build.sh` | completed; 16 pages, no undefined references/citations or overfull boxes; one PDF-string warning and one underfull paragraph fixed before final build | `paper/main.pdf`; `/mnt/data/nhatnc129/jepa_runs/logs/paper_build_38381.out` |
| 38382 | `sbatch diagnosis/scripts/slurm_paper_build.sh` | completed; clean final pass before task-breadth addition | `paper/main.pdf`; `/mnt/data/nhatnc129/jepa_runs/logs/paper_build_38382.out` |
| 38391 | `sbatch diagnosis/scripts/slurm_paper_build.sh` | completed exit 0; final 17-page US-Letter PDF; no final-pass warnings, undefined citations/references, underfull/overfull boxes, or duplicate labels | `paper/main.pdf`; `/mnt/data/nhatnc129/jepa_runs/logs/paper_build_38391.out` |

## Accept-quality review round (2026-08-11)

An independent paper review identified three evidence gaps that must be closed
before strengthening the recommendation: the main causal audit used a shaped
reference score rather than a direct task outcome; the complete scalar
stateprobe composition had no held-out validation; and CEM refitting had only
been intervened on for stateprobe, not the deployed latent-L2 cost.  The
following jobs address those gaps without overwriting the registered artifacts.

| Job | Exact command | State | Planned output |
|---|---|---|---|
| 38667 | `sbatch diagnosis/scripts/slurm_task_aligned_cost_audit.sh` | completed exit 0 in 57 s | `diagnosis/results/task_aligned_cost_audit*`; task-distance null excess is small/non-robust, while the stateprobe branch loses 0.84--1.55 cm in best task distance |
| 38668 | `sbatch diagnosis/scripts/slurm_stateprobe_scalar_validation.sh` | both array cells completed exit 0 in 37 s | `diagnosis/results/stateprobe_scalar_validation_{dino,jepa}*`; held-out scalar MAE 2.61/2.75 cm and trajectory-mean Spearman 0.832/0.822 |
| 38669 | `sbatch diagnosis/scripts/slurm_shared_population_l2_branch.sh` | completed exit 0 in 56 m 57 s | `diagnosis/results/shared_branch_l2_dino_push*`; shared-noise latent-L2 versus privileged true-state CEM refitting on DINO push, 8 paired seeds |
| 38671 | `sbatch diagnosis/scripts/slurm_stateprobe_scalar_validation.sh` | both cells completed exit 0 in 35/33 s | push/pick-only scalar MAE 2.43/2.84 cm and trajectory-mean Spearman 0.800/0.826 over 11 held-out primary-task trajectories |
| 38678 | `sbatch --dependency=afterok:38669 diagnosis/scripts/slurm_task_aligned_cost_audit.sh` | completed exit 0 in 58 s | L2 refitting worsens final best task distance by 2.04 cm [1.58,2.51] and shaped cost by 3.64 cm [2.83,4.36], 8/8 seed means positive, exact sign/sign-flip p=0.0078 |
| 38679 | `sbatch diagnosis/scripts/slurm_paper_build.sh` | completed exit 0 in 3 s; 18 US-Letter pages; current final-pass `main.log` has no warnings, undefined citations/references, or box errors (stdout retains expected first-pass undefined messages before BibTeX/reruns); pages 7--11 visually inspected with no clipping or unreadable tables | interim `paper/main.pdf`, 544,944 bytes; separate final build remains required after L2 integration |
| 38691 | `sbatch diagnosis/scripts/slurm_paper_build.sh` | completed exit 0 in 3 s; 18 US-Letter pages; clean final pass | integrated latent-L2 branch result and limitations; pages 1, 11, and 14 visually inspected with no clipping or unreadable tables |
| 38692 | `sbatch diagnosis/scripts/slurm_paper_build.sh` | completed exit 0 in 3 s; 18 US-Letter pages; final `main.log` has no warnings, undefined citations/references, underfull/overfull boxes, duplicate labels, or errors; PDF metadata has blank Author | final `paper/main.pdf`, 546,100 bytes |
| 38694 | `sbatch diagnosis/scripts/slurm_paper_build.sh` | completed exit 0 in 3 s | substantive narrative rewrite: simplified abstract/introduction/related work, consolidated Results, 14 pages |
| 38695 | `sbatch diagnosis/scripts/slurm_paper_build.sh` | completed exit 0 in 3 s; final `main.log` has no warnings, undefined citations/references, underfull/overfull boxes, duplicate labels, or errors; blank Author metadata | final polished 14-page `paper/main.pdf`, 518,400 bytes; title and pages 1, 2, 5, 6, 8, 9, and 12 visually inspected |

All terminal job states and generated numbers have been checked. The final PDF
was rebuilt on a compute node and rereviewed against the TMLR criteria.

## Submission-readiness checklist

- [x] All headline numbers trace to current, non-retracted artifacts.
- [x] Shared-encoder limitation is explicit in the main paper.
- [x] OGBench historical Stage-0 artifacts are never cited.
- [x] Causal language is restricted to the MetaWorld stateprobe interventions
      and the targeted DINO-push latent-L2 intervention.
- [x] Reference/simulator arms are labelled privileged diagnostics.
- [x] Bibliography metadata and anonymous TMLR front matter are valid.
- [x] LaTeX builds without undefined references/citations or material layout
      warnings.
- [ ] Supplementary archive and final OpenReview metadata remain author tasks.
