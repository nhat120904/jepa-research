# AGENTS.md

## Default workflow: implement, run, debug, improve

- Carry authorized work through implementation, training and closed-loop evaluation;
  then fix the observed bottleneck and iterate. Do not stop at a plan or an offline probe.
- Do not add chains of novelty, headroom or go/no-go gates before implementation.
  Integrate focused diagnostics and useful controls into the end-to-end run. Run a
  separate check only when it resolves a concrete implementation or experiment decision.
- Fix correctness failures (leakage, alignment, simulator restore, numerical errors)
  before trusting results. A weak metric or failed configuration calls for diagnosis;
  it is not an automatic reason to abandon CTA. Do not assume the WM or cost is always
  the bottleneck based on a different experiment.
- Continue routine fixes and bounded iterations within the authorized scope and compute
  budget without repeatedly asking permission. Avoid speculative sweeps and redundant
  experiments; explain what each substantive run is meant to improve or establish.
- Report actual progress plainly: changes, verified job state, results, remaining issue
  and next action. Keep method explanations understandable and distinguish hypotheses
  from findings.

## Evidence and context

- Compare relevant baselines with matched candidates, horizons, data and compute, or
  disclose differences. Separate offline ranking, oracle results and learned closed-loop
  success. Development wins need held-out evaluation and uncertainty before strong claims.
- Older programmes are historical evidence, not the current agenda or universal prohibitions. 
  Consult their reports when relevant; do not reopen them by default. Latest user instructions 
  take precedence over old protocols.

## Compute and shared workspace (mandatory)

- This is a Slurm **login node**. Use it only for source/git work, small metadata reads,
  syntax checks and scheduler queries. Run physics, rendering, model loading, training,
  encoding and bulk result analysis through `sbatch` on compute nodes, including CPU work.
- Give each batch job explicit resources and a `--time` limit. Never use interactive
  GPU allocations/`salloc`, idle GPU holders, or long blocking commands from an agent
  session. After submission, record the job ID and let it run; do not foreground-poll.
- Verify job state with **both `squeue` and `sacct`** before reporting or acting on it.
  Check for peer/duplicate work before submission. Never submit speculative duplicate arrays.
- Immediately `scancel` jobs found to be wrong or unnecessary. Never leave an idle job
  holding a GPU.
- Preserve other sessions' dirty files. Use distinct run directories; record source/config,
  checkpoints, partial results and job IDs in the direction's ledger. Do not overwrite runs.
- **Monthly usage cap (user rule):** the account `nhatnc129` must never be among the
  top 5 users of the cluster in the current calendar month for GPU, CPU or memory hours.
  Before every GPU submission and every CPU array, check the month-to-date ranking:
  `sreport -t hours -T gres/gpu,cpu,mem cluster UserUtilizationByAccount start=$(date +%Y-%m-01) end=now -P -n`
  (column 6 = hours). Stay at or below **90% of the 5th-ranked user's hours**, *counting
  the planned job at its time limit*. If a submission would cross that, make it smaller
  (fewer episodes/seeds, shorter limit) or wait, and tell the user. The CVPR deadline
  does not relax this rule; spend the budget on the runs the paper needs, not on sweeps.
