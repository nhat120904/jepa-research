# Oracle candidate coverage-versus-selection protocol (2026-07-13)

## Question

When latent-oracle planning fails despite perfect simulator dynamics, is the
failure caused by the search proposal not sampling a physically useful plan, or
by the representation-induced proxy preferring the wrong plan among useful
candidates already present?

This is the discriminator needed before attributing a failure solely to proxy
geometry. It tests candidate coverage and candidate ranking separately on the
same simulator rollouts.

## Locked audit

- Checkpoints: `dino_wm_metaworld`, `jepa_wm_metaworld`.
- Contact tasks: `mw-push`, `mw-pick-place`.
- Costs: latent `l2` and off-policy-robust `stateprobe`.
- Honest controls: `mw-reach × l2` for each checkpoint.
- Planner: CEM, 100 candidates, 6 iterations, horizon 6, three model actions
  executed per replan, strict 100-step episodes.
- Unseen audit seeds: 40000--40015.
- Top-k set: lowest-cost 10% of the population.

The runner records, for every candidate in every iteration:

- latent proxy cost;
- exact MetaWorld success at any point and at the candidate horizon endpoint;
- physical task progress (EE-to-goal for reach; object-to-goal for contact);
- the privileged state-oracle shaped cost (object-to-goal plus hand approach on
  contact tasks);
- a deterministic action-sequence hash and compact action norms.

## Estimands

For each candidate population:

1. **Success coverage:** whether at least one exact simulator-successful candidate
   exists.
2. **Selected success:** whether the proxy argmin candidate is successful.
3. **Missed success:** success was covered but the proxy argmin is unsuccessful.
4. **Physical opportunity regret:** physical distance of the proxy argmin minus
   the best physical distance in the same sampled population.
5. **Ordering agreement:** Spearman correlation and top-10% overlap between proxy
   and physical costs on identical candidates.

Intervals resample episode seeds as clusters. Replans and iterations are not
treated as independent episodes.

## Decision logic

- Low success coverage and low best progress: proposal/budget failure remains a
  viable explanation; proxy-only conclusions are not licensed.
- High success coverage with high missed-success rate, positive within-population
  regret, and poor/negative ordering agreement: candidate selection by the proxy
  is the dominant local failure.
- Both can coexist. Report their quantities separately rather than forcing one
  binary attribution.

The selected candidate here is the lowest-proxy member of the evaluated
population. Standard CEM executes the final refitted mean, which need not equal
that candidate; therefore this is a direct ranker audit, not a complete measure
of executed-plan regret or global planning regret.

## Slurm commands

```bash
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
COVSEL_JOB=$(sbatch --parsable scripts/slurm_oracle_coverage_selection.sh)
sbatch --dependency=afterok:${COVSEL_JOB} \
  scripts/slurm_oracle_coverage_selection_analysis.sh
```

Main outputs are `results/oracle_covsel_*_{episodes,iterations}.csv`, compressed
raw candidate dumps `results/oracle_covsel_*_candidates.csv.gz`, and
`results/oracle_coverage_selection.md` plus summary/validation CSVs.
