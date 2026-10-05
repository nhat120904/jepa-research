# Job ledger: latent_planning_idea_20261001

## 2026-10-01 review check (existing data only, no model/physics)

Question: in the corrected Cube audit (32 snapshots x 300 final CEM candidates,
`diagnosis/results/ogb_true_endpoint_corrected/locked/candidate_costs.csv.gz`), at which
primitive step do successful candidates terminate, and do the predicted-endpoint misses
involve hits between coarse block boundaries?

| Job | Work | Resources | State |
|---|---|---|---|
| 56315 | analysis script, submitted from node-local /tmp | main, 2 CPU, 4 GB, 10 min | FAILED 1 s (script path not visible on compute node); no output |
| 56316 | same script from `/mnt/data/nhatnc129/jepa/review_checks/hit_timing_20261001/` | main, 2 CPU, 4 GB, 10 min | COMPLETED 1 s; `result.json`, `hit_timing_56316.out` |

Result (56316), measured on this single cohort:
- 4,780/9,600 candidates succeed. Success step by block: 1-5: 2,794; 6-10: 566; 11-15: 344;
  16-20: 139; 21-24: 753; 25: 184. 54% of successes occur within 3 steps; in 7/32 snapshots
  every success occurs within 3 steps (near-trivial starts).
- Predicted-endpoint top-1: 16/32; true stopped-endpoint top-1: 21/32; any success: 25/32.
- 9 predicted misses. In 7 of them the successful candidates hit at steps 21-25 (median 23-25);
  the true stopped endpoint fixes 6 of these, and their best successful candidate already has
  predicted rank 1-10 of 300 (near-tie at the top). The 2 misses with mid-horizon hits
  (snapshot 1: steps 9-13; snapshot 17: step 14) are not fixed by the true stopped endpoint
  either.
- Not measured: whether hit candidates leave the goal before step 25 (needs continued simulation).
