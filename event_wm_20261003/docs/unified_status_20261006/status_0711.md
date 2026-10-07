
## 2026-10-06 live verification (07:11 ICT)

- Verified both schedulers: account squeue empty. 57636 TIMEOUT at05:00:01 elapsed, killed06:49:58 ICT; 57637 remains COMPLETED.
- Puzzle57636 trained and saved model, completed dev seed0 closed loop:6/30 (6,0,0,0,0 of6), first_plan_found0.2,28.7min. Seed1 eval did not finish/save final JSON: log contains96 unique completed episodes,20 successes (task1 20/20; tasks2-4 each0/20; task5 0/16). Seed2 not started. Partial20/96 must not be labelled a complete100-episode score.
- Dev first calls for tasks2-5 exhaust20053 expansions despite best_h approximately0.70/0.63/0.32/0.34; initial planning latency15.65/24.40/23.79/23.81s. The previous flat-at30 diagnosis no longer describes these search outputs; the new config still fails to produce complete plans. Root cause needs separating heuristic optimism, WM rollout error, goal checks, and candidate constraints before another rerun. No new job submitted.
- Local evidence: docs/unified_status_20261006/{ew_stwm_57636.out,puzzle_57636_dev.json}. Latest cube57637 result unchanged:seed1 99/100,seed2 97/100, state track.
