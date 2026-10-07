
## 2026-10-06 live verification (03:50 ICT)

- Verified squeue and sacct: 57636 RUNNING ~02:00 elapsed, 5 h limit; 57637 RUNNING ~01:56 elapsed, 4 h limit. Both currently train cost-to-go, not closed loop. Latest logged h steps: puzzle 85000/150000; cube 115000/150000. WM training finished.
- Verified saved JSON summaries directly: pixel cube pf_57587 22/30 (6,6,6,4,0 of 6); state cube wm_57631 seed1 100/100 and seed2 97/100 (20,19,20,19,19 of 20); state puzzle initial 57614 seed1 20/100 (20,0,0,0,0 of 20).
- Latest running unified config uses event-pos-only, walk goals, width2048, absdiff, 150k h steps and 300k walk pool. Earlier state skills and events are reused; no new skill training in these two jobs. Outputs have no final closed-loop result yet.
- State input is the benchmark observation vector with documented object-factored layout, rather than the SAM2/pixel reader. State data use all3000 episodes; pixel runs use self1000. Do not combine the two tracks. One training seed; evaluation env seeds1/2 have already informed prior model iterations and should not be called newly untouched holdouts for the ongoing config.
