# Scene 3x2: event-state source against planner feedback

Verdict: **INCONCLUSIVE**

128 paired resets at K=112.  Reproduction of the `event_progress` column against the ablation run: 896 rows, 0 mismatches.  Task-4 feedback identity: 0 mismatches.

| Feedback | `frame_full` | `obs_history_full` | simulator q |
|---|---:|---:|---:|
| `event_progress` | 69.01% | 93.49% | 87.50% |
| `automaton_potential` | 47.40% | 49.74% | 86.72% |

| Contrast | points | 95% CI |
|---|---:|---|
| `STATE_GAP_FRAME_FULL__event_progress` | -18.49 | [-25.00, -12.50] |
| `STATE_GAP_OBS_HISTORY_FULL__event_progress` | +5.99 | [+1.30, +11.20] |
| `STATE_GAP_FRAME_FULL__automaton_potential` | -39.32 | [-47.40, -31.25] |
| `STATE_GAP_OBS_HISTORY_FULL__automaton_potential` | -36.98 | [-45.57, -28.65] |
| `FEEDBACK_GAP_FRAME_FULL__task5` | +43.23 | [+31.77, +54.69] |
| `FEEDBACK_GAP_OBS_HISTORY_FULL__task5` | +87.50 | [+79.17, +94.79] |
| `FEEDBACK_GAP_ORACLE__task5` | +1.56 | [-9.38, +12.50] |
