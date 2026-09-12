# Stage A result: `REPLAY_DIVERGED`

Stage A completed on 10 September 2026 in Slurm job `51835`. The harness processed all
ten selected demonstrations and wrote a complete result file. Slurm records the job as
`FAILED` with exit code `2:0` because the harness deliberately returns exit code 2 when
an execution gate fails; this was not another cluster or rendering failure.

## Gate results

| Gate | Observation | Result |
|---|---|---|
| Assets and simulator startup | Revision-pinned assets loaded and all ten episodes ran under CPU OSMesa | PASS |
| Released action-to-state replay | All ten episodes first exceeded tolerance at action step 0; maximum released-state error was 11.558760525351506 | **FAIL** |
| Native task success | `ScrubCuttingBoard`: 4/5; `RinseSinkBasin`: 1/5; total 5/10 | **FAIL** |
| Required camera observations | All three required 256x256 RGB streams were present and nonconstant for every episode | PASS |
| History signal | The monitored history fields changed for both tasks | PASS |
| Snapshot/suffix repeatability | Reward, done flag, image keys, and dynamics keys were stable, but numerical simulator state, images, and dynamics were not restored exactly; `ScrubCuttingBoard` task state matched in only 2/5 episodes | **FAIL** |

The step-0 errors range from order 1 to order 10, so relaxing the locked `1e-5`
tolerance would not turn this into a numerical-drift pass. The released action playback
and the released state trajectory are presently incompatible under this execution path.

## Decision

Do not start Stage B training or planning comparisons. Stage A has not qualified the
execution substrate, so a downstream model result would not be interpretable. This does
not falsify the proposed compositional JEPA method; it is an execution blocker upstream
of that scientific test.

The next bounded experiment should be an **A2 replay-alignment diagnostic** on one
episode per task, with rendering disabled where possible. It should identify whether the
step-0 mismatch comes from action ordering/controller metadata, action/state indexing,
or an incompatible playback convention by comparing action playback against direct
recorded-state playback. Snapshot capture/restore coverage should be audited separately
before any suffix or counterfactual comparison is trusted.

Full machine-readable result:
`/mnt/data/nhatnc129/jepa/latent_scope_stage_a/outputs/replay_51835/stage_a_result.json`.
