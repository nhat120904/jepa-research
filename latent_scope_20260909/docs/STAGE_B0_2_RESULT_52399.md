# Stage B0.2 result: job 52399

Job 52399 completed in 02:03:42 with exit code 0. The direct event snapshot restored
simulator state and native task history exactly, all candidate carriers matched their
canonical restored observations, and all candidate post-states were diverse.

| Task | Intervention | Candidate successes | Outcome varies | Candidate-0 to oracle gain |
|---|---:|---:|---:|---:|
| ScrubCuttingBoard | 8 steps | 7/8 | yes | 0 pp |
| ScrubCuttingBoard | 16 steps | 8/8 | no | 0 pp |
| RinseSinkBasin | 8 steps | 8/8 | no | 0 pp |
| RinseSinkBasin | 16 steps | 8/8 | no | 0 pp |

This is the first valid evidence that candidate identity can change native eventual
completion in the selected arena: one Scrub H8 candidate failed while seven succeeded.
It is not positive headroom under the predeclared comparison because candidate 0 already
succeeded. Every branch reached maximum recorded progress, so progress headroom was also
zero at these near-completion event anchors.

The profile projects 82.39 serial GPU-hours for 40 prefixes per task, above the locked
24 GPU-hour tranche. Do not launch that serial expansion or Stage C. The next justified
step is to improve branch throughput and then evaluate several independent, non-saturated
prefixes without changing the candidate-0 definition.
