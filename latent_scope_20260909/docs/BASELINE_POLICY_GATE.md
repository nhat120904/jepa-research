# Official policy gate before Stage A4

## Purpose

Test whether the strongest directly applicable released policy is useful enough to
produce successful rollouts in the current RoboCasa `1.0.1` runtime. This is an
execution gate, not evidence for or against the compositional-JEPA hypothesis.

## Locked policy and scope

- official RoboCasa GR00T N1.5 checkpoint after target post-training on
  `composite_seen`, checkpoint 60000;
- exact official source and checkpoint revisions are pinned in
  `configs/baseline_policy_gate.json`;
- only inference files are downloaded (about 7.59 GB); optimizer and trainer state are
  excluded;
- native `target` split, official horizons and success predicates;
- exactly 10 episodes each for `ScrubCuttingBoard` and `RinseSinkBasin`;
- official `panda_omron` transforms, 16-action chunks and four denoising steps;
- no training and no altered task definitions.

The runtime uses two isolated environments on the same Slurm node. The policy server
keeps the official GR00T Torch stack; the simulator client keeps the RoboCasa/LeRobot
stack. They exchange observations and actions through GR00T's local ZeroMQ protocol.
This preserves each side's declared versions instead of forcing mutually incompatible
tianshou/protobuf and LeRobot/torchvision dependencies into one environment. The setup
preflight imports both complete paths before permitting evaluation.
The simulator side retains RoboCasa/LeRobot's declared `gymnasium==0.29.1`; GR00T's
`gymnasium==1.0.0` remains confined to the policy environment and is not imported by the
standalone simulator client.

The official 40.6% composite-seen number is an aggregate suite result and is recorded
only as context. It is not treated as the expected rate for either selected task.

## Decision rule

Pass only if each task obtains at least 2 successes in 10 episodes. A pass means the
policy is useful enough to support a bounded A4 rollout/calibration design; it does not
automatically authorize Stage B. A failure retires this RoboCasa pairing for the current
proposal rather than triggering another policy or compatibility search.

Environment or model-load errors are reported as infrastructure errors and are not
interpreted as a negative scientific gate.
