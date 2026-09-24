# Execution plan after the CompPlan discussion

2026-09-22. This addendum takes precedence where the original contract differs.
No method benefit or native policy reproduction has yet been measured.

## What CompPlan establishes, and what it does not

[CompPlan v1, Table 1 and section 4.1](https://arxiv.org/html/2602.19634v1)
supports the quoted Cube-4 and AntMaze-Giant numbers. Low zero-shot success does
not imply that useful local behaviors are absent. Conversely, a low CompPlan score
does not isolate absence of skills from proposal, model, horizon or objective error.
The five policy types are five experimental settings, not necessarily a bank that
mixes all five policies at inference. CompPlan searches sequences of parameterized
feedback policies and durations; sampling chunks from one fixed-goal diffusion
policy is a different, narrower planning interface.

We borrow the experimental separation of base capability and planning improvement.
We do NOT claim to implement CompPlan, transfer its headroom to PushT, or revive
the closed learned-composer experiments. PushT is an interface/control qualification,
not yet a sufficient test of novel task composition or query transfer.

## Locked next steps

1. CPU preparation: isolated dependency overlay, original LeRobot source revision,
   resolved checkpoint revision and hashes, strict configuration compatibility,
   native environment creation, action replay from reset including observations.
   No model instantiation/training or GPU required. Tests run on the compute node.
2. GPU smoke (only after inspecting preparation): strict checkpoint load, native
   policy rollout on four development roots, chunk API equivalence with official
   select_action, timing and bank diversity. This is not a baseline estimate.
3. Policy reproduction: 100 qualification roots, paired policy/oracle evaluation
   only after runtime equivalence is established. No abstractor training yet.
4. Proposal support: primary K=8; nested K=16/32 are predeclared sensitivity panels
   with shared candidate IDs/seeds. Do not change temperature or inject witnesses.
   A signal only at larger K is reported as such and charged its full latency.
   It is not a retrospective pass of the original K=8 gate.
5. Physical oracle and actual-future visual scorer, both offline and with repeated
   native-prefix decisions. Pin scorer BEFORE viewing bank outcomes. If no oracle
   helps, stop this control contract rather than calling the WM hypothesis false.
6. Only after qualification: observed-future codec test, action-predicted codec,
   same-bank frame WM and cached direct-query controls, then confirmation.

The initial budget remains <=8 GPU-hours and <=32 CPU-hours. Submit bounded jobs
one gate at a time; no dependent GPU job before the setup result has been reviewed.
The preparation script never edits existing environments or other research branches.

## Baselines and research question

History is not the justification for a world model. Policy-only with native history,
a longer-history policy (with matched extra-finetuning control), cached direct scorer,
full-frame WM, and abstraction must be distinguished. Longer-history policy training
is NOT part of initial qualification or assumed to be free: its bounded recipe/data
budget must be specified before a final WM-versus-policy claim. If that requires a
large VLA project, do not silently expand this project to build it.

Every scorer comparison uses identical observed history and the SAME frozen
proposal bank. If the proposer is later improved, repeat its support qualification
and give every scorer that improved proposer. Do not select a weak policy to create
an artificially favorable WM comparison. Report inference and training resources.

Question: at matched history/proposal support, does a self-supervised conditional
trajectory code offer a better query-retention/prediction-cost frontier than frame
codes and direct prediction, while preserving useful policy-guided control?
Task/query transfer requires a separately specified task family and a proposer that
accepts those goals. The fixed-goal released PushT policy does not establish this.

## Native interface facts discovered before running

- HF config: observation history 2, denoising tensor horizon 16, execute 8 actions,
  DDPM with 100 training/inference timesteps (num_inference_steps is null).
- Original source slices action indices [1:9] for execution. Horizon 16 includes
  one historical action slot: it is NOT 16 prospective executed actions. Initially
  evaluate the native eight-action prefix; any 15-future-action panel is separate.
- Original PushtEnv config sets episode limit 300 and physics/control fps 10.
- Current HF config contains runtime keys device/use_amp absent from the original
  policy dataclass. Record and remove only these runtime fields for dataclass decode;
  unknown architecture fields fail closed. Never silently ignore weight keys.
- gym-pusht 0.1.5 satisfies the original source minimum dependency. The exact author
  runtime is not pinned by that minimum; reproduction remains an empirical check.

Source: original LeRobot commit
`3c0a209f9fac4d2a57617e686a7f2a2309144ba2`, diffusion/modeling_diffusion.py
and common/envs/configs.py; HF lerobot/diffusion_pusht config and model card.
