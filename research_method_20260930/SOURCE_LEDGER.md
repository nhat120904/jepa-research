# Source ledger — research recommendation, 30 September 2026

Scope: accepted main-conference work used for framing, design and prior-art challenges. CoRL is included as a leading robotics venue. Author manuscripts were read where they were easier to access; the linked proceedings establish publication venue. No unaccepted proposal is treated as a failed method. No new experiment or scheduler-state claim is made in these notes.

## Closest papers and design lessons

| Work / verified venue | Primary sources and material read | Relevance / constraint on the proposed contribution |
|---|---|---|
| One-Step Diffusion Policy (OneDP), ICML 2025 | [Proceedings](https://proceedings.mlr.press/v267/wang25ba.html), [author manuscript](https://arxiv.org/html/2410.21257) — distribution distillation and simulation/real-robot evaluation | KL along the diffusion chain trains a one-step generator. Evaluation covers six simulation and four real-robot tasks. A fast policy maintaining teacher quality is established prior art; use this as the strong base objective. Do not describe it as per-sample action MSE. |
| WPT, CVPR 2026 | [CVF proceedings](https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_WPT_World-to-Policy_Transfer_via_Online_World_Model_Distillation_CVPR_2026_paper.html), [author manuscript §3.4](https://arxiv.org/html/2511.20095v1#S3.SS4) — equations 15–16 and evaluations | Student query alignment and the reward of the teacher's best trajectory transfer WM knowledge; the deployed student removes the WM. nuScenes and Bench2Drive test planning and control. This directly blocks a broad novelty claim for WM-to-policy distillation. CVF metadata was confirmed by its indexed entry; direct fetching returned 403 during this session. |
| Fast Flow-based Visuomotor Policies via Conditional Optimal Transport Couplings (COT Policy), CoRL 2025 | [Proceedings](https://proceedings.mlr.press/v305/sochopoulos25a.html), [author manuscript](https://arxiv.org/html/2505.01179) — coupling construction and few-step controls | Conditional noise–action OT straightens flow trajectories. It is prior art for fast conditional policies using transport, but the objects being coupled differ from teacher–student executed outcomes. A context-conditioned OT loss alone is not a novelty claim. |
| Sparse Imagination for Efficient Visual World Model Planning, ICLR 2026 | [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/a750d52284ff70c6d6bab8072c392d74-Abstract-Conference.html), [author manuscript](https://arxiv.org/html/2506.01392) — randomized grouped attention and planning controls | Sparse prediction reduces planning tokens. The lesson is to verify native quality versus total compute and simple selection controls; generic sparse visual imagination is already published. |
| DINO-WM, ICML 2025 | [Proceedings](https://proceedings.mlr.press/v267/zhou25t.html), [author manuscript](https://arxiv.org/html/2411.04983v2) — spatial features, dynamics and planning evaluation | Predicting pretrained spatial visual features can support planning with a conventional search algorithm. This supports reusing a fixed feature space, but does not establish that its local derivatives are suitable for training a student policy. |
| Diffusion Policies Creating a Trust Region for Offline Reinforcement Learning (DTQL), NeurIPS 2024 | [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/59a48c111f97f2174709ea9ed8e920d1-Abstract-Conference.html), [author manuscript](https://arxiv.org/html/2405.19690) — diffusion trust region and one-step Q optimization | A diffusion behavior model constrains a fast policy while a value objective guides it. Evaluate against this mechanism when the setting supports its value learning; a Q-guided one-step policy is not new. Its state-based benchmarks do not replace visual manipulation evidence. |
| Planning in 8 Tokens / CompACT, CVPR 2026 | [CVF paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Kim_Planning_in_8_Tokens_A_Compact_Discrete_Tokenizer_for_Latent_CVPR_2026_paper.pdf), [author manuscript](https://arxiv.org/html/2603.05438) — tokenizer and downstream planning | Compact discrete observation tokens for efficient latent planning already have direct accepted precedent. Reusing CTA must be justified by its role in outcome supervision, not by a generic claim that compact codes make planning faster. |

## Accepted precedents that make several attractive pivots too broad

| Pivot | Primary accepted precedent | Why it was not selected as the recommendation |
|---|---|---|
| Add terminal value / TD reader | [TD-MPC2, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/cf73d57b6dcda32b293df7c2d5341f49-Abstract-Conference.html) | Short model rollouts with reward/value are established. A new reader needs a more specific mechanism and a verified control gap. |
| Goal-conditioned code/query factorization | [Contrastive Learning as Goal-Conditioned RL, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/e7663e974c4ee7a2b475a4775201ce1f-Abstract-Conference.html) | Contrastive representations already implement goal-conditioned value structure. A compact query reader by itself is insufficient differentiation. |
| Learn recovery continuation / policy-conditioned occupancy | [Compositional Planning with Jumpy World Models, ICML 2026](https://proceedings.mlr.press/v306/farebrother26a.html), [author manuscript](https://arxiv.org/html/2602.19634) | Policy-conditioned, multi-horizon occupancy prediction and compositional planning directly occupy this design space. |
| Hierarchical long-horizon subgoals | [HIQL, NeurIPS 2023](https://papers.neurips.cc/paper_files/paper/2023/hash/6d7c4a0727e089ed6cdd3151cbe8d8ba-Abstract-Conference.html) | A well-defined advantage-signal problem already motivates hierarchical latent goals. Simply moving CTA to hierarchy is not a hypothesis. |
| Distill/prune diffusion for on-device visuomotor control | [On-Device Diffusion Transformer Policy / LightDP, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Wu_On-Device_Diffusion_Transformer_Policy_for_Efficient_Robot_Manipulation_ICCV_2025_paper.html) | Efficient policy deployment has strong vision-conference precedent. “Smaller/fewer-step policy” alone is not the proposed contribution. |

These exclusions concern the breadth of novelty claims. They are not verdicts that improving any of these mechanisms cannot work. The recommendation remains a hypothesis, not a proof that the closest accepted papers leave a guaranteed publishable gap.

## Repository evidence checked

| Recorded evidence | Local source | Implication and limitation |
|---|---|---|
| CTAV2 score ~4.1 ms; policy sampling ~770 ms at the recorded K8/G1 setting | [CTA ledger, line 733](../trajectory_innovation_20260922/JOB_LEDGER.md) | Candidate generation dominates this configuration. This does not establish that a strong distilled student loses quality. |
| PushT200 development successes: P0 122, CTA4 142, CTAV2 140, ENDV2 138, DIRV2 135, DINO-WM 136 | [CTA ledger, line 730](../trajectory_innovation_20260922/JOB_LEDGER.md) | The pipeline works and there is a development gain over P0; separation from learned comparators remains uncertain. |
| Parameter-matched DIRECT retained gap .502 versus CTAV2 .680; many missed crossings recover later | [CTA ledger, lines 735 onward](../trajectory_innovation_20260922/JOB_LEDGER.md) | Better immediate ranking is real in these records; it is not a linear predictor of closed-loop gain. |
| K64 improves offline selection gain without a success improvement in the recorded 50-root comparison | [CTA ledger, line 760 onward](../trajectory_innovation_20260922/JOB_LEDGER.md) | More candidates alone is not an established route to stronger PushT control here. |
| Reacher 76.7→96.7 after valid history prefill; innovation adds no clear gain on the corrected baseline | [State-estimation ledger](../state_estimation_20260930/JOB_LEDGER.md) | Carry evaluator fixes into any future comparison. Do not claim the broken baseline gap as a learned-method contribution. |
| Cube informative-bank top-1 .85 predicted vs .97 true-endpoint, n=33 | [Bottleneck ledger](../bottleneck_ladder_20260930/JOB_LEDGER.md) | An offline WM-related gap exists. It does not imply an attainable 12-point closed-loop improvement. |
| Existing action-response modifications reduce selected losses without improving Cube control | [Action-response Cube ledger](../action_response_cube_20260928/JOB_LEDGER.md) | Improving action-response metrics is not enough. Outcome-distillation needs actual student control results. |

## Implementation feasibility inspected in source

| Asset | Source | What was verified / not verified |
|---|---|---|
| Native teacher sampling | [pusht_runtime.py](../trajectory_innovation_20260922/ti_wm/pusht_runtime.py) | PolicyRunner builds native observation conditioning and samples chunks. Sampling wrappers use inference_mode/NumPy, so they cannot carry student training gradients. |
| Tensor action normalization | [cta.py](../trajectory_innovation_20260922/ti_wm/cta.py) | Continuous torch operations preserve a possible action-gradient path. No runtime gradient test was run. |
| Parallel predicted codes | [cta_parallel.py](../trajectory_innovation_20260922/ti_wm/cta_parallel.py) | Expected coordinates come from softmax probabilities. Local physical derivative accuracy and decoder validity at expected codes remain unverified. |
| Future decoder | [cta.py](../trajectory_innovation_20260922/ti_wm/cta.py) | Decoder reconstructs visual future features from context/code, without direct candidate-action input. |
| Branch timing/stopping | [cta_runtime.py](../trajectory_innovation_20260922/ti_wm/cta_runtime.py) | Intermediate steps 2/4/6, endpoint8 and early termination/padding have explicit semantics. Student evaluation must preserve them. |

## Calendar and resources

[Official CVPR 2027 dates](https://cvpr.thecvf.com/Conferences/2027/Dates): paper registration November 10, 2026 AoE; paper submission November 16, 2026 AoE; supplementary November 23, 2026 AoE.

This session performed source/document inspection and web research. No model was loaded, no simulation/training was executed and no Slurm job was submitted. A future experiment must obey [AGENTS.md](../AGENTS.md): batch-only heavy work, explicit limits, peer checks, both squeue/sacct for live job-state claims, and the GPU/CPU/memory month-to-date cap including planned resource-hours at job time limits.
