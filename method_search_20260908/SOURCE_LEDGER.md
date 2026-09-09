# Primary-source ledger

Checked 2026-09-08. Publication labels below distinguish accepted proceedings from preprints and workshops. Older papers are included because they can invalidate novelty even when the initial survey emphasized 2025–2026. Presence of code does not mean reproduction.

| Source | Status / evidence | Consequence for this proposal |
|---|---|---|
| [COT-GAN](https://papers.nips.cc/paper_files/paper/2020/hash/641d77dd5271fca28764612a028d9c8e-Abstract.html) | NeurIPS2020 proceedings; technical PDF read | Causal transport objectives and adapted critics are established |
| [Conditional COT-GAN](https://arxiv.org/abs/2106.05658) | 2021 preprint; [author code](https://github.com/neuripss2020/kccotgan) includes BAIRPushSmall | Conditional sequential generation is a direct comparator; code availability only, no reproduction |
| [Wasserstein Believer](https://proceedings.iclr.cc/paper_files/paper/2024/hash/d0261e5705d6ec8d54fa1b3c22b69baf-Abstract-Conference.html) | ICLR2024 proceedings; paper read | OT-based belief learning and value-related analysis already exist |
| [Optimal Transport World Models](https://delgrange.me/publication/ropke-2025-integrating-rl-planning-ot-world-models/) | ALA2025 workshop, author publication page | Do not claim first use of OT in latent-model planning |
| [Estimating processes in adapted Wasserstein distance](https://arxiv.org/abs/2002.07261) | Mathematical foundation; author paper | Filtration matters for sequential optimization; empirical estimation is nontrivial |
| [Scalable Bi-causal OT via KL Relaxation and Policy Gradients](https://arxiv.org/abs/2605.17271) | May2026 preprint, no acceptance claimed | New bicausal solvers are also occupied territory |
| [Variational Deficiency Bottleneck](https://arxiv.org/html/1810.11677v2) | Author manuscript, original submission2018; no ICLR acceptance inferred | Deficiency-based representation learning and variational objectives already exist |
| [Le Cam Distortion](https://arxiv.org/html/2512.23617v1) | Dec2025 preprint, no acceptance verified | Directional observation simulation with control applications is close prior art. Its empirical MMD proxies are not automatically TV-deficiency certificates; its broad claims were not independently validated here |
| [Learning active tactile perception through belief-space control](https://arxiv.org/abs/2312.00215) | 2023 preprint; title also present in [ICRA2025 publisher contents](https://www.proceedings.com/content/081/081087webtoc.pdf) | Learned observation/dynamics, filtering and information-gathering MPC already combined in robotics; runnable author implementation not verified here |
| [BayesContact](https://arxiv.org/abs/2607.16123) | July2026 preprint | Active contact inference is a strong task-specific prior; do not claim its mechanism as new |
| [Tree-structured Policy Planning](https://arxiv.org/abs/2301.11902) | ICRA2023, confirmed by [author publications](https://www.borisivanovic.com/publications.html) | Learned predictions plus contingency trees already exist outside manipulation |
| [MIND](https://arxiv.org/abs/2408.13742) | 2024 manuscript; acceptance not checked here | Interaction-conditioned scenario branching is additional prior art |
| [ContactNets](https://corlconf.github.io/corl2020/paper_506/) | CoRL2020 accepted paper | Learning contact structure is established |
| [Neural Hybrid Automata](https://proceedings.neurips.cc/paper/2021/hash/5291822d0636dc429e80e953c58b6a76-Abstract.html) | NeurIPS2021 proceedings | Learned modes and event transitions are established |
| [Saltation matrices overview](https://doi.org/10.1109/JPROC.2024.3440211) | Proceedings of the IEEE2024 | Event-time sensitivities have specific hard-event assumptions |
| [Saltation-consistent event-aware digital twins](https://www.nature.com/articles/s41598-026-53809-5) | Scientific Reports2026 | Further overlap for an event-sensitivity proposal; not a robotics top-conference acceptance |
| [PH-Dreamer](https://arxiv.org/abs/2605.18303) | May2026 preprint | Generic port-Hamiltonian world-model transplant is occupied |
| [Parareal with a Learned Coarse Model for Robotic Manipulation](https://arxiv.org/abs/1912.05958) | 2019 manuscript; [2020 journal article](https://link.springer.com/article/10.1007/s00791-020-00327-0) | Learned multi-fidelity time propagation for manipulation is established |
| [IMBench](https://arxiv.org/html/2607.15641v1), [project](https://imbench.org/) | July2026 preprint and official project | Relevant task descriptions/data verified; usable simulator repository not verified |
| [ManiSkill task documentation](https://maniskill.readthedocs.io/en/latest/tasks/table_top_gripper/index.html) | Official docs | Public robot task substrate; proposed information-gathering variants are not stock tasks |
| [Tactile Gym](https://github.com/ac-93/tactile_gym) | Public author code | Alternative tactile simulator; dependency age and native-task information requirements need checking before selection |

## Local evidence revisited

- `/Users/nhatcuong/code_project/vin-research/moment_wm_h0/docs/DECISION_REPORT.md`: Gate4 STOP, recoverable hidden drag but no true-planning benefit from the tested regularizer. This is evidence about that implementation, not a universal negative result.
- `../reselection_20260908/METHOD_REFRAMING.md`: earlier two questions, now narrowed by this audit.
- `../reselection_20260908/REPORT_VI.md`: efficiency proposal previously withdrawn as primary.

## Search boundary

Searches covered causal/adapted transport, conditional sequential generators, belief-model learning, information-gathering control, scenario-tree prediction, Blackwell/Le Cam representation learning and transfer, hybrid contact models, and candidate simulator task availability. No search can certify the absence of prior art. No claim is made to have reviewed every accepted 2025–2026 paper.

## Next novelty check before substantial implementation

The precise remaining claim is **learning action-dependent sequential observation abstractions with constructive causal policy adapters**, beyond existing belief learning and directional sensor simulation. If the proposed estimator reduces to an existing VDB/COT/WBU objective with actions appended, treat the method as an application and reconsider its suitability for the user's goal. A new name is not a novelty result.
