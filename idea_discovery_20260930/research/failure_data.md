# Failure monitoring and demonstration selection

Scan date: 30 September 2026. Accepted primary papers motivate the research questions; explicitly marked preprints constrain novelty. Literature results are not our reproductions. Root executes all compute.

## Accepted papers

| Work | Existing mechanism and implication |
|---|---|
| [Sentinel, CoRL2024](https://arxiv.org/html/2410.04640v2), [official release](https://github.com/agiachris/sentinel) | Overlapping action-distribution consistency plus VLM progress. A policy may consistently do the wrong thing. Offline outputs permit direct comparison without a policy checkpoint. |
| [FAIL-Detect, RSS2025](https://www.roboticsproceedings.org/rss21/p073.html) | Learns uncertainty/density from successful references and calibrates functional thresholds. Success-only training is not training-free. |
| [SAFE, NeurIPS2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/392d0d05e2f514063e6ce6f8b370834c-Abstract-Conference.html) | Policy-feature readouts, transfer and manually annotated failure-time evaluation. Onset-aware evaluation itself is not novel. |
| [NWPU Campus, CVPR2023](https://openaccess.thecvf.com/content/CVPR2023/html/Cao_A_New_Comprehensive_Benchmark_for_Semi-Supervised_Video_Anomaly_Detection_and_CVPR_2023_paper.html) | Predicts future errors for anomaly anticipation. Future-window labels include ongoing anomalies unless onset is separated. |
| [LAVAD, CVPR2024](https://openaccess.thecvf.com/content/CVPR2024/html/Zanella_Harnessing_Large_Language_Models_for_Training-free_Video_Anomaly_Detection_CVPR_2024_paper.html) | Frozen multimodal models provide anomaly scores. Centered windows and full-video refinement must be distinguished from causal monitoring. |
| [DemInf, RSS2025](https://www.roboticsproceedings.org/rss21/p023.html) | Mutual-information-style estimates balance action diversity and predictability in learned state/action spaces; generic predictable-demo selection is established. |
| [DataMIL, ICLR2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/033d9e8dbbbc0ad90e59222cf1db0fc2-Abstract-Conference.html) | Selects data by target-policy learning utility. Actual utility requires downstream policy training/control, not merely a quality-label correlation. |
| [ReMix, CoRL2024](https://proceedings.mlr.press/v270/hejna25a.html) | Reference-policy excess loss informs mixtures; easy data and unlearnable noise can both have low marginal utility. |
| [CUPID, CoRL2025](https://cupid-curation.github.io/) | Uses influence information linked to rollout returns. Smoothness and usefulness are not interchangeable. |
| [VIP, ICLR2023](https://github.com/facebookresearch/vip) | Released goal-progress representation is a useful quality-control baseline; use a valid successful goal rather than every demo's own endpoint. |

## Closest 2026 collisions (preprints, not accepted-paper framing)

- [FARM](https://arxiv.org/html/2609.11445v1): freezes V-JEPA2 and VLA-JEPA, trains a small failure readout, uses causal histories and cross-policy/platform evaluations. Broad frozen predictive-state failure monitoring is already claimed. Its [code](https://github.com/HaoranPei-casia/FARM) was a coming-soon notice during inspection.
- [Foresight](https://arxiv.org/html/2606.23085v1): frozen visual backbone with a newly trained action-conditioned predictor and causal detector. Full-rollout maximum AUROC alone does not prove pre-onset warning.
- [Gauge](https://arxiv.org/html/2602.16182v1): learned predictive/reconstruction errors for robotic inspection failure.
- [Instruction–trajectory auditing](https://arxiv.org/html/2608.07895v1): multimodal quality auditing; trained/transductive state representations complicate a broad training-free claim.

## Feasible bounded probe

The official [PushChair ZIP](https://drive.google.com/uc?id=1UW2lxqQiu90NTDsPmqxx8wVZ7aSlp504) contains 10 successful calibration episodes, 10 successful test episodes and 10 failed test episodes, with RGB, native timestamps/actions and precomputed STAC/VLM outputs. Size269,809,141bytes; no6GB policy checkpoint required. Data were transferred via desktop because Drive was unreachable from cluster, then extracted in a CPU batch.

The primary question is whether an external frozen action-free WM gives useful evidence **beyond feature change and RGB motion**. All scores use current/past arrived images; prediction residual is timestamped when its target actually arrives. It cannot warn based on a residual against an unseen future. V-JEPA2-AC's DROID/Franka action conventions do not match PushChair actions, so no fake action conversion is used.

Reported metrics: same20test-episode balanced detection accuracy under success-only calibration; max-score outcome AUROC; fixed3/6/9/12-second prefix AUROC only where all episodes reach the cutoff. Episodes ending early must not be silently excluded. No failure-onset labels exist in this release; prefix outcome forecasting must not be renamed pre-onset warning. Full-rollout published Sentinel performance is already high, so an overall detector improvement is not assumed to have headroom.

## Why curation is not a third no-training task verdict

Verified public assets: [Lift-MH images](https://downloads.cs.stanford.edu/downloads/rt_benchmark/lift/mh/image.hdf5),2.684GB; [paired Can images](https://downloads.cs.stanford.edu/downloads/rt_benchmark/can/paired/image.hdf5),1.716GB. MH contains6operators×50successful demos, with100/tier. PairedCan contains100matched-initialization good-placement/bad-throw pairs. HEAD returned200 during source research; these datasets were not downloaded in this sprint.

A frozen scorer could measure paired quality preference without training. That would not establish imitation-data utility: the relevant final comparison is matched student closed-loop success after training on selected data. This region was therefore deferred rather than counted as a failed no-training experiment.
