# Discovery sprint job ledger

All experimental source is exploratory. No training is performed. Heavy work is sbatch-only. Remote source root: `/home/nhatnc129/nhat.nc/jepa-research/idea_discovery_20260930`; artifacts: `/mnt/data/nhatnc129/jepa/idea_discovery_20260930`.

## 2026-09-30 (Asia/Saigon)

Before preparation, both squeue and sacct checked: no running jobs for account nhatnc129. Month-to-date sreport: GPU 217h; CPU 1817h; memory 14,702,641 MB-h. Fifth-user totals, independently sorted per resource: GPU 486h, CPU 3890h, memory 50,118,738 MB-h. Allowed 50% thresholds: 243 GPU-h, 1945 CPU-h, 25,059,369 MB-h. These are point-in-time readings, not permission for later submissions without recheck.

| Job | Work | Resources/time limit | Verified state |
|---|---|---|---|
| 56170 | Public IntPhys2 subset, TAP-Vid-DAVIS, Sentinel PushChair and released tracker asset preparation | CPU main, 2 CPUs, 8GB, 25min | FAILED in 1s, exit127; uv absent from batch PATH. No GPU used. Fixed explicit uv path before retry. |
| 56173 | Asset preparation retry | CPU main, 2 CPUs, 8GB, 25min | COMPLETED 6m41s. 93 complete IntPhys2 scenes/372 videos, TAP-Vid-DAVIS1.67GB archive, CoTracker3 online101.7MB weight. Drive inaccessible from cluster, so Sentinel transferred through desktop from the same public URL; unzip remains compute-node work. |
| 56178 | IntPhys2 frozen-model discovery subset: 24 complete scenes, 96 videos; global/centered/local surprise | MIG GPU1,4CPU,64GB,45min | COMPLETED 2m41s, exit0. Strict all-keys model loading passed. Modified protocol; three principles covered, solidity absent. |
| 56190 | Tracking first attempt | MIG GPU1,4CPU,64GB,45min | FAILED 3s, exit1: missing mediapy, before model loading. |
| 56191 | Extract transferred official PushChair archive | main CPU2,8GB,5min | COMPLETED 6s, exit0. |
| 56192 | Install mediapy in task dependency directory | main CPU1,4GB,5min | COMPLETED 2s, exit0. |
| 56193 | PushChair: 10 calibration successes + 20 test episodes; frozen WM / motion / released STAC | MIG GPU1,4CPU,64GB,45min | COMPLETED 40s, exit0. Strict checkpoint loading passed. |
| 56194 | Tracking retry | MIG GPU1,4CPU,64GB,45min | FAILED 3s, exit1: missing IPython, before model loading. |
| 56206 | Partial result summary and episode bootstrap | main CPU1,4GB,5min | COMPLETED 1s, exit0. Preserved as results/summary_partial.json. |
| 56218 | Python-3.10-compatible IPython installation and dependency import preflight | main CPU2,8GB,5min | COMPLETED 47s, exit0; IMPORT_PREFLIGHT_OK. |
| 56223 | Tracking, five deterministic videos | MIG GPU1,4CPU,64GB,45min | COMPLETED 37s, exit0. **Debug only:** observed features lacked non-affine target normalization; excluded from selection. |
| 56224 | Expand old tracking implementation | MIG GPU1,4CPU,64GB,45min | CANCELLED 36s; stopped immediately when feature-space mismatch was found. Partial results excluded. |
| 56226 | Corrected tracking on all 30 TAP-Vid-DAVIS videos | MIG GPU1,4CPU,64GB,45min | COMPLETED 1m45s, exit0. Target normalization and source metadata fixed; strict checkpoint loading passed. |
| 56227 | Final summary, video/episode bootstrap, timing totals | main CPU2,4GB,5min | COMPLETED 0s at scheduler resolution, exit0. results/summary_final.json. |

Upstream source pins: IntPhys2 `7f3f589b81168049c6330d077976ac4e76ef7d46`; V-JEPA2 `204698b45b3712590f06245fbfba32d3be539812`; CoTracker `82e02e8029753ad4ef13cf06be7f4fc5facdda4d`; Sentinel `6dc89ca4cb30f75ce834811c2d1b31f7ffadf7b5`. Model checkpoint reused from existing official HF original Giant256 cache; strict key/shape loading is required, no random fallback. Current upstream hub localhost URL is bypassed by direct cached loading.

## Final verification and reproducibility

All final states above were checked with **both squeue and sacct**. Final squeue contained no jobs for nhatnc129. Total allocated GPU time, including failed/cancelled/debug jobs: **385 seconds = 0.10694 GPU-h** on one MIG allocation per GPU job. This is not a full-H100 throughput-equivalent claim.

Every GPU submission was preceded by the month-to-date ranking guard and a duplicate-work check. Latest quota: GPU217h, CPU1817h, memory14,703,140MB-h; fifth-user totals remained486h/3890h/50,118,738MB-h. Planned full allocation was +0.75GPU-h, +3CPU-h, +49,152MB-h; all checks passed. Future work must recheck these values.

Python: `/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python` (3.10). Task extras: artifact root `/extra`. Giant checkpoint path: `/mnt/data/nhatnc129/jepa/ossckpt/vjepa2_opensource/vjepa2_vit_giant.pth`; resolved official HF blob SHA256 filename `67129f011434e605d894e69f2c8e13d9db118deabe59d54bf6e0fa62c2c5cb8e`. Tracker: official `scaled_online.pth`,101,695,610bytes.

Source SHA256 for the final inference jobs:

| Source | SHA256 |
|---|---|
| scripts/common_model.py | cc10e8e3d857f37dd515f3d11dfdd1e67cbf08cdd710e6ceb8c9ac00235741e1 |
| scripts/probe_physics.py | 26ef04785926a0d0cce5677315014fd5ea6b5a1923acc1611e8dac3c4f494c1b |
| scripts/probe_failure.py | 66afa9d5ec8a4816312c4dd448f37b7cce5e403de6df5465bad80e45db8e8424 |
| corrected scripts/probe_tracking.py | 673ca1b83146f039e64c4baef72bc9537f3e3496b0c6be36565c1aa8c33faec2 |

Old tracking summaries incorrectly reported enclosing research-repo commit `2159e67` as upstream revisions because `.git` was excluded from rsync. This provenance bug is disclosed, not silently rewritten. Job56226 reads explicit `.upstream-revision` pins. The summary marks old tracking runs `eligible_for_selection:false`.

Run configurations and artifacts:

- Physics56178: defaults max-scenes24, frame-step8, max-windows8; 16-frame clips, 12-frame context. Local results/physics_56178/result.json; original per-video windows and manifests remote under runs/physics_56178. This is a modified pilot, not the published48-frame multi-context protocol.
- Failure56193: all10calibration/20test; native timestep<40; score direction high=failure fixed; threshold uses successful calibration only. Local results/failure_56193/failure_probe.json. Precomputed STAC scores reused; STAC policy inference was not rerun.
- Tracking56226: --max-videos30; default stride4 plus query-frame observations, seven context anchors, radius4patches. Local results/tracking_56226/summary.json; prediction arrays remote under runs/tracking_56226/predictions.
- Summary56227: scripts/summarize_results.py --runs /mnt/data/nhatnc129/jepa/idea_discovery_20260930/runs --out /mnt/data/nhatnc129/jepa/idea_discovery_20260930/summary_final.json. Local results/summary_final.json preserves older debug runs.

PushChair assets.json retains the original cluster Drive-download error. Recovery used the same public archive via desktop network transfer, followed by compute-node extraction. No simulator, model loading, encoding or bulk analysis ran on the login node or desktop.
