# Demo report ledger

2026-10-05: submitted CPU export job **57577**, main, 2 CPU, 16 GB, 20-minute limit, no GPU. Output `/mnt/data/nhatnc129/jepa/event_wm/demo_report_57577`. No peer report export was running at submission. Sources preserve all earlier runs.

Month-to-date snapshot before submission: nhatnc129 67 GPU h, 449 CPU h, 5,137,430 MB-h. Fifth user thresholds: GPU 100 h, CPU 993 h, memory 7,094,397 MB-h. Existing two jobs have about 10.32 combined GPU h remaining at their time limits; the new single CPU export adds at most 0.67 CPU h and about 5,461 MB-h. Including current remaining allocations stays below the 90% ceilings (90 GPU h, 893.7 CPU h, 6,384,957.3 MB-h).

Status and artifact verification to be appended after completion.

57577 COMPLETED 56 s, exit 0; all event / rollout exports valid, but fresh reset rendering for task 3/4 was corrupted. 57578 COMPLETED 3 s: initial covered-bit quantization and complete static step export; invalid task 3/4 inputs retained, not reportable. 57579 COMPLETED 9 s with LP_NUM_THREADS=1: task 3 valid, task 4 goal still corrupt due to renderer lifetime. OGBench Env.close does not release the renderer; dropping the old environment after constructing the next one invalidates OSMesa context. 57580 COMPLETED 7 s, exit 0: reuse one env/renderer across tasks, validate pixel mean/dtype/shape/black fraction. Both task 3/4 image pairs visually inspected. All jobs confirmed with squeue and sacct; no GPU, arrays or new closed-loop execution. Final report uses only job57580 assets. Earlier outputs preserved for audit.

Final UI verification: event clip plays; task selector and event-step controls update WM diagrams; video seek reaches replan step482 at playback24.1s using byte-range preview server. No console errors observed. Asset links and 8/30 reference totals verified; all selected event / planning / rollout files present. Portable ZIP excludes intermediate invalid jobs and contains only job57580 assets. Browser preview kept open as deliverable.
