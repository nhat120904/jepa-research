# Unified STATE rerun status

One frozen code/protocol, separate checkpoints for cube-triple/puzzle-4x5/scene. Scratch training on1000TRAIN+100play-validation episodes per family. This is lower than the previous3000episode cube/puzzle runs. Same WM/h/BC/support/search/control recipe; no task solver, finite-onlycache, LHBL or scripted executor. Native benchmark horizons differ.

Submitted jobs:preparation57815; trainingcube57816,puzzle57817,scene57818; evaluationseed6/7cube57819/57820,puzzle57821/57822,scene57823/57824; aggregation57825. Model/physics/dataalignment checks run in preparation, allnumerical work oncompute nodes. Source/config/checkpoints/partialepisodes/job IDs retained separately; full600episode aggregate only after complete evaluation. No new score yet.

Resources:4h training limit/job,2h evaluation limit/job; caps, not ETAs. Whole protocol fits90%-of-fifth monthlyGPU/CPU/memory rule; each GPU submission independently checked.

Remote root:`/mnt/data/nhatnc129/jepa/event_wm/unified_rerun_20261006`. Durable registry:job_registry.json; frozen protocol:protocol.json; source:source/. Event preparation:prep/{family}; training/evaluation:runs/{family}; aggregate:results.json andRESULTS.md when completed.

Latest dual scheduler verification: preparation57815COMPLETED5:21; cube57816/puzzle57817RUNNING; scene57818PENDING(QOSMaxGRESPerUser). All3 preparation checks passed. Actual TRAINevents10792/29474/17858, VAL1080/2950/1851, K3/20/5. BC release clipping required200puzzle/57sceneTRAINsegments. Actualgradient steps confirmed in logs. No learned closed-loop result from these runs yet. The GPU queue limit is external; historical scores are not new results.
