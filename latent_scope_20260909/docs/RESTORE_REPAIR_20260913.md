# Restore repair after 52410

The pilot 52410 failed its replay gate after 143 allocated GPU-seconds, before
any candidate evaluation. Its initial observations/state matched, and two fresh
restores matched one another, but source-versus-restored suffixes differed by
up to 0.594189719 in flattened simulator state. This is not method evidence.

CPU-only diagnostic 52416 replays the saved source actions, compares compiled
MuJoCo model arrays before/after the XML round trip, and tests transferring those
arrays as a controlled intervention. It does not load a policy or request a GPU.

52416 completed in 1m43s. The recreated source anchor and original reference
suffix matched exactly. The XML-restored model differed in body masses/inertias,
derived mass terms, visual settings and mesh precision, among other arrays.
Replacing numerical arrays alone reduced suffix state error to 0.533213395 but
did not pass. This is evidence that the XML restore path is not an identity,
not a complete attribution of every source of error to one array.

The v2 implementation serializes the compiled MuJoCo model and full MjData with
native binding serialization. A new carrier follows the usual RoboCasa metadata,
controller and observable initialization lifecycle, but loads the saved compiled
model instead of recompiling edited XML; native MjData deserialization restores
full dynamics while preserving the robosuite data wrapper and its references.
The MuJoCo runtime version must match. These are trusted local binary artifacts.

CPU job 52417 tests untouched live versus v2 restores at H8/H16, two independent
fresh-carrier repeats, and a compiled-model-only ablation. Its verified source
will be reused as attempt zero only if the unchanged gate passes. GPU submission
and the batch script both require a successful CPU proof; the pilot repeats the
gate on its own node. No old failed result is overwritten.

52417 exposed an installed-binding incompatibility: `mj_copyData` is listed in
the C API metadata but is not exposed by this Python build. The implementation
now uses the MjData pickle round trip exercised by the installed MuJoCo binding
tests, rather than calling that unavailable function. This failed CPU run used
no GPU allocation.

## Verified correction

CPU job **52418** passed. The saved source was recreated exactly. Against the
untouched live suffix, both H8 and H16 had **zero flattened-state error and zero
pixel error**, with equal histories, labels, rewards and clocks. Two fresh
compiled restores also matched exactly. The compiled-model-only ablation passed
with zero errors too, so full MjData is a conservative addition, not a demonstrated
necessary component of this fix. Native model serialization, unlike the XML
round trip, preserves the tested carrier's transition behavior.

Proof: `/mnt/data/nhatnc129/jepa/latent_scope_baseline/outputs/compiled_restore_52418.json`.
Verified source: `/mnt/data/nhatnc129/jepa/latent_scope_baseline/outputs/compiled_restore_52418.verified_source.pkl`.

The snapshot is about 1.1 GB. Completed-prefix snapshots are retained on disk and
loaded for confirmation as needed, rather than accumulating every model in RAM.
The pilot still rechecks every source on its own compute node. One passing source
does not certify all other scenes, nor does replay validity demonstrate headroom.

## Pilot restart

GPU job **52419** submitted after successful CPU jobs 52418 and 52409; hard limit
07:45:00, N8/H8, eight target prefixes with independent-seed confirmation. Maximum
combined GPU allocation with failed 52410 is 28,043 seconds (7h47m23s), below the
original eight-hour tranche cap. No other GPU retry was submitted.

Result: `/mnt/data/nhatnc129/jepa/latent_scope_baseline/outputs/qualification_52419/qualification_result.json`.
Log: `/mnt/data/nhatnc129/jepa/latent_scope_baseline/logs/qualification_52419.out`.
Verify both squeue and sacct on the next status check. Submission is not completion;
there is no foreground monitoring loop and no Stage-C training job.

One independent test defect was found and fixed: `collect_source` must run the
reference suffix on the untouched source carrier, without calling restore first.
The old 52410 artifact remains unchanged, including its reference provenance.

Remaining GPU ceiling for the originally approved tranche is 28,657 seconds
(8 hours minus the 143 seconds consumed by 52410). A subsequent 07:45:00 job fits
inside that ceiling. CPU diagnostics consume no allocated GPU-hours.

Do not relax the original replay tolerances or reinterpret a failed calibration
as a positive candidate-selection result. No pilot retry until the correction
has passed saved-action compute-node replay validation.
