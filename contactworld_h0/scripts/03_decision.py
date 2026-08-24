"""Phase 0 decision report: collate probe + latent-probe JSONs into DECISION.md."""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "outputs"
THRESH = 1.0  # cm, ContactWorld success tolerance on plug_pos

TASKS = ["exploration_search", "insertion_usb"]
COND_LABEL = {
    "LIN_proprio": "linear readout, proprio only (no history)",
    "A_pc_now": "pointcloud, current frame",
    "B_pc_prop_now": "pointcloud + proprio, current frame",
    "C_pc_prop_hist": "pointcloud + proprio, 8-step history",
    "D_pc_prop_tac_hist": "pointcloud + proprio + TACTILE, 8-step history",
    "E_prop_tac_hist": "proprio + TACTILE, 8-step history (no vision)",
    "F_prop_hist": "proprio only, 8-step history",
}


def load(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def fmt_ci(v):
    return f"[{v[0]:.2f}, {v[1]:.2f}]"


lines = ["# ContactWorld Phase 0 — decision report", ""]
lines += [f"Success tolerance the benchmark itself uses on `plug_pos`: **{THRESH:.0f} cm** "
          "(`eval_planner.py:313`). All errors below are held-out, split by episode, "
          "episode-clustered bootstrap CIs, 3 seeds.", ""]

gate_rows = []
for task in TASKS:
    d = load(OUT / f"probe_{task}_plugpos.json")
    if d is None:
        lines += [f"## {task}", "", "_probe results not available yet_", ""]
        continue
    lines += [f"## {task} ({d['n_episodes']} episodes)", "",
              "### Raw-observation readout of object state (upper bound on available information)", "",
              "| condition | mean err (cm) | 95% CI | hit@1cm |", "|---|---|---|---|"]
    for k, lab in COND_LABEL.items():
        if k not in d["conditions"]:
            continue
        c = d["conditions"][k]
        lines.append(f"| {lab} | {c['mean_err_cm']:.2f} | {fmt_ci(c['mean_err_cm_ci'])} | {c['hit_at_thresh']:.3f} |")
    lines.append("")

    lines += ["### Paired effect tests (same held-out windows)", "",
              "| test | delta (cm) | 95% CI | CI excludes 0 |", "|---|---|---|---|"]
    for key, lab in [("tactile_effect_D_minus_C", "tactile added to vision+proprio history"),
                     ("tactile_effect_E_minus_F", "tactile added to proprio history (no vision)"),
                     ("vision_effect_C_minus_F", "pointcloud added to proprio history")]:
        if key not in d:
            continue
        e = d[key]
        lines.append(f"| {lab} | {e['delta_mean_err_cm']:+.3f} | {fmt_ci(e['delta_ci_cm'])} | "
                     f"{'**yes**' if e['ci_excludes_zero'] else 'no'} |")
        if key == "tactile_effect_D_minus_C":
            gate_rows.append((task, e["delta_mean_err_cm"], e["delta_ci_cm"], e["ci_excludes_zero"]))
    lines.append("")

    lines += ["### Frozen world-model latent readout (what the CEM planner actually scores)", "",
              "| checkpoint | readout | mean err (cm) | 95% CI | hit@1cm |", "|---|---|---|---|---|"]
    for tag, lab in [("pc", "pointcloud only"), ("pc_ff", "pointcloud + TacFF")]:
        L = load(OUT / f"latent_{task}_{tag}.json")
        if L is None:
            continue
        for r, rd in L["readouts"].items():
            lines.append(f"| {lab} | {r} | {rd['mean_err_cm']:.2f} | {fmt_ci(rd['mean_err_cm_ci'])} | "
                         f"{rd['hit_at_thresh']:.3f} |")
    lines.append("")

lines += ["## Gate verdict", ""]
lines += ["Gate condition (pre-registered): *GO only if tactile history measurably reduces "
          "task-relevant uncertainty.*", ""]
for task, delta, ci, excl in gate_rows:
    verdict = "REDUCES" if (excl and delta < 0) else ("INCREASES" if (excl and delta > 0) else "NO EFFECT")
    lines.append(f"- **{task}**: tactile delta {delta:+.3f} cm, CI {fmt_ci(ci)} -> **{verdict}**")
lines.append("")

lines += ["""
## Interpretation

Two tasks, two different bottlenecks. Tactile is not the fix in either.

| task | object state in raw obs | object state in WM latent | implied bottleneck |
|---|---|---|---|
| insertion_usb | present (0.23 cm) | mostly present (0.58 cm, 88% under 1 cm) | downstream of perception: cost / planner |
| exploration_search | present only via proprio (1.23 cm) | absent (3.5-4.0 cm) | representation / input modality |

**1. Tactile carries object-state information, but only where vision already works.**
On insertion_usb, adding tactile to proprio-only history is a real gain
(-0.25 cm, CI [-0.28, -0.22]) -- so TacFF is not noise. But adding it on top of the
pointcloud is a null (-0.006 cm, CI [-0.02, 0.01]). It is redundant with vision.
On exploration_search, where the pointcloud readout is 3.81 cm and effectively useless,
tactile does *not* step in: every tactile contrast there is a tight null.
So tactile does not compensate for vision exactly where compensation would be needed.

**2. The information the benchmark's own metric needs is often not in the latent the planner scores.**
ContactWorld's success test thresholds plug_pos at 1 cm. On exploration_search the frozen
latent reads out plug_pos at 3.5-4.0 cm, i.e. the CEM cost is computed in a space that cannot
resolve the object to within the tolerance it is being scored against.

**3. The world model never sees proprioception.**
`train.py:73` loads only `["action", vision_key]` (+ tactile); the planner's history buffers
(`planner_utils.py:285-300`) carry only vision (+tactile). Yet on exploration_search a readout
from proprioceptive history alone reaches 1.23 cm, three times better than the latent.

**4. On insertion_usb, perception is NOT the bottleneck.**
The latent already resolves plug_pos to 0.58 cm with 88% of frames under the 1 cm threshold,
while published planning success at this modality/horizon is far below 88%. Whatever fails
there fails after perception -- consistent with a cost/planner-side failure rather than a
missing-information failure.

## Caveat that bounds all of the above

Every number here is measured on **demonstration** trajectories. Part of the proprio readout's
accuracy plausibly comes from the demonstrator's behaviour correlating with object pose (the arm
goes where the object is), which need not survive under CEM-generated motion. Testing that
requires executing off-policy actions in the simulator, which is blocked by the missing NVIDIA
graphics stack (see docs/GATE_0_INFRA.md). Until that is unblocked, "add proprioception to the
world model" is the best-supported lead, not an established fix.
"""]

Path(OUT / "DECISION.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
