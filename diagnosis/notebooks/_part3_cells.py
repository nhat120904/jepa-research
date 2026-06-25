"""Part 3 of regime_visualization.ipynb — cross-dataset regimes + diagnostic results.

Single source of truth for the Part 3 cells, shared by:
  * `_build_regime_viz_nb.py`  (full rebuild — emits these cells unexecuted)
  * `_append_part3.py`         (append + execute these onto the existing notebook,
                                preserving the already-executed Metaworld/DROID
                                frame galleries that can't be re-run on the cluster
                                box, which has no Metaworld parquet / no jupyter)

Every cell here is **frame-free** — it reads only the regime sidecars
(`*.regimes.json`), the latent caches (`*.h5`), and the result CSVs — so it runs
on any box with numpy/pandas/h5py/matplotlib (no torchcodec/decord/parquet).

`PART3` is a list of (kind, source) with kind in {"md", "code"}.
"""

PART3 = []


def _md(s):
    PART3.append(("md", s))


def _code(s):
    PART3.append(("code", s.strip("\n")))


# ---------------------------------------------------------------------------
_md(r"""---
# Part 3 · Other datasets & the diagnostic scaling curve

Parts 1–2 cover the two diagnostics with frame galleries (Metaworld, DROID).
This part adds the **remaining datasets** — the toy free-space tasks
(`pusht`, `point_maze`, `wall`), the small real-Franka set (`franka_custom`) — and
visualizes the **diagnostic results** across all of them, including the new
**DROID encoder-scaling curve** (22M → 300M → 1B).

All cells here are frame-free (sidecars + caches + result CSVs only), so they run
on the cluster box that has no Metaworld parquet / no Jupyter frame stack.""")

_md("## P3.0 · Setup (self-contained)")
_code(r"""
import sys, json, glob
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd, h5py
import matplotlib.pyplot as plt, matplotlib as mpl

mpl.rcParams["figure.dpi"] = 110
mpl.rcParams["axes.grid"] = True
mpl.rcParams["grid.alpha"] = 0.25

def _find_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "data" / "precomputed_latents").is_dir() and (p / "stratification").is_dir():
            return p
    raise FileNotFoundError("diagnosis/ root not found")
ROOT = _find_root(Path.cwd()); sys.path.insert(0, str(ROOT))
PL = ROOT / "data" / "precomputed_latents"
RESULTS = ROOT / "results"

REGIMES = ["free_space", "pre_grasp", "gripper_actuation", "contact_manipulation"]
REGIME_COLORS = {"free_space": "#4C72B0", "pre_grasp": "#DD8452",
                 "gripper_actuation": "#55A868", "contact_manipulation": "#C44E52"}
ID_TO_REGIME = {0: "free_space", 1: "pre_grasp", 2: "gripper_actuation", 3: "contact_manipulation"}

# dataset -> (sidecar model used for regimes, diagnostic CSV, model col in CSV)
DATASETS = {
    "metaworld":     ("dino_wm_metaworld",  "metaworld_diagnostic.csv",     "dino_wm_metaworld"),
    "droid":         ("dino_wm_droid",      "droid_diagnostic.csv",         "dino_wm_droid"),
    "franka_custom": ("dino_wm_droid",      "franka_custom_diagnostic.csv", "dino_wm_droid"),
    "pusht":         ("dino_wm_pusht",      "pusht_diagnostic.csv",         "dino_wm_pusht"),
    "point_maze":    ("dino_wm_pointmaze",  "point_maze_diagnostic.csv",    "dino_wm_pointmaze"),
    "wall":          ("dino_wm_wall",       "wall_diagnostic.csv",          "dino_wm_wall"),
}

def sidecar_for(dataset, model=None):
    if model:
        p = PL / f"{dataset}__{model}.h5.regimes.json"
        if p.exists(): return p
    hits = sorted(PL.glob(f"{dataset}__*.h5.regimes.json"))
    return hits[0] if hits else None

def regime_counts(dataset, model=None):
    sc = sidecar_for(dataset, model)
    if sc is None: return None
    d = json.load(open(sc))
    c = Counter()
    for v in d.values(): c.update(np.asarray(v).tolist())
    return pd.Series({ID_TO_REGIME[k]: c.get(k, 0) for k in range(4)}).reindex(REGIMES).fillna(0).astype(int)

print("root:", ROOT)
print("sidecars present:", [d for d in DATASETS if sidecar_for(d, DATASETS[d][0])])
""")

# --- P3.1 cross-dataset regime distribution --------------------------------
_md(r"""## P3.1 · Regime distribution across every dataset

How each dataset populates the four manipulation regimes. The **toy** tasks
(`pusht`/`point_maze`/`wall`) are pure `free_space` (2-D point dynamics, no
gripper/contact). **Metaworld** lacks a usable gripper signal. Only the real
robots (`droid`, `franka_custom`) populate all four — which is why DROID is the
load-bearing diagnostic for the contact/gripper regimes.""")
_code(r"""
present = [d for d in DATASETS if regime_counts(d, DATASETS[d][0]) is not None]
tbl = pd.DataFrame({d: regime_counts(d, DATASETS[d][0]) for d in present}).T[REGIMES]
display(tbl.assign(total=tbl.sum(1)))

frac = tbl.div(tbl.sum(1), axis=0).fillna(0)
fig, ax = plt.subplots(figsize=(9, 4.4))
bottom = np.zeros(len(frac))
for rg in REGIMES:
    ax.bar(frac.index, frac[rg], bottom=bottom, color=REGIME_COLORS[rg], label=rg)
    bottom += frac[rg].values
ax.set_ylabel("fraction of transitions"); ax.set_ylim(0, 1)
ax.set_title("Regime composition per dataset")
ax.set_xticklabels(frac.index, rotation=20, ha="right")
ax.legend(fontsize=8, ncol=2, loc="lower right"); plt.tight_layout(); plt.show()
""")

# --- P3.2 franka gripper-primary panel -------------------------------------
_md(r"""## P3.2 · `franka_custom` — gripper-primary regimes (cross-embodiment)

A second real robot (a different Franka teleop set). It is **small** (16 clips,
~112 transitions — below `min_transitions_per_cell`, so it yields no CRA verdict)
but it does exercise the *same* gripper-primary stratifier as DROID, so the
regime structure transfers: `gripper_actuation` owns the large-|Δgrip| tail, and
the gripper-closed line splits `contact` from the open-gripper regimes.""")
_code(r"""
FR_CACHE = PL / "franka_custom__dino_wm_droid.h5"
if FR_CACHE.exists():
    sc = json.load(open(str(FR_CACHE) + ".regimes.json"))
    fr = {k: np.asarray(v, np.int8) for k, v in sc.items()}
    rows = []
    with h5py.File(FR_CACHE, "r") as h:
        grp = h["trajectories"]
        for key in grp.keys():
            g = grp[key]; tid = g.attrs.get("traj_id", key)
            z = g["z"][:]; zf = z.reshape(z.shape[0], -1)
            grip = g["gripper"][:] if "gripper" in g else g["state"][:, 6]
            reg = fr.get(tid)
            if reg is None: continue
            for t in range(len(reg)):
                rows.append(dict(regime=ID_TO_REGIME[int(reg[t])], grip=float(grip[t]),
                                 grip_delta=float(abs(grip[t + 1] - grip[t])),
                                 dz=float(np.linalg.norm(zf[t + 1] - zf[t]))))
    fdf = pd.DataFrame(rows)
    FBASE = float(fdf.dz.median())
    display(fdf.regime.value_counts().reindex(REGIMES).fillna(0).astype(int).to_frame("n"))

    from stratification.droid_regimes import (
        GRIPPER_DELTA_THRESHOLD, GRIPPER_CLOSED_THRESHOLD, SCENE_CHANGE_RATIO)
    pres = [r for r in REGIMES if (fdf.regime == r).any()]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.3))
    data = [fdf.loc[fdf.regime == r, "grip_delta"].values for r in pres]
    parts = a1.violinplot(data, showmedians=True, widths=0.85)
    for pc, r in zip(parts["bodies"], pres): pc.set_facecolor(REGIME_COLORS[r]); pc.set_alpha(0.6)
    a1.axhline(GRIPPER_DELTA_THRESHOLD, color="k", ls="--", lw=1, label=f"grip Δ thr = {GRIPPER_DELTA_THRESHOLD}")
    a1.set_xticks(range(1, len(pres) + 1)); a1.set_xticklabels(pres, rotation=15, ha="right")
    a1.set_ylabel("|Δ gripper|"); a1.set_title("franka_custom — gripper actuation signal"); a1.legend()
    for r in pres:
        if r == "gripper_actuation": continue
        ss = fdf[fdf.regime == r]
        a2.scatter(ss.grip, ss.dz, s=18, alpha=0.6, color=REGIME_COLORS[r], label=r)
    a2.axhline(SCENE_CHANGE_RATIO * FBASE, color="k", ls="--", lw=1, label="scene-change baseline")
    a2.axvline(GRIPPER_CLOSED_THRESHOLD, color="dimgray", ls=":", lw=1.4,
               label=f"grip closed > {GRIPPER_CLOSED_THRESHOLD} → contact")
    a2.set_xlabel("gripper state (0=open .. 1=closed)"); a2.set_ylabel("‖Δz‖")
    a2.set_title("franka_custom — gripper-closed (contact) + scene split"); a2.legend(fontsize=8)
    plt.tight_layout(); plt.show()
else:
    print("franka_custom cache missing")
""")

# --- P3.3 dz across datasets -----------------------------------------------
_md(r"""## P3.3 · Latent change ‖Δz‖ per dataset

The diagnostic's effect threshold is τ = median ‖Δz‖ *within each dataset*. This
shows the ‖Δz‖ scale differs by encoder/dataset (so τ is dataset-relative), and
that the toy tasks have plenty of effectful (Δz > τ) transitions — the substrate
for their high CRA below.""")
_code(r"""
def dz_sample(dataset, model, cap_traj=120):
    cache = PL / f"{dataset}__{model}.h5"
    if not cache.exists(): return None
    vals = []
    with h5py.File(cache, "r") as h:
        grp = h["trajectories"]
        for i, key in enumerate(grp.keys()):
            if i >= cap_traj: break
            z = grp[key]["z"][:]; zf = z.reshape(z.shape[0], -1)
            vals.append(np.linalg.norm(np.diff(zf, axis=0), axis=1))
    return np.concatenate(vals) if vals else None

dz = {d: dz_sample(d, DATASETS[d][0]) for d in DATASETS}
dz = {d: v for d, v in dz.items() if v is not None and len(v)}
fig, ax = plt.subplots(figsize=(9, 4.2))
labels = list(dz.keys())
parts = ax.violinplot([dz[d] for d in labels], showmedians=True, widths=0.85)
for pc in parts["bodies"]: pc.set_alpha(0.55)
for i, d in enumerate(labels, 1):
    ax.text(i, np.median(dz[d]), f"τ={np.median(dz[d]):.0f}", fontsize=7, ha="center", va="bottom")
ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels, rotation=20, ha="right")
ax.set_ylabel("‖Δz‖ per transition"); ax.set_title("Latent change distribution by dataset (median = effect τ)")
plt.tight_layout(); plt.show()
display(pd.Series({d: float(np.mean(v > np.median(v)) * 100) for d, v in dz.items()},
                 name="% effectful (Δz>τ)").round(1).to_frame())
""")

# --- P3.4 DROID scaling curve (THE result) ---------------------------------
_md(r"""## P3.4 · DROID encoder-scaling curve — action-grounding does **not** scale

The headline diagnostic result, now a **4-point curve** over a 45× encoder
scale-up and two encoder families:

| model | encoder | params |
| --- | --- | --- |
| `dino_wm_droid` | DINOv2 ViT-S/14 | ~22M |
| `jepa_wm_droid` | DINOv3 ViT-L/16 | ~300M |
| `vjepa2_ac_droid` | V-JEPA-2 ViT-G/16 | ~1.0B |
| `vjepa2_ac_oss` | V-JEPA-2 ViT-G/16 (OSS) | ~1.0B |

Left: deterministic **`hard_nn` effect-conditioned CRA** per action-critical
regime — every model sits at the 16-way chance floor (0.0625). Right:
**`bb_boundary`** per regime — every model is boundary-blind (BB > 0), pre-grasp
the locus. Scaling the encoder does not close the gap.

*(We lead with `hard_nn`: it is deterministic, while `random`/`opposite` negatives
are unseeded and drift run-to-run.)*""")
_code(r"""
SCALE = [("dino_wm_droid", "DINO-WM 22M"), ("jepa_wm_droid", "JEPA-WM 300M"),
         ("vjepa2_ac_droid", "V-JEPA2-AC 1B"), ("vjepa2_ac_oss", "OSS 1B")]
MCOLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]
ACT_REGIMES = ["pre_grasp", "gripper_actuation", "contact_manipulation"]

dc = pd.read_csv(RESULTS / "droid_diagnostic.csv")
hn = dc[(dc.strategy == "hard_nn") & (dc.status == "ok")]
def eff(model, rg):
    r = hn[(hn.model == model) & (hn.regime == rg)]
    return float(r["cra_top1_eff"].iloc[0]) if len(r) else np.nan

bb = pd.read_csv(RESULTS / "droid_boundary.csv")
bb = bb[bb.status == "ok"]
def bbq(model, rg):
    r = bb[(bb.model == model) & (bb.regime == rg)]
    return float(r["bb_boundary"].iloc[0]) if len(r) else np.nan

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
x = np.arange(len(ACT_REGIMES)); w = 0.8 / len(SCALE)
for i, (m, lbl) in enumerate(SCALE):
    a1.bar(x + i * w, [eff(m, rg) for rg in ACT_REGIMES], w, label=lbl, color=MCOLORS[i])
a1.axhline(1 / 16, color="k", ls=":", lw=1.2, label="chance (1/16)")
a1.set_xticks(x + w * (len(SCALE) - 1) / 2); a1.set_xticklabels(ACT_REGIMES, rotation=12, ha="right")
a1.set_ylabel("hard_nn effect-CRA (top-1)"); a1.set_title("CRA stays at chance across 22M → 1B")
a1.legend(fontsize=8)

allr = REGIMES
for i, (m, lbl) in enumerate(SCALE):
    a2.bar(np.arange(len(allr)) + i * w, [bbq(m, rg) for rg in allr], w, label=lbl, color=MCOLORS[i])
a2.axhline(0, color="k", lw=0.8)
a2.set_xticks(np.arange(len(allr)) + w * (len(SCALE) - 1) / 2); a2.set_xticklabels(allr, rotation=12, ha="right")
a2.set_ylabel("bb_boundary (>0 = blind)"); a2.set_title("Boundary-blind at every scale (pre-grasp locus)")
a2.legend(fontsize=8)
plt.tight_layout(); plt.show()

display(pd.DataFrame({lbl: [eff(m, rg) for rg in ACT_REGIMES] for m, lbl in SCALE},
                     index=ACT_REGIMES).round(3).T.rename_axis("hard_nn effect-CRA"))
""")

# --- P3.5 cross-dataset positive control -----------------------------------
_md(r"""## P3.5 · Positive control — the metric returns **high** when actions matter

Across datasets, effect-conditioned CRA (`opposite` and the hard `hard_nn`) for
the primary model. The **toy free-space** tasks score near-ceiling (the world
models *do* encode the action when dynamics are simple and action-driven) — which
is exactly why the near-chance scores on DROID/Metaworld contact regimes are a
real action-grounding failure, not a broken metric. `franka_custom`/`robocasa`
are `insufficient_data` (too few transitions) and omitted.""")
_code(r"""
def wmean(g):
    w = g["n_effect"].clip(lower=1) if "n_effect" in g else np.ones(len(g))
    return np.average(g["cra_top1_eff"], weights=w)

rows = []
for ds, (_, csv, model) in DATASETS.items():
    p = RESULTS / csv
    if not p.exists(): continue
    t = pd.read_csv(p)
    if "cra_top1_eff" not in t.columns:
        rows.append((ds, np.nan, np.nan, "insufficient")); continue
    t = t[(t.model == model) & (t.status == "ok")]
    if not len(t): rows.append((ds, np.nan, np.nan, "insufficient")); continue
    opp = t[t.strategy == "opposite"]; hn = t[t.strategy == "hard_nn"]
    rows.append((ds, wmean(opp) if len(opp) else np.nan,
                 wmean(hn) if len(hn) else np.nan, "ok"))
summ = pd.DataFrame(rows, columns=["dataset", "opposite", "hard_nn", "status"]).set_index("dataset")
display(summ.round(3))

ok = summ[summ.status == "ok"]
fig, ax = plt.subplots(figsize=(9, 4.3))
x = np.arange(len(ok)); w = 0.38
ax.bar(x - w/2, ok["opposite"].values, w, label="opposite (easy neg)", color="#DD8452")
ax.bar(x + w/2, ok["hard_nn"].values, w, label="hard_nn (hard neg)", color="#C44E52")
ax.axhline(1 / 16, color="k", ls=":", lw=1.2, label="chance ≈ 1/16")
ax.set_xticks(x); ax.set_xticklabels(ok.index, rotation=20, ha="right")
ax.set_ylabel("effect-conditioned CRA (top-1)"); ax.set_ylim(0, 1)
ax.set_title("Effect-CRA by dataset — toy free-space near ceiling, real-robot contact near chance")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()
""")

# --- P3.6 sanity -----------------------------------------------------------
_md("## P3.6 · Part 3 sanity checks")
_code(r"""
checks = []
toy_counts = {d: regime_counts(d, DATASETS[d][0]) for d in ["pusht", "point_maze", "wall"]}
checks.append(("toy datasets are free_space-only",
               all(c is not None and c.drop("free_space").sum() == 0 for c in toy_counts.values())))
dr = regime_counts("droid", "dino_wm_droid")
checks.append(("DROID populates all four regimes", dr is not None and (dr > 0).all()))
fr = regime_counts("franka_custom", "dino_wm_droid")
checks.append(("franka_custom populates all four regimes", fr is not None and (fr > 0).all()))
dc = pd.read_csv(RESULTS / "droid_diagnostic.csv")
checks.append(("DROID CSV has all 4 scaling models",
               set(["dino_wm_droid","jepa_wm_droid","vjepa2_ac_droid","vjepa2_ac_oss"]).issubset(set(dc.model))))
hn = dc[(dc.strategy == "hard_nn") & (dc.status == "ok") & (dc.regime != "free_space")]
checks.append(("all 4 models ≤ 0.12 hard_nn effect-CRA in action regimes",
               (hn.groupby("model")["cra_top1_eff"].max() <= 0.12).all()))
for name, ok in checks:
    print(("PASS" if ok else "FAIL"), "-", name)
print(f"\n{sum(o for _, o in checks)}/{len(checks)} Part-3 checks passed")
""")
