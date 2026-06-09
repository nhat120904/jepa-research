# Diagnostic Implementation Plan v2 — CAI-JEPA Idea Validation

## Purpose

Determine whether the CAI-JEPA paper idea is worth pursuing by quantitatively measuring action grounding pathology in state-of-the-art JEPA world models on Franka manipulation tasks. The diagnostic answers one core question:

> **Do existing action-conditioned JEPA world models (DINO-WM, V-JEPA-2-AC, Terver-WM) exhibit measurable action grounding failures on quantitative counterfactual ranking, particularly in contact-rich Franka manipulation regimes where these models report low planning success?**

If yes → idea is worth 3-4 months of full implementation. If no → pivot or abandon.

This document is structured as actionable tasks for a coding agent. All model identifiers, paper references, dataset URLs, and task specifications are given explicitly.

---

## 1. Models, Papers, and Repositories — Complete Reference

### 1.1. The Three Baselines

All three baselines are available as pretrained checkpoints in the official `facebookresearch/jepa-wms` repository, published by Meta FAIR alongside the Terver et al. (2025) paper. This is critical because it means we have **fair, matched implementations** of all three models — same training data, same evaluation protocol — rather than reproducing from separate codebases.

#### Baseline 1: DINO-WM (Zhou et al., 2024)

- **Paper:** Zhou et al., "DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning"
- **arXiv ID:** 2411.04983
- **Original repo:** `https://github.com/gaoyuezhou/dino_wm`
- **Reproduced checkpoints (use these):** `facebookresearch/jepa-wms`
- **HuggingFace hub identifiers:**
  - `dino_wm_droid` (Real Franka, 256×256, DINOv3 ViT-L/16, predictor depth 12)
  - `dino_wm_metaworld` (Metaworld 42-task suite, 224×224, DINOv2 ViT-S/14, depth 6)
  - `dino_wm_pusht` (Push-T sanity check)
  - `dino_wm_pointmaze` (PointMaze sanity check)
  - `dino_wm_wall` (Wall navigation)

#### Baseline 2: V-JEPA-2-AC (Assran et al., 2025)

- **Paper:** Assran et al., "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning"
- **arXiv ID:** 2506.09985
- **Original repo:** `https://github.com/facebookresearch/vjepa2`
- **Fixed-version checkpoint (use this — original had eval bugs that Terver et al. fixed):** `facebookresearch/jepa-wms`
- **HuggingFace hub identifier:**
  - `vjepa2_ac_droid` (Real Franka DROID only; V-JEPA-2-AC was trained on DROID specifically)

#### Baseline 3: JEPA-WM / Terver-WM (Terver et al., 2025)

- **Paper:** Terver, Yang, Ponce, Bardes, LeCun, "What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?"
- **arXiv ID:** 2512.24497
- **Repo:** `https://github.com/facebookresearch/jepa-wms` (this is the same repo hosting the other baseline checkpoints)
- **HuggingFace hub identifiers:**
  - `jepa_wm_droid` (Real Franka, 256×256, DINOv3 ViT-L/16, depth 12, AdaLN conditioning + multistep rollout)
  - `jepa_wm_metaworld` (Metaworld 42-task, 224×224, DINOv2 ViT-S/14, depth 6)
  - `jepa_wm_pusht`
  - `jepa_wm_pointmaze`
  - `jepa_wm_wall`

Note: "Terver-WM" is informal terminology. The paper itself calls these models "JEPA-WMs". For consistency with the paper, refer to them as "JEPA-WM (Terver et al., 2025)" in code and writing.

### 1.2. Loading the Checkpoints

The unified loading API for all three baselines:

```python
import torch

# JEPA-WM (Terver et al., 2025)
model, preprocessor = torch.hub.load(
    'facebookresearch/jepa-wms', 
    'jepa_wm_droid'  # or jepa_wm_metaworld, etc.
)

# DINO-WM (reproduced by Terver et al.)
model, preprocessor = torch.hub.load(
    'facebookresearch/jepa-wms', 
    'dino_wm_metaworld'
)

# V-JEPA-2-AC (fixed version)
model, preprocessor = torch.hub.load(
    'facebookresearch/jepa-wms', 
    'vjepa2_ac_droid'
)
```

Alternative direct download from HuggingFace:
```python
from huggingface_hub import hf_hub_download
checkpoint_path = hf_hub_download(
    repo_id="facebook/jepa-wms",
    filename="jepa_wm_droid.pth.tar"
)
checkpoint = torch.load(checkpoint_path, map_location="cpu")
# Contains: 'encoder', 'predictor', 'heads', 'opt', 'scaler', 'epoch', 'batch_size', 'lr', 'amp'
```

### 1.3. Datasets

All datasets are hosted on the same HuggingFace organization: `huggingface.co/datasets/facebook/jepa-wms`

| Dataset | Config name | Robot | Visual | What it tests |
|---|---|---|---|---|
| **Metaworld** | `metaworld` | Sawyer (Franka-class) | 224×224 RGB | 42 tabletop manipulation tasks |
| **RoboCasa** | `robocasa` | Franka Panda | 256×256, multi-camera | Kitchen manipulation, multi-stage |
| **franka_custom** | `franka_custom` | Real Franka Panda | 256×256, 3 cameras | Real hardware tabletop |
| **DROID** | (external download) | Real Franka Panda | 720×1280 → 256×256 | Real-world manipulation diversity |
| **Push-T** | `pusht` | Planar end-effector | 224×224 | Sanity check (saturated) |
| **PointMaze** | `point_maze` | Point robot | 224×224 | Sanity check (saturated) |
| **Wall** | `wall` | Point robot | 224×224 | Navigation through doors |

Download script:
```bash
# From the jepa-wms repo
python src/scripts/download_data.py --dataset metaworld robocasa franka_custom
```

For DROID (separate gsutil download):
```bash
uv pip install gsutil
# Follow instructions in jepa-wms README; only left camera, no SVO files
gsutil -m cp -r gs://gresearch/robotics/droid_raw/... ./data/droid_raw/
```

---

## 2. Task Selection — Why These, Not Push-T

The previous version of this plan was too narrow. Push-T and PointMaze are inappropriate for the diagnostic because they are **saturated benchmarks** (DINO-WM achieves 90% on Push-T, 98% on PointMaze) with little room for action grounding pathology to manifest. Action grounding fails most visibly in **contact-rich, fine-precision Franka manipulation regimes**. Three task suites cover this range:

### 2.1. Primary Diagnostic: Metaworld (42 Franka tabletop tasks)

Metaworld is the optimal primary diagnostic target because:

- **Scale:** 42 distinct manipulation tasks in the JEPA-WM Metaworld checkpoint, all sharing the same Franka-class arm
- **Diversity:** Tasks range from easy (reach) to hard (peg-insert, assembly), giving natural difficulty stratification
- **Contact-rich:** Most tasks involve sustained contact (grasping, pushing, sliding objects)
- **MuJoCo ground truth:** Contact state, object positions, and gripper state are directly available, making regime stratification easy and reliable (unlike DROID which requires heuristics)
- **Pretrained checkpoints available** for both `jepa_wm_metaworld` and `dino_wm_metaworld`
- **Fast eval:** Smaller resolution (224×224) and ViT-S backbone means evaluation is 4-8× faster than DROID

**Selected diagnostic task subset (12 tasks across difficulty levels):**

Easy (expected high baseline performance — verify no false negatives):
- `reach-v2`
- `push-v2`
- `pick-place-v2`

Medium (contact-rich, expected moderate failures):
- `door-open-v2`
- `door-close-v2`
- `drawer-close-v2`
- `button-press-v2`
- `window-open-v2`

Hard (fine-precision, expected significant failures):
- `peg-insert-side-v2`
- `assembly-v2`
- `hammer-v2`
- `stick-pull-v2`

This 12-task subset captures the full difficulty spectrum while remaining computationally tractable. If baseline JEPA-WMs achieve >90% CRA on hard tasks (peg-insert, assembly), the action grounding hypothesis is empirically wrong and we should abandon. If they achieve <70% CRA on hard tasks, the hypothesis is strongly supported.

### 2.2. Secondary Diagnostic: DROID (Real Franka)

DROID provides validation that pathology found in Metaworld also exists in real-world data. The reasoning:

- **Real-world distribution shift:** DROID transitions are noisier, more diverse, and include real contact dynamics that simulation doesn't capture
- **Documented failure modes:** V-JEPA-2-AC paper itself reports grasp box at 25% SR and grasp cup at 65% SR on real Franka — these failure modes should manifest as low CRA in the diagnostic
- **Three baselines available:** `jepa_wm_droid`, `dino_wm_droid`, `vjepa2_ac_droid` all trained on the same data

**Diagnostic subset:** ~10K transitions from DROID validation split, stratified by task type (grasping, pushing, pick-and-place, manipulation with tools).

### 2.3. Tertiary Diagnostic: RoboCasa (Kitchen, Multi-Stage)

RoboCasa tests action grounding on **multi-stage long-horizon manipulation**, which exposes action grounding failures that accumulate over rollout (relevant for Metric 3, Counterfactual Trajectory Divergence). Use if Metaworld and DROID diagnostics are inconclusive or if time permits.

### 2.4. Sanity Checks (Not Main Diagnostic)

Push-T and PointMaze are run only to verify that the diagnostic implementation gives expected results on saturated baselines

These should never be reported as evidence for the paper's thesis.

---

## 3. Project Structure

```
cai_jepa_diagnostic/
├── README.md
├── pyproject.toml             # uv-managed dependencies
├── configs/
│   ├── diagnostic_metaworld.yaml
│   ├── diagnostic_droid.yaml
│   └── diagnostic_robocasa.yaml
├── data/
│   ├── metaworld/             # downloaded from facebook/jepa-wms HF
│   ├── droid_subset/          # gsutil download
│   ├── robocasa/              # downloaded from facebook/jepa-wms HF
│   └── precomputed_latents/   # cached encoder outputs per model
├── models/
│   ├── adapters/
│   │   ├── base.py            # WorldModelAdapter ABC
│   │   ├── dinowm_adapter.py
│   │   ├── vjepa2ac_adapter.py
│   │   └── jepawm_adapter.py
│   └── checkpoints/           # symlinked from torch.hub cache
├── metrics/
│   ├── cra.py                 # Counterfactual Ranking Accuracy
│   ├── aug.py                 # Action Usage Gap
│   ├── ecs.py                 # Effect-Conditional Sensitivity
│   ├── ctd.py                 # Counterfactual Trajectory Divergence (optional)
│   └── negative_samplers.py
├── stratification/
│   ├── metaworld_regimes.py   # uses MuJoCo ground truth
│   ├── droid_regimes.py       # uses heuristics + proprioception
│   └── robocasa_regimes.py
├── scripts/
│   ├── 01_setup_environment.sh
│   ├── 02_download_checkpoints.py
│   ├── 03_extract_latents.py
│   ├── 04_classify_regimes.py
│   ├── 05_run_diagnostic.py
│   ├── 06_analyze_results.py
│   └── smoke_test.py
├── results/
│   ├── metaworld_diagnostic.csv
│   ├── droid_diagnostic.csv
│   ├── robocasa_diagnostic.csv
│   ├── figures/
│   │   ├── figure_a_cra_per_regime.pdf
│   │   ├── figure_b_metaworld_per_task.pdf
│   │   └── figure_c_correlation_planning.pdf
│   └── decision_report.md
└── external/
    ├── jepa-wms/              # git clone facebookresearch/jepa-wms
    └── vjepa2/                # git clone facebookresearch/vjepa2 (reference only)
```

---

## 4. Phase 0: Environment Setup (Day 1, morning)

### Task 0.1: Clone the canonical repository

The `facebookresearch/jepa-wms` repo provides everything needed: checkpoints, datasets, baseline implementations, and the original counterfactual evaluation function.

```bash
git clone https://github.com/facebookresearch/jepa-wms.git external/jepa-wms
cd external/jepa-wms

# Use uv for environment management (the repo uses uv.lock)
uv sync
source .venv/bin/activate
```

If `torch.hub.load` returns 503 errors during checkpoint loading:
```bash
rm uv.lock
uv sync
```

### Task 0.2: Download all required checkpoints

Create `scripts/02_download_checkpoints.py`:

```python
import torch

CHECKPOINTS = [
    # Primary diagnostic: Metaworld
    ('facebookresearch/jepa-wms', 'jepa_wm_metaworld'),
    ('facebookresearch/jepa-wms', 'dino_wm_metaworld'),
    
    # Secondary diagnostic: DROID
    ('facebookresearch/jepa-wms', 'jepa_wm_droid'),
    ('facebookresearch/jepa-wms', 'dino_wm_droid'),
    ('facebookresearch/jepa-wms', 'vjepa2_ac_droid'),
    
    # Sanity check: Push-T
    ('facebookresearch/jepa-wms', 'jepa_wm_pusht'),
    ('facebookresearch/jepa-wms', 'dino_wm_pusht'),
]

for repo, hub_id in CHECKPOINTS:
    print(f"Loading {hub_id}...")
    model, preprocessor = torch.hub.load(repo, hub_id, trust_repo=True)
    print(f"  OK. Model class: {type(model).__name__}")
    print(f"  Preprocessor: {type(preprocessor).__name__}")
```

Run and verify all checkpoints load successfully before proceeding.

### Task 0.3: Download datasets

```bash
# From external/jepa-wms
python src/scripts/download_data.py --dataset metaworld robocasa pusht

# For DROID (larger, ~150GB)
# Follow the gsutil instructions in the jepa-wms README
gsutil -m cp -r gs://gresearch/robotics/droid_raw/1.0.1/*.mp4 ./data/droid_subset/
```

Storage estimate: Metaworld ~5GB, RoboCasa ~30GB, DROID ~150GB. Start with Metaworld; add DROID later if compute permits.

### Task 0.4: Smoke test

Create `scripts/smoke_test.py`:

```python
import torch
from PIL import Image
import numpy as np

# Load JEPA-WM on Metaworld
model, preprocessor = torch.hub.load(
    'facebookresearch/jepa-wms', 'jepa_wm_metaworld', trust_repo=True
)
model.eval().cuda()

# Create dummy 224×224 RGB observation
dummy_obs = torch.randn(1, 3, 224, 224).cuda()
dummy_action = torch.randn(1, 4).cuda()  # Metaworld uses 4-dim end-effector action

# Test encoder
with torch.no_grad():
    z = model.encoder(dummy_obs)
    print(f"Encoder output shape: {z.shape}")
    
    # Test predictor
    z_next = model.predictor(z, dummy_action)
    print(f"Predictor output shape: {z_next.shape}")

print("Smoke test PASSED")
```

If this fails, stop and debug — do not proceed to metric implementation.

---

## 5. Phase 1: Core Infrastructure (Day 1 afternoon - Day 2 morning)

### Task 1.1: Unified model adapter

```python
# models/adapters/base.py
from abc import ABC, abstractmethod
from typing import Tuple
import torch

class WorldModelAdapter(ABC):
    """Unified interface for DINO-WM, V-JEPA-2-AC, and JEPA-WM."""
    
    @abstractmethod
    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, C, H, W) → z: (B, N_tokens, D) or (B, D)"""
        pass
    
    @abstractmethod
    def predict(self, z_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        """One-step prediction. Returns same shape as z_t."""
        pass
    
    @abstractmethod
    def predict_rollout(
        self, z_t: torch.Tensor, actions: torch.Tensor  # (B, H, A)
    ) -> torch.Tensor:
        """H-step autoregressive rollout. Returns (B, H+1, ...)"""
        pass
    
    @abstractmethod
    def action_dim(self) -> int:
        pass
    
    @abstractmethod
    def normalize_action(self, a: torch.Tensor) -> torch.Tensor:
        """Apply the dataset's action normalization."""
        pass
    
    @abstractmethod
    def distance_for_planning(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Same distance function the model uses in CEM planning."""
        pass
```

Implement concrete adapters. The exact API of each model is in the `jepa-wms` repo source — inspect `src/models/` to find the predictor's forward signature. Document each adapter's preprocessing in `models/adapters/preprocessing_notes.md`.

### Task 1.2: Action normalization — the #1 source of bugs

Each model uses different action normalization. From the `jepa-wms` codebase:

- **Metaworld actions:** 4-dim (delta x, y, z, gripper), normalized to roughly [-1, 1]
- **DROID actions:** 7-dim (end-effector velocity + gripper), L1-ball constrained to radius 0.075
- **Push-T actions:** 2-dim planar velocity

To verify normalization is correct: load a transition from the dataset, encode obs_t and obs_{t+1}, predict z_{t+1} from (z_t, a_t), and compute the prediction error. The error should match the model's reported eval loss (within 2×). If error is 10× higher, action normalization is wrong.

### Task 1.3: Precompute and cache latents

```python
# scripts/03_extract_latents.py
# For each (model, dataset) pair:
#   For each trajectory:
#     Encode all frames once, save to HDF5
#     Also save: action_t, gripper_state, proprioception, contact_info (if available)
```

Storage estimate per model on Metaworld 12-task subset: ~10K trajectories × ~50 frames × latent dim. Roughly 5-20 GB per model.

Cache aggressively: encoding is the most expensive operation, and you'll iterate on metrics many times.

### Task 1.4: Reproduce a published number (sanity check)

Before trusting the metric implementation, verify that `jepa_wm_metaworld` achieves the planning success rates reported in Terver et al. (2025) Table 1 on at least 3 Metaworld tasks. Use the original `jepa-wms` evaluation script.

If reproduced numbers are off by more than 5%, debug before proceeding.

---

## 6. Phase 2: Implement Diagnostic Metrics (Day 2 afternoon - Day 3)

### Task 2.1: Negative action samplers

```python
# metrics/negative_samplers.py

def random_negative(a_t, action_bounds, K=16):
    """Uniformly sample K negatives within action_bounds.
    
    For Metaworld 4-dim: bounds = (-1, 1)^3 × (-1, 1) for gripper
    For DROID 7-dim: L1-ball radius 0.075
    """
    pass

def opposite_negative(a_t, sigma=0.1, K=16):
    """a^- = -a_t + N(0, sigma^2 I).
    
    For action spaces with gripper (Metaworld, DROID), flip gripper dim explicitly:
    a^-[gripper_dim] = 1 - a_t[gripper_dim]  (assuming gripper in [0, 1])
    """
    pass

def hard_nn_negative(z_t, a_t, candidate_pool, K=16, rho=0.5):
    """For each (z_t, a_t), find K transitions from candidate_pool with
    similar z (within distance rho) but maximally different a.
    
    candidate_pool: dict with 'z' (N, D) and 'a' (N, A) tensors from the batch
    
    Use FAISS or torch's topk for efficiency.
    """
    pass
```

### Task 2.2: Implement four metrics

Reference the paper proposal document (sections 4.2 — Metrics 1-4) for mathematical definitions. Key implementation notes:

- **CRA:** use stop-gradient on $z_{t+1}$ (target). Report both top-1 and MRR.
- **AUG:** stratify by regime; report per-regime AUG.
- **ECS:** calibrate `effect_threshold` per model — compute the median of $\|z_{t+1} - z_t\|$ over the eval set for that model. Use this median as the threshold.
- **CTD (optional):** multi-step rollout is expensive; defer if time-constrained.

### Task 2.3: Regime stratification

Metaworld stratification is clean because MuJoCo provides ground truth:

```python
# stratification/metaworld_regimes.py

def classify_metaworld_regime(transition_info: dict) -> str:
    """Returns one of: 'free_space', 'pre_grasp', 'gripper_actuation', 'contact_manipulation'"""
    
    gripper_t = transition_info['obs_t']['gripper_state']      # scalar in [0, 1]
    gripper_t1 = transition_info['obs_t1']['gripper_state']
    contact_t = transition_info['obs_t']['contact_indicator']  # MuJoCo contact
    contact_t1 = transition_info['obs_t1']['contact_indicator']
    
    # Gripper actuation: gripper state changes significantly
    if abs(gripper_t1 - gripper_t) > 0.2:
        return 'gripper_actuation'
    
    # Contact: object contact detected
    if contact_t or contact_t1:
        return 'contact_manipulation'
    
    # Pre-grasp: end-effector close to target object
    ee_pos = transition_info['obs_t']['ee_position']
    obj_pos = transition_info['obs_t']['target_object_position']
    if np.linalg.norm(ee_pos - obj_pos) < 0.10:
        return 'pre_grasp'
    
    return 'free_space'
```

DROID stratification uses heuristics (no contact ground truth):

```python
# stratification/droid_regimes.py

def classify_droid_regime(transition_info: dict, latent_t, latent_t1) -> str:
    gripper_t = transition_info['gripper_state_t']
    gripper_t1 = transition_info['gripper_state_t1']
    
    # Gripper actuation: obvious from proprioception
    if abs(gripper_t1 - gripper_t) > 0.2:
        return 'gripper_actuation'
    
    # Contact estimation via latent change magnitude
    # If z change is large relative to baseline and gripper is closed → likely contact
    latent_change = (latent_t1 - latent_t).norm()
    baseline_change = transition_info.get('global_median_latent_change', 1.0)
    
    if latent_change > 1.5 * baseline_change and gripper_t > 0.5:
        return 'contact_manipulation'
    
    # Pre-grasp: gripper open and moving in latent space
    if gripper_t < 0.3 and latent_change > baseline_change:
        return 'pre_grasp'
    
    return 'free_space'
```

### Task 2.4: Validate metrics with synthetic models

Before running on real models, test metric implementations with three synthetic baselines:

```python
class PerfectModel(WorldModelAdapter):
    """Returns true z_{t+1} (oracle). Expected: CRA=100%, AUG>>0."""
    def predict(self, z_t, a_t):
        return self._lookup_true_z_next(z_t, a_t)

class ActionIgnoringModel(WorldModelAdapter):
    """Returns z_t regardless of a_t. Expected: CRA≈chance, AUG≈0."""
    def predict(self, z_t, a_t):
        return z_t

class RandomModel(WorldModelAdapter):
    """Returns z_t + noise. Expected: CRA≈chance, AUG≈0."""
    def predict(self, z_t, a_t):
        return z_t + torch.randn_like(z_t) * 0.1
```

If any synthetic test fails, the metric is buggy. Fix before proceeding.

---

## 7. Phase 3: Run Full Diagnostic (Day 3 afternoon - Day 4)

### Task 3.1: Metaworld diagnostic (primary)

For each of the 12 selected Metaworld tasks (Section 2.1):

```
For each model in {jepa_wm_metaworld, dino_wm_metaworld}:
    For each negative strategy in {random, opposite, hard_nn}:
        For each regime in {free_space, pre_grasp, gripper_actuation, contact_manipulation}:
            Compute: CRA top-1, CRA MRR, AUG, ECS
            Save to results/metaworld_diagnostic.csv
```

Output columns:
```
task,model,strategy,regime,n_transitions,cra_top1,cra_mrr,aug,ecs,bootstrap_ci_low,bootstrap_ci_high
peg-insert-side-v2,jepa_wm_metaworld,hard_nn,gripper_actuation,1843,0.42,0.51,0.08,0.12,0.39,0.45
peg-insert-side-v2,jepa_wm_metaworld,hard_nn,contact_manipulation,2104,0.51,0.59,0.11,0.15,0.48,0.54
...
```

Sample sizes per (task, regime) cell: aim for ≥500 transitions. If a regime has <100 transitions on a task, mark as "insufficient data" and exclude from analysis.

### Task 3.2: DROID diagnostic (secondary)

```
For each model in {jepa_wm_droid, dino_wm_droid, vjepa2_ac_droid}:
    For each strategy in {random, opposite, hard_nn}:
        For each regime in {free_space, pre_grasp, gripper_actuation, contact_manipulation}:
            Compute all four metrics
            Save to results/droid_diagnostic.csv
```

Use ~10K-50K transitions from DROID validation. Stratification uses heuristics (Task 2.3).

### Task 3.3: RoboCasa diagnostic (tertiary, optional)

Same protocol on RoboCasa. Only run if Metaworld and DROID results are ambiguous, or if time permits.

### Task 3.4: Reproduce Terver's qualitative gripper test quantitatively

Use the `create_counterfactual_actions()` function from `external/jepa-wms/evals/unroll_decode/eval.py` to generate the exact "open gripper + up" vs "close gripper + up" counterfactual pair. Compute CRA on a large batch of cup-grasping scenarios from DROID using this specific counterfactual.

**Expected:** All three DROID models achieve CRA > 90% on this binary contrast, reproducing Terver et al.'s qualitative finding quantitatively. This establishes that the diagnostic captures the easy case correctly.

If models achieve only ~50% on this easy case, the diagnostic has a bug.

---

## 8. Phase 4: Analyze Results and Decide (Day 4 afternoon - Day 5)

### Task 4.1: Generate decision figures

**Figure A — Per-regime CRA across models (the key figure):**

Heatmap or grouped bar chart. X-axis: regime (4 categories). Y-axis: CRA top-1. Colored bars per model. Use the `hard_nn` negative strategy for this figure (hardest, most diagnostic).

Separate panels for Metaworld and DROID.

**Figure B — Per-task CRA on Metaworld (12 tasks):**

Bar chart showing CRA per task for JEPA-WM vs DINO-WM. Tasks sorted by JEPA-WM CRA. Identifies which specific tasks have the strongest action grounding pathology.

**Figure C — Correlation between CRA and Metaworld task SR (optional but valuable):**

Scatter plot. For each (task, model) pair, plot the model's hard_nn CRA on that task vs the model's planning success rate on that task (the published numbers from Terver et al. 2025 Table 1). Compute Pearson correlation. If $r > 0.6$, this is strong evidence that CRA predicts planning ability.

### Task 4.2: Decision logic

```python
def make_decision(metaworld_df, droid_df):
    # Focus on the most stringent test: JEPA-WM (best baseline) on hard NN negatives
    # in contact-rich regimes on hard tasks
    
    hard_tasks = ['peg-insert-side-v2', 'assembly-v2', 'hammer-v2', 'stick-pull-v2']
    contact_regimes = ['gripper_actuation', 'contact_manipulation']
    
    metaworld_critical = metaworld_df.loc[
        (metaworld_df.model == 'jepa_wm_metaworld') &
        (metaworld_df.strategy == 'hard_nn') &
        (metaworld_df.task.isin(hard_tasks)) &
        (metaworld_df.regime.isin(contact_regimes))
    ]
    
    droid_critical = droid_df.loc[
        (droid_df.model == 'jepa_wm_droid') &
        (droid_df.strategy == 'hard_nn') &
        (droid_df.regime.isin(contact_regimes))
    ]
    
    mw_cra = metaworld_critical['cra_top1'].mean()
    droid_cra = droid_critical['cra_top1'].mean()
    
    if mw_cra < 0.60 and droid_cra < 0.65:
        return 'GO', f'Strong pathology: MW CRA={mw_cra:.2f}, DROID CRA={droid_cra:.2f}'
    elif mw_cra < 0.75 or droid_cra < 0.75:
        return 'CONDITIONAL_GO', f'Moderate pathology in at least one dataset'
    elif mw_cra >= 0.85 and droid_cra >= 0.85:
        return 'ABANDON', f'No measurable pathology (MW={mw_cra:.2f}, DROID={droid_cra:.2f})'
    else:
        return 'PIVOT', 'Pathology exists but not where expected; reconsider task selection'
```

### Task 4.3: Sanity checks before acting on decision

1. **Easy-case CRA:** All models achieve CRA > 90% on `free_space` regime with `random` negatives. If not, metric is broken.
2. **Sanity check tasks:** All models achieve CRA > 90% on Push-T and PointMaze. If not, evaluation pipeline is broken.
3. **Model ordering:** JEPA-WM should generally have CRA >= DINO-WM (it's the better architecture). If DINO-WM consistently beats JEPA-WM, adapter or loading is wrong.
4. **Regime ordering:** CRA should generally be higher in `free_space` than `contact_manipulation`. If reversed, regime classification is mislabeled.
5. **Negative strategy ordering:** CRA should be highest with `random` and lowest with `hard_nn`. If reversed, negative samplers are buggy.
6. **Qualitative reproduction:** Terver's gripper open/close test (Task 3.4) gives CRA > 90% on all models.

If any sanity check fails, fix before reporting decision.

### Task 4.4: Generate decision report

`results/decision_report.md` should contain:

1. **Summary table** with critical CRA values
2. **Decision:** GO / CONDITIONAL_GO / PIVOT / ABANDON
3. **Justification** with specific numbers
4. **Figures A, B, C** embedded
5. **Sanity check log** showing all 6 checks passed
6. **Recommended next steps** based on decision

---

## 9. Decision Outcomes — What to Do Next

### If GO

The diagnostic confirms quantitative action grounding pathology on Franka manipulation. Proceed with the full 3-4 month implementation per the paper proposal.

**Critical assets to preserve:**
- All metric implementations (reusable as `CounterfactualBench`)
- Model adapters (reusable for full experiments)
- Regime classifiers
- Figure A becomes Figure 1 of the paper

### If CONDITIONAL_GO

Pathology is moderate. Two viable paths:

**Path A:** Drop the method contribution and reframe as a pure benchmark + correlation paper. The diagnostic protocol itself is a contribution (no quantitative protocol existed before). This is a safer publication path.

**Path B:** Extend the diagnostic to harder settings (LIBERO-Long multimodal demonstrations, RoboCasa multi-stage tasks) to find regimes where pathology is stronger, then proceed with full implementation focused on those regimes.

### If PIVOT

Pathology exists somewhere but not where expected. Investigate:

- If pathology is in `free_space` only: uninteresting, abandon.
- If pathology grows with horizon (high CTD at H=10 but low CRA at H=1): pivot to long-horizon planning angle.
- If pathology is dataset-specific (only DROID, not Metaworld): consider real-world-only paper.

### If ABANDON

Baseline JEPA-WMs are well-grounded on the diagnostic. The paper as proposed won't show improvements. Pivot to:

- VLA-JEPA action collapse (different setting documented in VLA-JEPA paper, Feb 2026)
- Pure methodology paper proposing CounterfactualBench as a community benchmark, without method
- Different research direction entirely

---

## 10. Critical Implementation Notes (Updated)

### Note 1: Action normalization

Each model expects normalized actions. Specifically:
- **JEPA-WM Metaworld:** 4-dim actions in approximately [-1, 1]. Check `external/jepa-wms/src/datasets/metaworld.py` for the exact normalization.
- **JEPA-WM DROID / V-JEPA-2-AC:** end-effector commands constrained to L1-ball radius 0.075 (≈13cm max displacement). Actions outside this ball are out-of-distribution.
- **DINO-WM:** matches the JEPA-WM normalization in the reproduced version.

Test by reproducing one (z_t, a_t) → z_{t+1} prediction on a known training transition. Prediction MSE should match the model's reported eval loss within 2×.

### Note 2: Use the same distance function as the model's planner

All three models use slightly different distance functions for CEM planning. Inspect `external/jepa-wms/src/planning/` to find the exact function. Use the same distance in CRA to ensure ranking accuracy predicts planning behavior.

### Note 3: V-JEPA-2-AC normalization quirk

The V-JEPA-2-AC model trained by Meta had an action normalization bug fixed by Terver et al. (hence the "fixed" version `vjepa2_ac_droid` in the jepa-wms repo). **Always use this fixed version**, not the original V-JEPA 2 release, for fair comparison.

### Note 4: Hard negative mining

For each transition, finding K=16 hard negatives within a batch of 256 requires 256×256 distance computations. For Metaworld (small latent dim), this is fast. For DROID (large latent dim, ViT-L), use FAISS or batched topk on GPU.

Alternative: precompute hard negatives once per (model, dataset) pair and cache to disk. Each transition gets a list of K hard negative action indices. This avoids redundant computation across CRA strategies.

### Note 5: Bootstrap confidence intervals

For all reported CRA values, compute bootstrap 95% CIs over the eval set. With 1000 samples per regime, CI width is ~3 percentage points. With 100 samples, CI width is ~10 points. The decision logic in Task 4.2 should use CI-aware comparisons, not point estimates.

### Note 6: Compute budget

| Phase | Time | GPU |
|---|---|---|
| 0 — Setup | 0.5 day | 1 GPU |
| 1 — Infrastructure | 1.5 days | 1 GPU for latent extraction |
| 2 — Metrics | 1 day | minimal |
| 3 — Run diagnostic | 1.5 days | 2 GPUs parallel (Metaworld + DROID) |
| 4 — Analyze | 0.5 day | none |
| **Total** | **5 days** | **1-2 GPUs continuous** |

Storage: ~50GB Metaworld + 150GB DROID + 50GB precomputed latents = ~250GB total.

### Note 7: Reuse for full implementation

If decision is GO, the diagnostic code becomes infrastructure for the full paper:
- `metrics/` → CounterfactualBench (Contribution 1)
- `stratification/` → regime classifiers for the paper
- `models/adapters/` → unified interface for adding CAI-JEPA as a 4th model
- Bootstrap and figure code → reused for paper figures

Architect the code with this reuse in mind. Avoid hardcoded paths; use config files.

---

## 11. Final Deliverables Checklist

After 5 days, the coding agent should produce:

- [ ] `results/metaworld_diagnostic.csv` — 12 tasks × 2 models × 3 strategies × 4 regimes = 288 rows
- [ ] `results/droid_diagnostic.csv` — 3 models × 3 strategies × 4 regimes = 36 rows (or more if stratified by task type)
- [ ] `results/figures/figure_a_cra_per_regime.pdf` — main decision figure
- [ ] `results/figures/figure_b_metaworld_per_task.pdf` — task-level breakdown
- [ ] `results/figures/figure_c_correlation_planning.pdf` — CRA vs SR correlation (optional)
- [ ] `results/decision_report.md` — formal report with GO/CONDITIONAL_GO/PIVOT/ABANDON
- [ ] `results/sanity_checks_log.md` — all 6 sanity checks documented as passed
- [ ] `models/adapters/` — three unified model adapters (reusable)
- [ ] `metrics/` — metric implementations (reusable)
- [ ] `stratification/` — regime classifiers (reusable)

If the decision is GO, items 7-9 are direct inputs to the 3-4 month implementation phase.

---

## 12. Plan adjustments (v2.1 — 2026-06-01, after reading the real upstream source)

The original plan guessed parts of the `facebookresearch/jepa-wms` API. After
cloning the repo (`diagnosis/external/jepa-wms`) and reading the source, the
following corrections were made. Full rationale: `diagnosis/docs/plans/2026-06-01-real-api-rewrite-design.md`.

**Corrected facts about the upstream:**

- The model is `EncPredWM` (wraps `VideoWM`); drive it via `EncPredWM.encode`
  (raw `[0,255]` visual in → `(B,T,V,H,W,D)` latent) and `EncPredWM.unroll`
  (the planner's primitive), **not** `.encoder`/`.predictor` directly.
- Datasets are in `app/plan_common/datasets/` (`MetaworldHFDataset`,
  `DROIDVideoDataset`, `RoboCasaDataset`), returning `(obs, act, state[, reward,
  info])` with `obs={"visual","proprio"}` — **not** `src/datasets/*` with a dict
  schema.
- Action normalization is `preprocessor.normalize_actions` (plural), stats from
  hardcoded `DATA_STATS`. The old `normalize_action` (singular) silently no-oped
  → the #1 bug. Now fixed and gated by `scripts/check_normalization.py`.
- All planning configs use **L2** distance (`*_L2_cem_*`) — CRA uses L2 for all
  baselines (the earlier "cosine for JEPA-WM" note was wrong).

**Methodology adjustments (Section 2.1 / 4.2 / 4.3 amendments):**

- **Metaworld stratification is a proxy, not MuJoCo contact GT.** The HF dataset
  ships no contact flags; it ships the 39-dim `state` (ee/object/goal positions).
  We derive regimes from `state` and use **object displacement** as the contact
  proxy. Metaworld remains the primary target; the "clean MuJoCo GT" claim in
  §2.1 is downgraded to "structured-state proxy".
- **Primary decision metric is effect-conditioned CRA** (CRA on transitions with
  `‖Δz‖>τ`), with raw CRA and CTD as support. The §4.2 decision logic is now
  CI-aware: ABANDON requires the upper CI bound to be confidently high, so a
  noisy 1-step number can't trigger abandonment (§8 Task 4.2 updated in code).
- **Bootstrap is trajectory-clustered**, not iid over transitions (§ Note 5),
  because within-trajectory transitions are correlated.
- Proprioception is threaded through prediction where the checkpoint uses it
  (DROID checkpoints are `_noprop`).

**Execution note:** the diagnostic code + metric unit tests are complete and run
offline (`pytest diagnosis/tests`, 23 green). The GPU/data path (real
checkpoints, Metaworld/DROID downloads, full pipeline) runs on a server per
`diagnosis/RUNBOOK.md`.

## 13. Execution status & doc map (2026-06-04)

**Where the study stands:**

- **Metaworld (primary): DONE.** Full sweep on real `dino_wm_metaworld` +
  `jepa_wm_metaworld` (12 tasks × 3 strategies × 4 regimes) →
  `diagnosis/results/metaworld_diagnostic.csv`, decision **CONDITIONAL_GO**. The
  gap is visible: `opposite` CRA ~0.97–0.99 (passes the qualitative-style test)
  but `hard_nn` effect-conditioned CRA ~0.46–0.57 in pre-grasp/contact; chance ≈
  0.059; `jepa_wm > dino_wm` consistently. Metaworld `gripper_actuation` is empty
  (HF release has no gripper signal).
- **DROID (secondary): SET UP & CACHED, metric step pending.** Built on the
  user's 8GB-VRAM box: lean venv, a hand-built **333-episode** public subset
  (IRIS+TRI, wrist cam, fps=4/fpc=8), latents encoded (`03`), regimes
  recalibrated (`04`, all 4 cells populated incl. `gripper_actuation` 16% and
  `contact_manipulation` 18% — the cells Metaworld can't fill). `05`/`06` are
  **not yet run**: the GPU fell off the bus mid-`05` (hardware/driver Xid, not a
  code defect). Finish on a healthy GPU or with `eval.device: cpu` — latents are
  cached, so no re-encode/re-download.
- **New since v2.1:** a 4th negative strategy **`hard_effect`** (similar-state +
  most-different *true* effect, preferring near-by actions) — a "fair-hard"
  precision negative that self-selects toward contact; in `metrics/negative_samplers.py`,
  wired through `05`, enabled in the DROID config.
- **Models on 8GB:** only `dino_wm_droid` runs; `vjepa2_ac_droid` (ViT-G ~1B)
  doesn't fit and `jepa_wm_droid` needs gated DINOv3 `.pth` weights (HF ships only
  safetensors). See `diagnosis/docs/HANDOFF_DROID.md` §1.

**Doc map (read in this order):**

1. `diagnosis/docs/METHODOLOGY.md` — concepts + code map + the
   dataset/task/regime/strategy matrix and *how each proves the gap*.
2. `diagnosis/docs/plans/2026-06-01-real-api-rewrite-design.md` — real upstream API.
3. `diagnosis/docs/HANDOFF.md` (Metaworld) / `diagnosis/docs/HANDOFF_DROID.md`
   (DROID) — operational run/finish handoffs.
