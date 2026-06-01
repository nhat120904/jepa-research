# Per-model preprocessing notes (verified against upstream source)

These notes were rewritten after reading the **real** `facebookresearch/jepa-wms`
source (cloned to `external/jepa-wms/`). The previous version guessed the API
and was wrong on every integration point. Key references:

- `hubconf.py` — entrypoints; returns `(model, preprocessor)`, `model` is `EncPredWM`.
- `app/vjepa_wm/modelcustom/simu_env_planning/vit_enc_preds.py` — `EncPredWM`.
- `app/vjepa_wm/video_wm.py` — `VideoWM` (`encode_obs`, `encode_act`, `forward_pred`).
- `app/plan_common/datasets/preprocessor.py` — `Preprocessor`.
- `app/plan_common/datasets/__init__.py` — `DATA_STATS` (dims + norm stats, hardcoded).
- `evals/unroll_decode/eval.py` — the reference counterfactual eval we mirror.

## The model object: `EncPredWM`

All three baselines load as `EncPredWM` (a wrapper around `VideoWM`). We never
touch `.encoder` / `.predictor` directly. We use:

- `EncPredWM.encode(obs)` — `obs` is a `TensorDict`/dict `{"visual": (B,T,C,H,W)
  in [0,255], "proprio": (B,T,P)}` **or** a raw visual tensor. It does `/255` +
  `preprocessor.transform` + the frozen encoder internally. Returns visual
  latent `(B, T, V, H, W, D)` (V=1) — or a `TensorDict` with `"proprio"` too.
- `EncPredWM.unroll(z_ctxt, act_suffix)` — `act_suffix` is **`(T, B, A)`**
  (time-first). Autoregressive; maintains a `ctxt_window` (default 2). Returns
  predicted latents time-first `(tau+T, B, V, H, W, D)`. This is the planner's
  primitive (`evals/.../planning/planner.py`). Our `predict` calls it with T=1.
- `EncPredWM.action_dim` = the **model** action dim (raw_action_dim × tubelet ×
  frameskip ÷ action_skip). Reshape actions to this before `unroll`, exactly as
  `evals/unroll_decode/eval.py` does: `a.reshape(B, -1, wm.action_dim)`.

## Preprocessor API (the part the old code got wrong)

The real method is **`normalize_actions`** (plural), not `normalize_action`:

```python
preprocessor.normalize_actions(actions)   # (a - action_mean) / action_std ; shape (b, t, A)
preprocessor.normalize_proprios(proprio)   # (p - proprio_mean) / proprio_std
preprocessor.transform(visual)             # resize + ImageNet normalize ; (b,t,c,h,w) in [0,1]
preprocessor.inverse_transform(...)        # denormalize back toward [0,1]
```

The stats come from hardcoded `DATA_STATS` (no dataset download needed to load a
model). Action normalization:

| env | action_dim | proprio_dim | action norm | gripper idx |
|---|---|---|---|---|
| metaworld | 4 (dx,dy,dz,grip) | 4 (ee_xyz, grip) | **real shift+scale** | 3 |
| droid     | 7 (3 pos,3 euler,grip) | 7 | **identity** (mean 0, std 1) | 6 |
| robocasa  | 7 (droid format) | 7 | identity | 6 |
| pusht     | 2 | 4 | real | — |

**Consequence of the old bug:** the old adapter checked `normalize_action`
(singular), found nothing, and silently fell back to identity. For Metaworld
that means actions were fed un-normalized → predictions ~10× off → exactly the
failure the plan's "Note 1" warns about. Now fixed: `normalize_action` →
`preprocessor.normalize_actions`.

## Proprioception

Whether proprio is used is **read off the loaded model** (`wm.use_proprio`), not
assumed. The DROID checkpoints are `*_noprop` (use_proprio=False) — pass visual
only. Metaworld may use proprio — then `predict` re-encodes raw proprio via
`encode_proprio` and threads proprio features through `unroll`.

## Planning distance

Every published config is `..._L2_cem_...` → the planner's latent distance is
**L2** for all three baselines. So CRA uses L2 uniformly, which also keeps
cross-model CRA comparable. (The earlier note claiming cosine for JEPA-WM was
wrong.)

## Image pipeline

Feed `encode` **raw [0,255]** `(B,T,C,H,W)`. Do NOT pre-divide by 255 or
pre-normalize — `EncPredWM.encode` does that internally. The data loader passes
`transform=None` to the upstream dataset and rescales back to [0,255] so the
model's own transform (the training-time one) is the single source of truth.

## Validation procedure (plan Note 1 — now implemented)

`scripts/sanity_check.py::check_action_normalization`:
1. Load a real `(o_t, a_t, o_{t+1})` from the dataset.
2. `z_t, z_{t+1} = encode(o_t), encode(o_{t+1})`.
3. `z_hat = predict(z_t, a_t)` (action normalized inside).
4. Compare `MSE(z_hat, z_{t+1})` to the model's reported eval loss.
   A ≤ 2× gap is fine; a ~10× gap means normalization is still wrong.
