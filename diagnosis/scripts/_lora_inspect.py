"""Throwaway: dump the real ViTPredictor module structure + test LoRA grad flow,
so we can fix scripts/26's injection target (post-mortem of the 22378 zero-grad null).
No cache; builds the model, injects LoRA, runs one predict, checks dL/dB."""
import sys
from pathlib import Path
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.adapters import build_adapter  # noqa: E402
from models.heads.lora_predictor import inject_lora, set_lora_enabled, _predictor_of  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
adapter = build_adapter("dino_wm_metaworld", device=str(device)).eval()
pred = _predictor_of(adapter)

print("=== ALL nn.Linear in the predictor (name : in->out) ===", flush=True)
lins = [(n, m) for n, m in pred.named_modules() if isinstance(m, nn.Linear)]
for n, m in lins:
    print(f"  {n:55s} {m.in_features}->{m.out_features}", flush=True)
print(f"total Linear: {len(lins)}", flush=True)

print("\n=== top-level predictor children ===", flush=True)
for n, _ in pred.named_children():
    print("  ", n, flush=True)

# what does the current target match?
for p in adapter.encpred.parameters():
    p.requires_grad_(False)
inj = inject_lora(adapter, r=8, alpha=16)
print(f"\ninject_lora matched {len(inj)} Linear layers with target ('layers','blocks')",
      flush=True)

# grad-flow test on the EXACT scripts/26 path: one cached transition -> predict.
import yaml  # noqa: E402
from data import LatentCache, latent_cache_path, read_regimes  # noqa: E402
from scripts._boundary_diagnostic import _load_runner_helpers  # noqa: E402

cfg = yaml.safe_load(open("configs/diagnostic_metaworld.yaml"))
helpers = _load_runner_helpers()
cache_path = latent_cache_path(cfg["latent_cache"]["root"], "dino_wm_metaworld",
                               cfg["dataset"]["name"])
with LatentCache(cache_path, mode="r") as cache:
    recs = helpers.build_transition_records(cache, read_regimes(cache_path),
                                            adapter.frames_per_step, per_task=True)
    d = helpers.materialize_records(cache, recs[:4], adapter.frames_per_step,
                                    want_proprio=adapter.uses_proprio(), want_state=True)
z_t = d["z_t"].to(device).float()
a = d["a_t"].to(device).float()
prop = d["proprio_t"].to(device) if d.get("proprio_t") is not None else None
print(f"\ncached z_t shape={tuple(z_t.shape)} a shape={tuple(a.shape)} "
      f"uses_proprio={adapter.uses_proprio()}", flush=True)
set_lora_enabled(inj, True)
out = adapter.predict(z_t, a, proprio_t=prop)
print(f"predict out shape={tuple(out.shape)} requires_grad={out.requires_grad} "
      f"grad_fn={out.grad_fn is not None}", flush=True)
loss = out.float().pow(2).mean()
loss.backward()
gB = sum(float(m.B.grad.abs().sum()) for m in inj if m.B.grad is not None)
gA = sum(float(m.A.grad.abs().sum()) for m in inj if m.A.grad is not None)
nB_none = sum(1 for m in inj if m.B.grad is None)
print(f"\nGRAD-FLOW: sum|dL/dB|={gB:.3e} sum|dL/dA|={gA:.3e}  B.grad None for "
      f"{nB_none}/{len(inj)} adapters", flush=True)
print("  -> grad reaches LoRA (training will work)" if gB > 0
      else "  -> ZERO grad through adapter.predict (no_grad/inference_mode in the path)",
      flush=True)
print("INSPECT_DONE", flush=True)
