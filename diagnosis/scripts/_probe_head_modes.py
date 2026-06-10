"""Diagnose the trained MDN head: did the component MEANS actually separate?

Mean collapse (components sharing one mu, differing only in sigma) is the failure
mode that leaves BB unchanged while still improving NLL. Prints, on a sample of
cached transitions: per-pair component mean distances (relative to the typical
true step size ||z_{t+1}-z_t||), sigma per component, pi stats, and whether the
argmax-pi component flips under neighbour actions.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import LatentCache, latent_cache_path, read_regimes  # noqa: E402
from models.adapters import build_adapter  # noqa: E402
from models.heads import MixturePredictorAdapter, flatten_tokens  # noqa: E402
from scripts._boundary_diagnostic import _load_runner_helpers  # noqa: E402
# inline load_head (digit-prefixed script modules cannot be imported normally)
from models.heads import MixtureDensityHead  # noqa: E402


def load_head(p, device):
    ckpt = torch.load(p, map_location=device, weights_only=False)
    h = MixtureDensityHead(latent_dim=ckpt["latent_dim"], action_dim=ckpt["action_dim"],
                           K=ckpt["K"], hidden=ckpt["hidden"], ctx_dim=ckpt["ctx_dim"],
                           state_dim=ckpt.get("state_dim", 0))
    h.load_state_dict(ckpt["state_dict"])
    return h.to(device).eval()


def main():
    cfg = yaml.safe_load(open("configs/diagnostic_metaworld.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = "dino_wm_metaworld"
    helpers = _load_runner_helpers()
    base = build_adapter(model, device=str(device)).eval()
    head = load_head(sys.argv[1] if len(sys.argv) > 1
                     else "checkpoints/mdn_dino_wm_metaworld_K3.pt", device)
    step = base.frames_per_step

    cache_path = latent_cache_path(cfg["latent_cache"]["root"], model, "metaworld")
    regimes = read_regimes(cache_path)
    with LatentCache(cache_path, mode="r") as cache:
        records = helpers.build_transition_records(cache, regimes, step, per_task=True)
        # pre_grasp cells are regime id 1 per REGIME_TO_ID order; just sample any 64
        sel = [records[i] for i in np.random.default_rng(3).choice(len(records), 64, replace=False)]
        d = helpers.materialize_records(cache, sel, step,
                                        want_proprio=base.uses_proprio(), want_state=True)

    from models.heads import metaworld_boundary_state_slice

    z_t = d["z_t"].to(device)
    a_t = d["a_t"].to(device)
    prop = d["proprio_t"].to(device) if d.get("proprio_t") is not None else None
    z_t1 = d["z_t1"].to(device)
    s = (metaworld_boundary_state_slice(d["state_t"].to(device).float())
         if head.state_dim > 0 else None)

    with torch.no_grad():
        base_pred = base.predict(z_t, a_t, proprio_t=prop)
        zt = flatten_tokens(z_t.float(), head.latent_dim)
        bt = flatten_tokens(base_pred.float(), head.latent_dim)
        B = a_t.shape[0]
        a_norm = base.normalize_action(a_t.reshape(B, -1, base.action_dim())).reshape(B, -1)
        out = head(zt, bt, a_norm, state=s)

        step_norm = (z_t1 - z_t).reshape(B, -1).norm(dim=-1)
        mu = out["mu"].reshape(B, head.K, -1)
        print(f"true step size ||z_t1 - z_t||: median {step_norm.median():.1f}")
        print(f"base residual ||base - z_t1||: median "
              f"{(base_pred - z_t1).reshape(B, -1).norm(dim=-1).median():.1f}")
        for i in range(head.K):
            for j in range(i + 1, head.K):
                dist = (mu[:, i] - mu[:, j]).norm(dim=-1)
                print(f"mu_{i} vs mu_{j}: median dist {dist.median():.3f}")
        print("sigma:", torch.exp(out["log_sigma"]).median(dim=0).values.cpu().numpy())
        pi = torch.softmax(out["pi_logits"], dim=-1)
        print("pi mean:", pi.mean(dim=0).cpu().numpy())

        # does the argmax component flip under perturbed actions?
        k0 = pi.argmax(-1)
        flips = 0
        for _ in range(8):
            a_pert = a_t + 0.3 * torch.randn_like(a_t)
            outp = head(zt, flatten_tokens(base.predict(z_t, a_pert, proprio_t=prop).float(),
                                           head.latent_dim),
                        base.normalize_action(a_pert.reshape(B, -1, base.action_dim())).reshape(B, -1),
                        state=s)
            flips += (torch.softmax(outp["pi_logits"], -1).argmax(-1) != k0).float().mean().item()
        print(f"argmax-pi flip rate under action perturbation: {flips / 8:.3f}")


if __name__ == "__main__":
    main()
