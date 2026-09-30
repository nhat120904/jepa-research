"""Frozen pi0.5 proposal policy for CTA on LIBERO-Safety (docs/CTA_LIBSAFE_PROTOCOL.md). openpi venv, GPU.

The released LIBERO-Safety checkpoint (HF LIBERO-Safety/pi05_libero_safety, openpi/orbax params) is loaded with openpi's
`pi05_libero` training config (action horizon 10) and the checkpoint's own normalization statistics
(assets/lerobot/norm_stats.json), through openpi's create_trained_policy, so the input/output transforms are openpi's.
A bank of K chunks is ONE batched flow-matching call: the observation is transformed once and tiled, and row k
integrates the flow from its own seeded noise (NumPy PCG64 seeded with ti_wm.contract.candidate_seed), so candidate k
depends only on (root, decision, k) and candidate 0 is the P0 draw.
"""
import numpy as np

from ti_wm.contract import candidate_seed

ACTION_HORIZON, MODEL_ACTION_DIM = 10, 32


def seeded_noise(seeds, horizon=ACTION_HORIZON, dim=MODEL_ACTION_DIM):
    return np.stack([np.random.default_rng(int(s)).standard_normal((horizon, dim)) for s in seeds]).astype(np.float32)


def bank_seeds(root, decision, k):
    return [candidate_seed(root, decision, j) for j in range(k)]


class Pi05:
    def __init__(self, checkpoint_dir):
        import jax
        from openpi.policies import policy_config
        from openpi.training import checkpoints, config

        cfg = config.get_config("pi05_libero")
        norm = checkpoints.load_norm_stats(f"{checkpoint_dir}/assets", "lerobot")
        self.policy = policy_config.create_trained_policy(cfg, checkpoint_dir, norm_stats=norm)
        self.jax = jax
        self._rng = jax.random.key(0)            # unused by the flow when noise is given; kept for the signature

    def bank(self, inputs, seeds):
        """inputs: one openpi LIBERO observation dict -> (K, ACTION_HORIZON, 7) env actions."""
        import jax.numpy as jnp
        from openpi.models import model as _model

        p = self.policy
        x = p._input_transform(self.jax.tree.map(lambda v: v, inputs))
        k = len(seeds)
        batch = self.jax.tree.map(lambda v: jnp.repeat(jnp.asarray(v)[None], k, axis=0), x)
        noise = jnp.asarray(seeded_noise(seeds))
        actions = np.asarray(p._sample_actions(self._rng, _model.Observation.from_dict(batch), noise=noise))
        state = np.asarray(x["state"])
        return np.stack([p._output_transform({"state": state, "actions": actions[j]})["actions"] for j in range(k)])
