"""Offline unit test for the `on_elites` hook added to `cem_plan_latent`
(scripts/30_latent_oracle.py) — the shared foundation both Track-1 Phase A/B
(scripts/35, scripts/_cem_mining) and Track-2's adversarial loss term rely on to
mine the frames CEM actually commits to.

No MuJoCo/GPU needed: `roll_final_frame`/`encode_batch`/`snapshot`/`restore` are
monkeypatched with cheap fakes so the CEM loop itself (sampling, elite selection,
mean/var update) is exercised directly, proving the hook fires with the right
population and the right cost-ascending order — the two properties
`scripts/_cem_mining.mine_cem_frames` depends on.
"""

from __future__ import annotations

import importlib.util as _ilu
import types
from pathlib import Path

import numpy as np
import torch

import scripts._cem_mining as _cem_mining

ROOT = Path(__file__).resolve().parents[1]


def _load_latent_oracle():
    spec = _ilu.spec_from_file_location("latent_oracle_test", str(ROOT / "scripts" / "30_latent_oracle.py"))
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_on_elites_called_each_iteration(monkeypatch):
    lo = _load_latent_oracle()
    dim = 8
    iterations = 4
    num_samples = 20
    elite_frac = 0.2

    # Make z_fin dependent on the sampled action so the cost can discriminate.
    captured = {}

    def fake_roll_final_frame(env, snap, plan_raw):
        z = np.asarray(plan_raw, dtype=np.float32).reshape(-1)
        raw = np.zeros(39, dtype=np.float32)
        raw[4:7] = z[:3]
        raw[0:3] = z[:3] * 0.5
        return z, np.zeros(4, dtype=np.float32), raw

    def fake_encode_batch(adapter, frames, proprios, device):
        return torch.tensor(np.stack([f[:dim] if len(f) >= dim else
                                      np.pad(f, (0, dim - len(f))) for f in frames]),
                            dtype=torch.float32)

    monkeypatch.setattr(lo, "snapshot", lambda env: {"dummy": True})
    monkeypatch.setattr(lo, "restore", lambda env, snap: None)
    monkeypatch.setattr(lo, "roll_final_frame", fake_roll_final_frame)
    monkeypatch.setattr(lo, "encode_batch", fake_encode_batch)

    calls = []

    def on_elites(z_elite, raw_elite, cost_elite, it):
        calls.append((z_elite.shape[0], len(raw_elite), len(cost_elite), it))
        # elite costs must already be sorted ascending (argsort order preserved).
        assert list(cost_elite) == sorted(cost_elite)

    z_goal = torch.zeros(dim)

    def cost_fn(z_fin):
        return (z_fin ** 2).sum(-1)     # goal is the origin; smaller z -> lower cost

    rng = np.random.default_rng(0)
    lo.cem_plan_latent(env=None, adapter=None, z_goal=z_goal, device=torch.device("cpu"),
                       plan_h=1, num_samples=num_samples, iterations=iterations,
                       elite_frac=elite_frac, var0=1.0, rng=rng, cost_fn=cost_fn,
                       on_elites=on_elites)

    n_elite = max(2, int(num_samples * elite_frac))
    assert len(calls) == iterations
    for shape0, n_raw, n_cost, it in calls:
        assert shape0 == n_elite
        assert n_raw == n_elite
        assert n_cost == n_elite
    assert [c[3] for c in calls] == list(range(iterations))


def test_on_elites_default_none_is_backward_compatible(monkeypatch):
    """Existing Phase-0/3 gate runs (scripts/30 without mining) must be unaffected —
    on_elites defaults to None and cem_plan_latent must not require it."""
    lo = _load_latent_oracle()
    dim = 4

    def fake_roll_final_frame(env, snap, plan_raw):
        z = np.asarray(plan_raw, dtype=np.float32).reshape(-1)
        return z, np.zeros(4, dtype=np.float32), np.zeros(39, dtype=np.float32)

    def fake_encode_batch(adapter, frames, proprios, device):
        return torch.tensor(np.stack([f[:dim] for f in frames]), dtype=torch.float32)

    monkeypatch.setattr(lo, "snapshot", lambda env: {"dummy": True})
    monkeypatch.setattr(lo, "restore", lambda env, snap: None)
    monkeypatch.setattr(lo, "roll_final_frame", fake_roll_final_frame)
    monkeypatch.setattr(lo, "encode_batch", fake_encode_batch)

    z_goal = torch.zeros(dim)
    plan = lo.cem_plan_latent(env=None, adapter=None, z_goal=z_goal, device=torch.device("cpu"),
                              plan_h=1, num_samples=10, iterations=2, elite_frac=0.2,
                              var0=1.0, rng=np.random.default_rng(1),
                              cost_fn=lambda z: (z ** 2).sum(-1))
    assert plan.shape[-1] == lo.RAW_A


def test_mine_cem_frames_accepts_full_cem_kw(monkeypatch):
    """Regression (found by job 23899's crash): `scripts/35_cem_exploit_precision.py`
    always builds `cem_kw` with num_samples/iterations/elite_frac/var0 already set
    (from CLI args) and passes it straight through — `mine_cem_frames` must not
    re-specify those same keys as explicit defaults AND unpack `**cem_kw` in the
    same `dict(...)` call (that raises "got multiple values for keyword argument").
    Faked latent_oracle module (no MuJoCo/GPU) so this exercises the exact merge
    logic without needing a real env."""
    dim = 4

    class FakeEnv:
        def reset(self):
            return np.zeros(39, dtype=np.float32), {}

        def step(self, a):
            return np.zeros(39, dtype=np.float32), 0.0, False, False, {"success": 0.0}

        def close(self):
            pass

    fake_lo = types.SimpleNamespace()
    fake_lo.make_env = lambda task, seed: (FakeEnv(), np.zeros(39, dtype=np.float32))
    fake_lo.rollout_expert = lambda env, init_state, task: (
        np.zeros((4, 4, 3), dtype=np.uint8), np.zeros(39, dtype=np.float32), 0)
    fake_lo.encode_frame = lambda adapter, frame, proprio, device: torch.zeros(dim)
    fake_lo.FRAMESKIP, fake_lo.RAW_A = 1, 2
    fake_lo.build_oracle_cost = lambda **kw: (lambda z: torch.zeros(z.shape[0]))

    def fake_cem_plan_latent(env, adapter, z_goal, device, *, plan_h, num_samples,
                             iterations, elite_frac, var0, rng, cost_fn, on_elites=None):
        if on_elites is not None:
            n_elite = max(2, int(num_samples * elite_frac))
            z_elite = torch.zeros(n_elite, dim)
            raw_elite = [np.zeros(39, dtype=np.float32) for _ in range(n_elite)]
            cost_elite = np.zeros(n_elite, dtype=np.float32)
            on_elites(z_elite, raw_elite, cost_elite, 0)
        return np.zeros((plan_h * fake_lo.FRAMESKIP, fake_lo.RAW_A))
    fake_lo.cem_plan_latent = fake_cem_plan_latent

    monkeypatch.setattr(_cem_mining, "_load", lambda modname, fname: fake_lo)

    cem_kw = dict(num_samples=10, iterations=1, elite_frac=0.2, var0=1.0)   # scripts/35's exact pattern
    buf = _cem_mining.mine_cem_frames(adapter=None, device=torch.device("cpu"),
                                      cost_spec_kwargs=dict(cost="l2"), tasks=["fake-task"],
                                      episodes=1, cem_kw=cem_kw, max_episode_steps=1,
                                      verbose=False)
    assert buf["z"].shape[0] == max(2, int(10 * 0.2))
