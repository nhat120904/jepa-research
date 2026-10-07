"""Alignment diagnostics integrated into preparation; public observations are the only method inputs."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from s_entities import state_entities

if 'SLURM_JOB_ID' not in os.environ:
    raise RuntimeError('Use sbatch')
ap = argparse.ArgumentParser()
ap.add_argument('--out', type=Path, required=True)
a = ap.parse_args()
L = json.loads((a.out / 'front/layout.json').read_text())
cache = a.out / 'cache/scene-play-v0'
ev = np.load(a.out / 'events/events_train.npz')
assert ev['before'].shape[1:] == (5, 6)
assert np.isfinite(ev['before']).all() and np.isfinite(ev['after']).all()
assert (np.bincount(ev['e'], minlength=5) > 0).all(), 'Missing a scene interaction from training events'
ob = np.load(cache / 'val_observations.npy', mmap_mode='r')
q = ob[:1000].copy()
q[:, [31, 35, 37, 39]] += 100
np.testing.assert_array_equal(state_entities(q, L)[0], state_entities(ob[:1000], L)[0])
import gymnasium
import ogbench.manipspace
env = gymnasium.make('scene-v0')
checks = []
for task in range(1, 6):
    obs, info = env.reset(seed=task * 100, options=dict(task_id=task, render_goal=False))
    S, _ = state_entities(obs[None], L)
    G, _ = state_entities(info['goal'][None], L)
    assert np.isfinite(S).all() and np.isfinite(G).all()
    # Simulator sites below are diagnostic measurements only, never training labels or policy inputs.
    u = env.unwrapped
    site = np.stack([u._data.site_xpos[u._drawer_site_id], u._data.site_xpos[u._window_site_id]])
    actual = 32 + 80 * (site[:, :2] - np.array([.425, 0]))
    error_m = np.linalg.norm(S[0, 3:5, :2] - actual, axis=-1) / 80
    checks.append(dict(task=task, handle_location_error_m=error_m.tolist(), read_start=S[0].tolist(), read_goal=G[0].tolist()))
env.close()
report = dict(training_events=len(ev['e']), acted_counts=np.bincount(ev['e'], minlength=5).tolist(),
              train_geometry_fit=L['geometry_fit_train'], public_state_semantics='passed',
              simulator_alignment_diagnostic=checks)
(a.out / 'preparation_checks.json').write_text(json.dumps(report, indent=2) + '\n')
print('PREPARATION_CHECKS', json.dumps({k: v for k, v in report.items() if k != 'simulator_alignment_diagnostic'}), flush=True)
