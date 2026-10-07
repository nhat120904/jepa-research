"""Preparation with the generic front end (g_entities.py) instead of the layout adapter (s_entities.py).

Per family: g_entities -> the same per-frame u_events call as the unified rerun -> integrated checks. The adapter
arm's entity tables (unified_rerun prep, s_entities) are re-extracted with the same current u_events into
adapter_events/ so that event quality is compared under one extractor (PRIVILEGED diagnostics, compare_events.py).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from stage_utils import cli_options, load_protocol, record_run, run_script

ap = argparse.ArgumentParser()
ap.add_argument('--root', type=Path, required=True)
ap.add_argument('--families', nargs='*', default=None)
a = ap.parse_args()
spec = load_protocol(a.root)
for family in a.families or list(spec['families']):
    data = spec['families'][family]
    out = a.root / 'prep' / family
    record_run(a.root, out, dict(stage='preparation', family=family, dataset=data['dataset'], frontend='g_entities'))
    run_script(a.root, 'g_entities.py', cli_options(dict(data=spec['data_root'], env=data['dataset'],
        train_episodes=spec['data']['train_episodes'], val_episodes=spec['data']['val_episodes'], out=out)), out / 'entities.log')
    run_script(a.root, 'u_events.py', cli_options(dict(entities=out / 'front', cache=out / 'cache' / data['dataset'],
        thresholds_from=out / 'thr', per_frame=True, out=out / 'events_unpruned')), out / 'events_unpruned.log')
    run_script(a.root, 'g_prune.py', cli_options(dict(front=out / 'front', events=out / 'events_unpruned')), out / 'prune.log')
    run_script(a.root, 'u_events.py', cli_options(dict(entities=out / 'front', cache=out / 'cache' / data['dataset'],
        thresholds_from=out / 'thr', per_frame=True, out=out / 'events')), out / 'events.log')
    adapter = Path(spec['adapter_prep']) / family
    run_script(a.root, 'u_events.py', cli_options(dict(entities=adapter / 'front', cache=adapter / 'cache' / data['dataset'],
        thresholds_from=adapter / 'thr', per_frame=True, out=out / 'adapter_events')), out / 'adapter_events.log')
    subprocess.run([sys.executable, str(a.root / 'verify_generic.py'), '--root', str(a.root), '--family', family], check=True)
    subprocess.run([sys.executable, str(a.root / 'compare_events.py'), '--root', str(a.root), '--family', family], check=True)
print('GENERIC_PREPARED', flush=True)
