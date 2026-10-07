import argparse
import json
import subprocess
import sys
from pathlib import Path
from stage_utils import load_protocol, cli_options, record_run, run_script, sha_file

ap = argparse.ArgumentParser()
ap.add_argument('--root', type=Path, required=True)
a = ap.parse_args()
spec = load_protocol(a.root)
for test in ('test_segments.py', 'test_state_semantics.py'):
    subprocess.run([sys.executable, str(a.root / test)], check=True)
for family, data in spec['families'].items():
    out = a.root / 'prep' / family
    record_run(a.root, out, dict(stage='preparation', family=family, dataset=data['dataset']))
    run_script(a.root, 's_entities.py', cli_options(dict(data=spec['data_root'], env=data['dataset'],
        train_episodes=spec['data']['train_episodes'], val_episodes=spec['data']['val_episodes'], out=out)), out/'entities.log')
    run_script(a.root, 'u_events.py', cli_options(dict(entities=out/'front', cache=out/'cache'/data['dataset'],
        thresholds_from=out/'thr', per_frame=True, out=out/'events')), out/'events.log')
    subprocess.run([sys.executable, str(a.root / 'verify_prepared.py'), '--root', str(a.root), '--family', family], check=True)
    hashes = {split: sha_file(Path(spec['data_root']) / (data['dataset'] + suffix))
              for split, suffix in [('train', '.npz'), ('val', '-val.npz')]}
    (out/'data_sha256.json').write_text(json.dumps(hashes, indent=2)+'\n')
(a.root/'preparation_complete.json').write_text(json.dumps({'families':list(spec['families']), 'protocol':spec['version']},indent=2)+'\n')
print('ALL_THREE_PREPARED', flush=True)
