"""Shared protocol, frozen-source verification and subprocess execution for Slurm stages."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

def load_protocol(root):
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('All numerical/physics stages must run under sbatch')
    for line in (root / 'SOURCE_SHA256SUMS').read_text().splitlines():
        expected, relative = line.split('  ', 1)
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Frozen source changed: {relative}')
    return json.loads((root / 'protocol.json').read_text())

def cli_options(values):
    result = []
    for key, value in values.items():
        if value is False or value is None:
            continue
        result.append('--' + key.replace('_', '-'))
        if value is not True:
            result.extend(str(x) for x in (value if isinstance(value, list) else [value]))
    return result

def record_run(root, out, metadata):
    out.mkdir(parents=True, exist_ok=False)
    for name in ('protocol.json', 'SOURCE_SHA256SUMS'):
        shutil.copy2(root / name, out / name)
    metadata.update(job_id=os.environ['SLURM_JOB_ID'], protocol_sha256=hashlib.sha256((root / 'protocol.json').read_bytes()).hexdigest())
    (out / 'run_manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')

def run_script(root, name, arguments, log=None):
    command = [sys.executable, str(root / 'source' / name), *arguments]
    print('COMMAND', command, flush=True)
    if log is None:
        subprocess.run(command, check=True)
    else:
        with log.open('w') as stream:
            subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)

def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
