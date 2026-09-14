"""Fixed-choice confirmation on trusted 52419 artifacts; compute nodes only."""
import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

from run_stage_b_qualification import BudgetStop, qualify, run_branch
from run_stage_b_profile import write_json
from run_baseline_sim_client import InferenceClient, load_multistep_wrapper
from run_stage_a4_collect import wait_for_server


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--server-pid', type=int, required=True)
    p.add_argument('--server-state', type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use sbatch')
    root = Path('/mnt/data/nhatnc129/jepa/latent_scope_baseline')
    original = root / 'outputs/qualification_52419/qualification_result.json'
    old = json.loads(original.read_text())
    cfg = old['config']['qualification']
    panel = [r for r in old['prefixes'] if r['complete'] and r['recoverable']]
    if [(r['attempt'], r['selected_candidate']) for r in panel] != [(4, 7), (5, 3)]:
        raise RuntimeError('Locked panel changed')
    start = float(os.environ['SEED_CONFIRM_STARTED_UNIX'])
    deadline = start + 13500  # 3h45, including startup; hard cap 4h.
    result = {'verdict': 'RUNNING', 'job_id': os.environ['SLURM_JOB_ID'],
              'source_result': str(original),
              'source_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
              'design': 'Two fixed choices, eight new paired seeds each; no reselection',
              'pairs': [], 'gates': [], 'stage_c_authorized': False}
    path = args.output_dir / 'seed_confirmation_result.json'
    write_json(path, result)
    client = InferenceClient('localhost', args.port, timeout_ms=1000)
    try:
        wait_for_server(client, args.server_pid, args.server_state)
        wrapper = load_multistep_wrapper(root / 'src/Isaac-GR00T')
        for row in panel:
            directory = Path(row['directory'])
            with (directory / 'source_snapshot.pkl').open('rb') as f:
                source = pickle.load(f)
            with (directory / 'candidate_bank.pkl').open('rb') as f:
                banks = pickle.load(f)
            gate = qualify(wrapper, cfg, source, deadline)
            result['gates'].append({'attempt': row['attempt'], 'gate': gate})
            write_json(path, result)
            if not gate['pass']:
                raise RuntimeError('Saved source replay gate failed')
            for repeat in range(8):
                seed = 153000 + row['attempt'] * 100 + repeat
                pair = {'attempt': row['attempt'], 'seed': seed,
                        'selected_candidate': row['selected_candidate'], 'branches': []}
                result['pairs'].append(pair)
                for index in (0, row['selected_candidate']):
                    branch, _ = run_branch(client, wrapper, cfg, source, banks[index], seed, deadline)
                    branch.pop('intervention_state')
                    branch['candidate_index'] = index
                    pair['branches'].append(branch)
                    write_json(path, result)
                print(f"attempt={row['attempt']} seed={seed} outcomes="
                      f"{[b['success'] for b in pair['branches']]}", flush=True)
            del source, banks
        result['verdict'] = 'SEED_CONFIRMATION_COMPLETE'
    except BudgetStop as error:
        result['verdict'] = 'SEED_CONFIRMATION_PARTIAL'
        result['error'] = str(error)
    except Exception as error:
        result['verdict'] = 'SEED_CONFIRMATION_ERROR'
        result['error'] = repr(error)
        raise
    finally:
        complete = [p for p in result['pairs'] if len(p['branches']) == 2]
        def summarize(pairs):
            baseline = sum(p['branches'][0]['success'] for p in pairs)
            selected = sum(p['branches'][1]['success'] for p in pairs)
            return {'pairs': len(pairs), 'baseline_successes': baseline,
                    'selected_successes': selected,
                    'gain_pp': 100 * (selected - baseline) / len(pairs) if pairs else None,
                    'wins': sum(not p['branches'][0]['success'] and p['branches'][1]['success'] for p in pairs),
                    'losses': sum(p['branches'][0]['success'] and not p['branches'][1]['success'] for p in pairs)}
        result['summary'] = summarize(complete)
        result['per_prefix'] = {str(r['attempt']): summarize([p for p in complete if p['attempt'] == r['attempt']]) for r in panel}
        result['interpretation'] = 'Conditional confirmation on two previously selected prefixes; not population headroom'
        result['elapsed_seconds'] = time.time() - start
        write_json(path, result)
        client.close()


if __name__ == '__main__':
    main()
