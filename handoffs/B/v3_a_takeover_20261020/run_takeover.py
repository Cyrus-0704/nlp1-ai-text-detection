"""Deferred A/seed43 takeover. Without --execute this only prints the queue."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import zipfile


VERSION = 'V3-20261020'
SEEDS = {'B': 42, 'C': 44}
ARMS = {'C': ('T0',), 'B': ('T1', 'T2')}
CODE_SHA = '67d98d34411215c3a91b1c781e344c913e80eaf8ffd7e2cb2f4a9b0e968ac4b7'
CODE_BYTES = 239474
MANIFEST_SHA = '56168d5f005d2cfa310660d5736dca6380afc38bc1c36342cd6b9ef952c8ea36'
DEV_SHA = '1d80e7b6045e0914f224dcbc3afe5f56569f5bd995f742607389c26316476600'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def completed(source, executor):
    """Read-only completion gate; never open a live resource ledger for recovery."""
    source = Path(source).resolve()
    receipts = {}
    for arm in ('T0', 'T1', 'T2'):
        identity = f'{VERSION}-{executor}-s{SEEDS[executor]}-{arm}'
        root = source / 'runs/v3' / identity / 'formal'
        execution = read(root / 'execution.json')
        protocol = read(root / 'protocol.json')
        signature = protocol['signature']
        expected = dict(experiment_id=identity, owner=executor, seed=SEEDS[executor], arm=arm)
        need(execution.get('status') == 'COMPLETE_LOCAL_DEV_NOT_INDEPENDENT_EFFECT'
             and all(execution.get(key) == value for key, value in expected.items()),
             identity + ': original task is not fully complete')
        need(signature['owner'] == executor and signature['experiment_id'] == identity
             and signature['phase'] == 'formal' and signature['settings']['seed'] == SEEDS[executor]
             and signature['settings']['arm'] == arm and signature['settings']['max_steps'] == 7500,
             identity + ': wrong original experiment protocol')
        protocol_sha = sha(root / 'protocol.json')
        need(execution['protocol_sha256'] == protocol_sha, identity + ': protocol SHA mismatch')
        for name, limits in (('ledger.json', 'limits'), ('evaluation_ledger.json', 'evaluation_limits')):
            ledger = read(root / name)
            spent = ledger['spent_seconds']
            need(ledger['protocol_sha256'] == protocol_sha and ledger['active'] is None
                 and isinstance(spent, (int, float)) and not isinstance(spent, bool)
                 and math.isfinite(spent) and 0 <= spent <= ledger['budget_seconds']
                 and ledger['budget_seconds'] == protocol[limits]['budget_seconds'],
                 identity + ': unfinished/over-budget original ledger')
        need(sha(root / 'ledger.json') == execution['training_ledger_sha256'],
             identity + ': original training ledger changed')
        checkpoint = Path(execution['checkpoint']).resolve()
        need(checkpoint.is_relative_to(root.resolve()), identity + ': foreign checkpoint')
        meta = read(checkpoint / 'resume_manifest.json')
        need(meta['signature'] == signature and meta['optimizer_step'] == 7500,
             identity + ': wrong final checkpoint boundary')
        results = [read(path) for path in root.glob('training/attempt-*/result.json')]
        # shortcut: recovery without a final training receipt needs manual review, never infer completion.
        need(any(result.get('optimizer_updates') == 7500 and result.get('completed') is True
                 and result.get('state_verified') is True
                 and Path(result['checkpoint']).resolve() == checkpoint for result in results),
             identity + ': final training/state verification missing')
        reload = read(root / 'reload_acceptance.json')
        pids = [reload.get(name) for name in ('reference_process_id', 'reload_process_id')]
        need(sha(root / 'reload_acceptance.json') == execution['reload_acceptance_sha256']
             and reload.get('status') == 'PASS_LOCAL_ACTUAL_MODEL_RELOAD'
             and reload.get('actual_model_reload_verified') is True
             and all(type(pid) is int and pid > 0 for pid in pids) and pids[0] != pids[1],
             identity + ': independent reload evidence missing')
        dev = execution['dev_receipt']
        need(dev.get('actual_model_reload_verified') is True
             and dev.get('input_sha256') == DEV_SHA and dev.get('rows') == 15086
             and dev.get('max_length') == 512 and dev.get('precision') == 'fp32' and dev.get('batch_size') == 1
             and dev.get('ordered_ids') is True and dev.get('finite_probability_range') is True
             and sha(root / 'dev_predictions.csv') == execution['dev_prediction_sha256'] == dev.get('predictions_sha256'),
             identity + ': dev completion/probability SHA mismatch')
        receipts[identity] = dict(execution_sha256=sha(root / 'execution.json'), protocol_sha256=protocol_sha)
    return receipts


def execute(args):
    need(sys.version_info[:2] == (3, 11), 'Use the executor existing verified Python 3.11')
    source = args.source_workspace.resolve()
    workspace = args.workspace.resolve()
    need(workspace != source and not workspace.exists(), 'Preserve existing workspaces; choose a new takeover path')
    receipts = completed(source, args.executor)
    host = read(source / '.local/v3' / (args.executor + '.json'))
    machine = hashlib.sha256((platform.node() + '|' + str(Path(sys.executable).resolve())).encode()).hexdigest()
    need(host['owner'] == args.executor and host['seed'] == SEEDS[args.executor]
         and host['machine'] == machine, 'Use this original executor host and verified interpreter')
    model, data = Path(host['model_dir']).resolve(), Path(host['data_dir']).resolve()
    need(model.is_dir() and data.is_dir(), 'Original pretrained model/frozen data missing; no automatic download')
    if os.name == 'nt':
        os.path.relpath(model, workspace).encode('ascii')
    archive = args.code_archive.resolve()
    need(archive.stat().st_size == CODE_BYTES and sha(archive) == CODE_SHA, 'Frozen code archive identity mismatch')
    with zipfile.ZipFile(archive) as package:
        need(package.testzip() is None and len(package.namelist()) == len(set(package.namelist())), 'Corrupt/duplicate ZIP members')
        workspace.mkdir(parents=True)
        package.extractall(workspace)
    need(sha(workspace / 'main_model/configs/v3/package_manifest.json') == MANIFEST_SHA, 'Wrong inner package manifest')
    cache = workspace / '.local/cache'
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1', PYTHONIOENCODING='utf-8',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    for variable, suffix in (('TEMP', 'tmp'), ('TMP', 'tmp'), ('HF_HOME', 'hf'),
                             ('TORCH_HOME', 'torch'), ('CUDA_CACHE_PATH', 'cuda')):
        path = cache / suffix
        path.mkdir(parents=True, exist_ok=True)
        env[variable] = str(path)
    record = dict(version=VERSION, logical_owner='A', seed=43, executor=args.executor,
                  machine=machine, source_workspace=str(source), code_archive_sha256=CODE_SHA,
                  assigned_arms=list(ARMS[args.executor]), predecessor_receipts=receipts,
                  historical_A_failures=dict(evidence_commit='6d28d84',
                      T1_prior_host_pilot_spent_seconds=25.085621000001993,
                      T1_prior_host_formal_started=False,
                      note='Original A ledgers remain intact; each new host requires its own pilot; report total migration costs'),
                  status='PREPARED_NOT_STARTED')
    record_path = workspace / '.local/a_takeover.json'

    def save():
        record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    def call(*command):
        subprocess.run([sys.executable, '-B', *map(str, command)], cwd=workspace, env=env, check=True)

    save()
    try:
        call('tools/v3_bootstrap.py', 'verify-code', '--root', '.')
        call('main_model/v3_delivery.py', 'configure', '--owner', 'A', '--model-dir', model, '--data-dir', data)
        call('-c', "import sys; sys.path.insert(0,'main_model'); import v3_run; v3_run.require_idle_cuda()")
        call('main_model/v3_delivery.py', 'preflight', '--owner', 'A', '--cuda')
        for arm in ARMS[args.executor]:
            record.update(status='RUNNING', active_arm=arm)
            save()
            for command in ('pilot', 'prepare', 'run', 'package'):
                flags = ['--execute'] if command in ('pilot', 'run') else []
                call('main_model/v3_run.py', command, '--owner', 'A', '--arm', arm, *flags)
            call('main_model/v3_run.py', 'package', '--owner', 'A', '--arm', arm, '--weights')
        record.update(status='COMPLETE_LOCAL_DEV_NOT_INDEPENDENT_EFFECT', active_arm=None)
    except BaseException as exc:
        record.update(status='FAILED_PRESERVE_ALL_EVIDENCE', error=repr(exc))
        raise
    finally:
        save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executor', choices=SEEDS, required=True)
    parser.add_argument('--source-workspace', type=Path)
    parser.add_argument('--workspace', type=Path)
    parser.add_argument('--code-archive', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status='DEFERRED_NOT_STARTED', executor=args.executor, logical_owner='A', seed=43,
                              prerequisite=f'{args.executor}/seed{SEEDS[args.executor]} T0,T1,T2 fully complete',
                              queue=list(ARMS[args.executor]), formal_budget_hours=10,
                              note='No files changed; no polling, GPU, pilot or training started'), ensure_ascii=False, indent=2))
        return
    if any(value is None for value in (args.source_workspace, args.workspace, args.code_archive)):
        parser.error('--execute requires --source-workspace, --workspace and --code-archive')
    execute(args)


if __name__ == '__main__':
    main()
