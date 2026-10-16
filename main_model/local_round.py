"""One-command foreground B length experiment. No scheduler, polling, or online service."""
import argparse
from contextlib import nullcontext
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import traceback
import zipfile

from round_support import (ROOT, StageBook, csv_out, digest, dump, foreground, git_record,
                           next_attempt, predictions, read_json, verify_checkpoint)


def load_config(path, include_optional=False, seed43_only=False):
    c = read_json(path)
    if c['data_version'] != 'internal-v2' or c['evaluation'] != dict(
            threshold=.5,short_by='word',short_threshold=100,temporary=True):
        raise ValueError('This round freezes internal-v2, threshold .5, temporary word<100')
    t = c['training']
    expected = dict(max_steps=2000,max_length=512,batch_size=1,gradient_accumulation=16,
        learning_rate=2e-5,class_weighting='none',inference_batch_size=8,save_steps=500,progress_every=100,
        resource_steps=50,evaluation_steps=[500,1000,2000],segment_steps=[50,500,1000,1500,2000],deterministic=True)
    if t != expected:
        raise ValueError('Unexpected protocol change; define a separate reviewed experiment')
    intervention=c.get('experiment')=='classifier_dropout_02'
    if not intervention and (c.get('experiment') is not None or c.get('classifier_dropout') is not None):
        raise ValueError('Unknown intervention; do not silently ignore requested hyperparameters')
    if intervention:
        if c.get('classifier_dropout')!=.2 or c['seeds']!=[42,43] or c['optional_seeds']!=[] or include_optional or seed43_only:
            raise ValueError('Dropout intervention fixes both seeds42/43 and classifier0.2')
    elif c['seeds'] != [42] or c['optional_seeds'] != [43]:
        raise ValueError('Unexpected seed protocol')
    if include_optional and seed43_only:
        raise ValueError('Choose either both seeds or seed43 only')
    c['effective_seeds'] = [43] if seed43_only else c['seeds'] + (c['optional_seeds'] if include_optional else [])
    # Paths used by this runner are pinned; no input path may reach test/official data.
    for key, expected_path in [('train_file','data/processed/v2/train.jsonl'),
                               ('dev_file','data/processed/v2/dev.jsonl'),
                               ('data_manifest','data/processed/v2/manifest.json'),
                               ('evaluation_script','data_tools/evaluate.py')]:
        if c[key] != expected_path:
            raise ValueError(f'Unexpected input path: {key}')
    return c


def preflight(c):
    from common import read_rows, validate_splits
    from gpu_guard import process_snapshot
    record = git_record()
    if record['branch'] != 'work/b-model':
        raise ValueError('B must remain on work/b-model; no branch was changed')
    versions = [p.name for p in (ROOT/'data/processed').iterdir() if p.is_dir() and p.name not in ('v1','v2')]
    if versions:
        raise ValueError(f'New data directory found: {versions}; report differences before replacing this round')
    for key, sha_key in [('train_file','train_sha256'),('dev_file','dev_sha256'),
                         ('data_manifest','manifest_sha256'),('evaluation_script','evaluator_sha256')]:
        actual = digest(ROOT/c[key])
        if actual != c[sha_key]:
            raise ValueError(f'Frozen version/hash differs: {key}; expected={c[sha_key]}, actual={actual}; do not auto-substitute')
    m = read_json(ROOT/c['data_manifest'])
    if m['version'] != c['data_version']:
        raise ValueError('Manifest version mismatch')
    train = read_rows(ROOT/c['train_file'],labeled=True)
    dev = read_rows(ROOT/c['dev_file'],labeled=True)
    if len(train)!=119842 or len(dev)!=15086:
        raise ValueError('Frozen train/dev row counts differ')
    for rows in (train,dev):
        if not all(isinstance(r['sample_id'],str) and (r['label']!=0 or r['generator']=='human') for r in rows):
            raise ValueError('String IDs/human generator contract violated')
    validate_splits(train,dev)
    norm = lambda r:' '.join(r['text'].casefold().split())
    if {norm(r) for r in train} & {norm(r) for r in dev}:
        raise ValueError('Casefold-normalized text overlap')
    source = read_json(ROOT/c['model_dir']/'source_manifest.json')
    if digest(ROOT/c['model_dir']/'source_manifest.json')!=c['source_manifest_sha256']:
        raise ValueError('Accepted pretrained source manifest differs')
    if source['repo_id']!=c['model_repo'] or source['revision']!=c['model_revision']:
        raise ValueError('Pretrained source/revision differs')
    for name,item in source['files'].items():
        if digest(ROOT/c['model_dir']/name)!=item['sha256']:
            raise ValueError(f'Pretrained source file differs: {name}')
    ids = [r['sample_id'] for r in dev]
    if c.get('experiment')=='classifier_dropout_02':
        control_manifest=read_json(ROOT/'handoffs/B/b_v2_length512_two_seed/manifest.json')['files']
        for control in c['comparison_baselines']:
            path=ROOT/control['run']
            ref_name=f'L512_s{control["seed"]}_u{control["steps"]}_dev_predictions.csv'
            if digest(path/'dev_predictions.csv')!=control_manifest[ref_name]['sha256']:
                raise ValueError('Dropout0.1 control predictions differ')
            predictions(path/'dev_predictions.csv',ids)
        if digest(ROOT/c['c_reference'])!=c['c_reference_sha256']:raise ValueError('C reference differs')
    shared = ROOT/c['baseline_handoff']
    index = read_json(shared/'manifest.json')['files']
    baselines = []
    for b in c['baselines']:
        for suffix in ('arguments.json','summary.json','dev_predictions.csv','a_evaluation.txt'):
            name = b['prefix']+'_'+suffix
            if digest(shared/name)!=index[name]['sha256']:
                raise ValueError(f'Historical handoff hash differs: {name}')
        s = read_json(shared/(b['prefix']+'_summary.json'))
        a = read_json(shared/(b['prefix']+'_arguments.json'))
        if (s['status']!='PASS' or s['optimizer_updates']!=b['steps'] or a['seed']!=b['seed']
                or a['max_length']!=256 or s['input_files']['dev']['sha256']!=c['dev_sha256']
                or s['input_files']['train']['sha256']!=c['train_sha256']
                or s['a_evaluation']['short_by']!='word' or s['threshold']!=.5
                or a['class_weighting']!='none' or a['learning_rate']!=2e-5
                or a['batch_size']!=1 or a['gradient_accumulation']!=16
                or s['data_provenance']['version']!=c['data_version']):
            raise ValueError(f'Historical baseline incompatible: {b["name"]}')
        predictions(shared/(b['prefix']+'_dev_predictions.csv'),ids)
        local = ROOT/b['run']
        if digest(local/'dev_predictions.csv') != digest(shared/(b['prefix']+'_dev_predictions.csv')):
            raise ValueError('Local/handoff historical predictions differ')
        if read_json(local/'run_summary.json') != s:
            # Paths can be curated in handoff copies; substantive fields must agree.
            for key in ('status','optimizer_updates','sample_order_hash','processed_samples','training_seconds'):
                if read_json(local/'run_summary.json')[key] != s[key]:
                    raise ValueError(f'Historical local summary differs: {key}')
        if not (local/'training_log.jsonl').is_file():
            raise ValueError('Historical training history unavailable; cannot audit sample exposure')
        baselines.append(dict(name=b['name'],status='REUSED_VERIFIED',steps=b['steps'],seed=b['seed']))
    dependency = {}
    for module,dist in [('torch','torch'),('transformers','transformers'),('numpy','numpy'),
                        ('sklearn','scikit-learn'),('accelerate','accelerate'),('matplotlib','matplotlib'),
                        ('fitz','PyMuPDF'),('PIL','Pillow'),('psutil','psutil')]:
        dependency[module] = importlib.metadata.version(dist) if importlib.util.find_spec(module) else None
    audits = ROOT/'main_model/figure_audit'
    missing_audits = [n for n in ('audit_panel_alignment.py','audit_pdf_text.py','audit_figure_collisions.py','validate_figure.py')
                      if not (audits/n).is_file()]
    for b in c['baselines']:
        old = read_json(shared/(b['prefix']+'_summary.json'))
        for key,module in [('torch','torch'),('transformers','transformers'),('numpy','numpy'),('accelerate','accelerate')]:
            if old['packages'][key]!=dependency[module]:
                raise ValueError(f'Training runtime differs from verified 256 baseline: {key}')
    return dict(status='CPU_PREFLIGHT_PASS',code=record,dependencies=dependency,
        missing_dependencies=[k for k,v in dependency.items() if v is None],missing_auditors=missing_audits,
        data=dict(version=c['data_version'],train_rows=len(train),dev_rows=len(dev),train_sha256=c['train_sha256'],dev_sha256=c['dev_sha256']),
        baseline_acceptance=baselines,
        missing_historical_nodes=[dict(length=256,seed=42,steps=1000,status='MISSING_NOT_RECONSTRUCTED'),
                                  dict(length=256,seed=43,steps=500,status='MISSING_NOT_RECONSTRUCTED'),
                                  dict(length=256,seed=43,steps=1000,status='MISSING_NOT_RECONSTRUCTED')],
        gpu_snapshot=process_snapshot(),source_manifest_sha256=digest(ROOT/c['model_dir']/'source_manifest.json'),
        experiment=c.get('experiment','length512_original'),classifier_dropout=c.get('classifier_dropout',.1),
        prerequisites='For dropout: current fusion PASS plus actual A receipt required before run; dry-run does not authorize training',
        evaluator_sha256=digest(ROOT/c['evaluation_script']),
        free_gib=shutil.disk_usage(ROOT).free/2**30,
        manual_visual_review='PENDING_AFTER_RUN')


def checkpoints(out,seed):
    found = {}
    for p in sorted((out/f'seed-{seed}'/'training').glob('*/attempt-*/checkpoints/step-*')):
        try:
            meta = verify_checkpoint(p)
            frozen = read_json(out/'config.json') if (out/'config.json').exists() else None
            if frozen:
                sig = meta['signature']
                for key,value in [('seed',seed),('max_length',512),('batch_size',1),('gradient_accumulation',16),
                        ('learning_rate',2e-5),('class_weighting','none'),('train_sha256',frozen['train_sha256']),
                        ('dev_sha256',frozen['dev_sha256'])]:
                    if sig.get(key)!=value:
                        raise ValueError(f'Checkpoint/config mismatch: {key}')
                if frozen.get('classifier_dropout') is not None and sig.get('classifier_dropout')!=frozen['classifier_dropout']:
                    raise ValueError('Dropout checkpoint mismatch; never resume original0.1 checkpoints')
            step = meta['optimizer_step']
            if step in found and digest(p/'model.safetensors')!=digest(found[step]/'model.safetensors'):
                raise ValueError('Conflicting checkpoint models at one node')
            found[step] = p
        except (OSError,ValueError) as exc:
            # A completed-looking but corrupt checkpoint is a blocker, not silently skipped.
            raise ValueError(f'Invalid recovery checkpoint {p}: {exc}') from exc
    return found


def train_command(c,out,seed,target,resume):
    t = c['training']
    cmd = [sys.executable,'-u','main_model/train.py','--train-file',c['train_file'],'--dev-file',c['dev_file'],
        '--model-dir',c['model_dir'],'--output-dir',str(out),'--data-kind','a-internal',
        '--data-manifest',c['data_manifest'],'--seed',str(seed),'--max-steps','2000','--max-length','512',
        '--batch-size','1','--inference-batch-size','8','--gradient-accumulation','16','--learning-rate','2e-5',
        '--class-weighting','none','--save-steps','500','--keep-checkpoints','10','--deterministic',
        '--training-only','--strict-resume-code','--protect-checkpoints','--progress-every','100']
    if target < t['max_steps']:
        cmd += ['--stop-after-steps',str(target)]
    if resume:
        cmd += ['--resume-from',str(resume)]
    if c.get('experiment')=='classifier_dropout_02':
        cmd += ['--classifier-dropout','0.2']
    return cmd


def recover_summary(c,checkpoint):
    """Recover only an exact serialized optimizer boundary, never partial gradients."""
    import torch
    from training_engine import load_training_state
    meta = verify_checkpoint(checkpoint)
    state = load_training_state(checkpoint,meta['signature'])
    counters = state['counters']
    args = read_json(checkpoint/'training_arguments.json')
    summary = dict(status='RECOVERED_CHECKPOINT_ONLY',optimizer_updates=counters['updates'],
        attempted_updates=counters['attempts'],fp16_skipped_updates=counters['attempts']-counters['updates'],
        training_loss=counters['total_loss']/counters['updates'],training_seconds=counters['training_seconds'],
        processed_samples=counters['seen'],sample_order_hash=counters['sample_order_hash'],
        peak_allocated_mib=counters.get('peak_allocated_mib'),peak_reserved_mib=counters.get('peak_reserved_mib'),
        max_observed_sequence_length=counters['max_observed_length'],
        optimizer_updates_per_second=counters['updates']/counters['training_seconds'],
        input_files={k:dict(path=c[k+'_file'],sha256=c[k+'_sha256']) for k in ('train','dev')},
        data_provenance=dict(version=c['data_version'],manifest_sha256=c['manifest_sha256']),
        threshold=.5,short_text_rule='TEMPORARY len(text.split()) < 100',
        checkpoint_recovered=True,training_arguments=args)
    return summary


def train_seed(c,out,seed,book):
    known = checkpoints(out,seed)
    for target in c['training']['segment_steps']:
        name = f'seed{seed}/train{target}'
        if target in known:
            book.set(name,'REUSED_CHECKPOINT',checkpoint=str(known[target].relative_to(out)))
        else:
            previous = max((k for k in known if k<target),default=None)
            segment = next_attempt(out/f'seed-{seed}'/'training'/f'to-{target:06d}')
            book.set(name,'RUNNING',directory=str(segment.relative_to(out)))
            from gpu_guard import require_idle_project
            require_idle_project()
            try:
                foreground(train_command(c,segment,seed,target,known[previous] if previous else None),
                           segment.parent/(segment.name+'.log'),echo=True)
                cp = segment/'checkpoints'/f'step-{target:06d}'
                verify_checkpoint(cp)
                known[target] = cp
                book.set(name,'PASS',checkpoint=str(cp.relative_to(out)))
            except BaseException as exc:
                book.set(name,'FAILED',error=repr(exc),directory=str(segment.relative_to(out)))
                raise
        if target == 50:
            book.set(f'seed{seed}/resource_gate','RUNNING')
            cp = known[50]
            s = recover_summary(c,cp)
            if s['optimizer_updates']!=50 or s['max_observed_sequence_length']!=512 or s['peak_allocated_mib'] is None:
                raise ValueError('512/50-step resource gate failed; no automatic fallback')
            dump(out/f'seed-{seed}'/'resource_gate.json',dict(status='PASS',summary=s,
                scope='Resources and full-state integrity only; no performance conclusion',
                full_state_resume_required=True))
            book.set(f'seed{seed}/resource_gate','PASS')
        if target in c['training']['evaluation_steps']:
            evaluate(c,out,seed,target,known[target],book)


def evaluate(c,out,seed,step,checkpoint,book,force=False):
    root = out/f'seed-{seed}'/'evaluation'/f'step-{step:06d}'
    if not force:
        for p in sorted(root.glob('attempt-*/run_summary.json'),reverse=True):
            s = read_json(p)
            if s.get('status')=='PASS' and s['checkpoint_manifest_sha256']==digest(checkpoint/'resume_manifest.json'):
                dev_ids = [json.loads(line)['sample_id'] for line in (ROOT/c['dev_file']).read_text(encoding='utf-8').splitlines() if line.strip()]
                predictions(p.parent/'dev_predictions.csv',dev_ids)
                inv = read_json(p.parent/'prediction_manifest.json')
                for n,item in inv['files'].items():
                    if digest(p.parent/n)!=item['sha256']:
                        raise ValueError('Completed evaluation prediction changed')
                for n,item in inv['export_files'].items():
                    if digest(p.parent/'checkpoint'/n)!=item['sha256']:
                        raise ValueError('Completed evaluation export changed')
                return p.parent
    directory = next_attempt(root)
    meta_summary = directory.parent/(directory.name+'.training_summary.json')
    dump(meta_summary,recover_summary(c,checkpoint))
    name = f'seed{seed}/predict{step}'
    book.set(name,'RUNNING',directory=str(directory.relative_to(out)))
    from gpu_guard import require_idle_project
    require_idle_project()
    try:
        foreground([sys.executable,'-u','main_model/evaluate_node.py','--checkpoint',str(checkpoint),
            '--training-summary',str(meta_summary),'--dev',c['dev_file'],'--evaluator',c['evaluation_script'],
            '--output',str(directory),'--batch','8'],directory.parent/(directory.name+'.log'))
        if read_json(directory/'run_summary.json')['status']!='PASS':
            raise ValueError('Evaluation/reload verification did not pass')
        book.set(name,'PASS',directory=str(directory.relative_to(out)))
        return directory
    except BaseException as exc:
        book.set(name,'FAILED',error=repr(exc))
        raise


def evaluation_runs(c,out):
    runs = {b['name']:ROOT/b['run'] for b in c.get('comparison_baselines',c['baselines'])}
    for seed in c['effective_seeds']:
        for step in c['training']['evaluation_steps']:
            root = out/f'seed-{seed}'/'evaluation'/f'step-{step:06d}'
            for p in sorted(root.glob('attempt-*/run_summary.json'),reverse=True):
                if read_json(p).get('status')=='PASS':
                    runs[f'{c.get("run_label","L512")}_s{seed}_u{step}'] = p.parent
                    break
    return runs


def sample_audit(c,out):
    """Hash-chain comparison at equal attempts AND optimizer nodes, with prefix checks."""
    result = []
    for seed in c['effective_seeds']:
        baseline = ROOT/next(b['run'] for b in c.get('comparison_baselines',c['baselines']) if b['seed']==seed and b['steps']==2000)
        reference = [json.loads(x) for x in (baseline/'training_log.jsonl').read_text(encoding='utf-8').splitlines()]
        for step,cp in checkpoints(out,seed).items():
            history = [json.loads(x) for x in (cp.parent.parent/'training_log.jsonl').read_text(encoding='utf-8').splitlines()]
            end_index = max(i for i,e in enumerate(history) if e['optimizer_step']==step and not e['skipped'])
            history = history[:end_index+1]
            equal_attempts = all(a['sample_order_hash']==b['sample_order_hash'] for a,b in zip(reference,history))
            ref_step = next((e for e in reference if e['optimizer_step']==step and not e['skipped']),None)
            same_exposure = ref_step is not None and ref_step['sample_order_hash']==history[-1]['sample_order_hash']
            result.append(dict(seed=seed,steps=step,attempts_512=len(history),
                attempts_256=ref_step['attempt'] if ref_step else None,
                equal_sample_prefix_at_equal_attempts=equal_attempts,
                same_exposure_at_equal_effective_updates=same_exposure,
                conclusion='MATCHED_EXPOSURE' if same_exposure else 'EXPOSURE_DIFFERS_REPORT_FP16_SKIPS',
                note='Intervention can change FP16 skips; equal attempts verify order, not equal effective gradients/exposure'))
            if c.get('experiment')=='classifier_dropout_02':
                result[-1]['attempts_intervention']=result[-1].pop('attempts_512')
                result[-1]['attempts_control']=result[-1].pop('attempts_256')
                result[-1].update(intervention_classifier_dropout=.2,control_classifier_dropout=.1,length=512)
            if not equal_attempts:
                raise ValueError('Unexpected sample-order mismatch at equal attempts; inspect trajectory before continuing')
    dump(out/'sampling_audit.json',dict(comparisons=result))


def figures(c,out,book,audit_only=False):
    if audit_only:
        candidates = sorted((out/'figures').glob('attempt-*/figure_index.json'))
        if not candidates:
            raise ValueError('No rendered figures available for audit')
        directory = candidates[-1].parent
    else:
        directory = next_attempt(out/'figures')
        directory.mkdir()
        runs = evaluation_runs(c,out)
        dump(directory/'runs.json',{k:str(v.relative_to(ROOT)) for k,v in runs.items()})
        book.set('plotting','RUNNING',directory=str(directory.relative_to(out)))
        try:
            foreground([sys.executable,'main_model/plot_local_round.py','--config',str(out/'config.json'),
                '--runs',str(directory/'runs.json'),'--output',str(directory)],directory/'plot.log')
            book.set('plotting','PASS_RENDERED_PENDING_AUDIT',directory=str(directory.relative_to(out)))
        except BaseException as exc:
            book.set('plotting','FAILED',error=repr(exc),directory=str(directory.relative_to(out)))
            raise
    book.set('automatic_audit','RUNNING',directory=str(directory.relative_to(out)))
    log = directory/f'audit-{len(list(directory.glob("audit-*.log")))+1:03d}.log'
    try:
        foreground([sys.executable,'main_model/plot_local_round.py','--audit-only','--output',str(directory)],log)
        book.set('automatic_audit','PASS',directory=str(directory.relative_to(out)),manual_visual_review='PENDING_AFTER_RUN')
    except BaseException as exc:
        book.set('automatic_audit','FAILED_OR_REVIEW_REQUIRED',error=repr(exc),directory=str(directory.relative_to(out)))
        raise


def bundle(c,out,book):
    """Small analysis export: no weights, environment, raw text or full training logs."""
    directory = next_attempt(out/'bundles')
    directory.mkdir()
    if c.get('experiment')=='classifier_dropout_02':
        from dropout_results import generate
        generate(c,out,directory/'regularization_comparison')
    safe = book.data['stages'].get('preflight',{}).get('status')=='PASS'
    runs = evaluation_runs(c,out) if safe else {}
    records = []
    for name,path in runs.items():
        if not (path/'run_summary.json').exists():
            continue
        s = read_json(path/'run_summary.json')
        from plot_local_round import metrics_for_run
        metric,_ = metrics_for_run(ROOT/c['dev_file'],path,c)
        records.append(dict(run=name,steps=s['optimizer_updates'],**metric,
            length=read_json(path/'arguments.json')['max_length'],seed=read_json(path/'arguments.json')['seed'],
            attempts=s['attempted_updates'],fp16_skips=s['fp16_skipped_updates'],processed_samples=s['processed_samples'],
            active_training_seconds=s['training_seconds'],before_save_inference_seconds=s.get('before_save_inference_seconds'),
            reload_inference_seconds=s.get('independent_reload_and_inference_seconds'),
            peak_allocated_mib=s['peak_allocated_mib'],peak_reserved_mib=s['peak_reserved_mib']))
        for suffix in ('dev_predictions.csv','a_evaluation_report.txt','internal_dev_metrics.json','prediction_manifest.json'):
            if (path/suffix).is_file():
                shutil.copy2(path/suffix,directory/(name+'_'+suffix))
    csv_out(directory/'summary.csv',records)
    book.set('analysis_bundle','PASS',directory=str(directory.relative_to(out)),
             archive=str((directory.parent/(directory.name+'_analysis_bundle.zip')).relative_to(out)))
    artifacts = {}
    for branch in ('figures',*[f'seed-{s}/evaluation' for s in c['effective_seeds']]):
        for f in (out/branch).rglob('*'):
            if f.is_file() and f.suffix.lower() in ('.csv','.png','.pdf','.svg'):
                artifacts[f.relative_to(out).as_posix()]=dict(bytes=f.stat().st_size,sha256=digest(f))
    dump(out/'artifact_manifest.json',dict(files=artifacts,scope='Predictions, figure exports and figure source tables; weights excluded'))
    for name in ('config.json','preflight.json','stages.json','sampling_audit.json','failure.json','protocol.json','artifact_manifest.json','latest_outcome.json'):
        if (out/name).exists():
            shutil.copy2(out/name,directory/name)
    errors=[]
    for f in out.glob('seed-*/**/failure.json'):
        errors.append(dict(path=f.relative_to(out).as_posix(),details=read_json(f)))
    for f in out.glob('failures/*/failure.json'):
        errors.append(dict(path=f.relative_to(out).as_posix(),details=read_json(f)))
    for name,item in book.data['stages'].items():
        if 'FAIL' in item['status'] and 'directory' in item:
            folder=out/item['directory']
            log=folder.parent/(folder.name+'.log')
            logs=[log] if log.is_file() else [f for f in [folder/'plot.log',*folder.glob('audit-*.log')] if f.is_file()]
            for log in logs:
                errors.append(dict(stage=name,log=log.relative_to(out).as_posix(),
                    tail=log.read_text(encoding='utf-8',errors='replace').splitlines()[-60:]))
    dump(directory/'failure_details.json',errors)
    dump(directory/'run_metadata.json',{name:dict(summary=read_json(path/'run_summary.json'),
        arguments=read_json(path/'arguments.json')) for name,path in runs.items()})
    dump(directory/'resource_gates.json',{str(seed):read_json(out/f'seed-{seed}'/'resource_gate.json')
        for seed in c['effective_seeds'] if (out/f'seed-{seed}'/'resource_gate.json').is_file()})
    figures_index = []
    for p in sorted((out/'figures').glob('attempt-*/figure_index.json')):
        figures_index.append(dict(directory=str(p.parent.relative_to(out)),index=read_json(p)))
        dest = directory/p.parent.name
        dest.mkdir(exist_ok=True)
        for f in p.parent.iterdir():
            if f.is_file() and f.suffix in ('.png','.pdf','.svg','.csv','.json'):
                shutil.copy2(f,dest/f.name)
    dump(directory/'figures.json',figures_index)
    text = ['# 实测结果摘要','', '仅内部internal-v2 dev；阈值0.5；短文本为临时空白计词<100。',
        '人工视觉复核：运行后待检查。缺失节点不插值、不重训补齐。', '']
    text += [f"- {r['run']}: 全体宏F1={r['macro_f1_all']:.6f}, 临时S={r['weighted_score_temporary']:.6f}, "
             f"human召回={r['human_recall']:.6f}, log loss={r['log_loss']:.6f}, Brier={r['brier']:.6f}" for r in records]
    text += ['', '阶段状态和失败原因见stages.json/failure.json；缺少512行表示该节点尚无通过验证的实测预测。',
             '抽样暴露是否相同见sampling_audit.json；不按中途分数改变计划或选择非预定模型。']
    (directory/'RESULTS.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    files = {p.relative_to(directory).as_posix():dict(bytes=p.stat().st_size,sha256=digest(p))
             for p in directory.rglob('*') if p.is_file()}
    dump(directory/'manifest.json',dict(files=files,excluded=['weights','environment','raw text','complete training logs']))
    archive = directory.parent/(directory.name+'_analysis_bundle.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in directory.rglob('*'):
            if p.is_file():
                z.write(p,p.relative_to(directory))
    book.set('analysis_bundle','PASS',directory=str(directory.relative_to(out)),archive=str(archive.relative_to(out)),sha256=digest(archive))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='main_model/configs/length512_v2.json')
    p.add_argument('--output',default='runs/b_v2_length512_round')
    p.add_argument('--phase',choices=['dry-run','run','resume','evaluate','plot','audit','bundle'],default='dry-run')
    seeds=p.add_mutually_exclusive_group()
    seeds.add_argument('--include-seed43',action='store_true')
    seeds.add_argument('--seed43-only',action='store_true',help='Independent seed43 output; never repeat seed42')
    p.add_argument('--dry-run-report',default='.local/b512_prepare/dry_run.json')
    args = p.parse_args()
    c = load_config(ROOT/args.config,args.include_seed43,args.seed43_only)
    if c.get('experiment')=='classifier_dropout_02' and args.phase in ('run','resume'):
        gate=ROOT/c['integration_gate']
        if not gate.is_file():raise ValueError('Complete current unified fusion acceptance before starting dropout training')
        accepted=read_json(gate)
        if accepted.get('status')!='PASS_B_LOCAL_PENDING_A' or accepted.get('bundle_sha256')!=c['integration_bundle_sha256']:
            raise ValueError('Current fusion gate missing, failed, or refers to a different candidate')
        receipt=ROOT/c['independent_acceptance_receipt']
        if not receipt.is_file():raise ValueError('Wait for A independent inference acceptance receipt before dropout experiment')
        received=read_json(receipt)
        if received.get('status')!='PASS' or received.get('reviewer')!='A' or received.get('bundle_sha256')!=c['integration_bundle_sha256'] or not received.get('evidence_url','').startswith('https://github.com/'):
            raise ValueError('A receipt must record actual PASS, exact bundle SHA and traceable GitHub evidence')
    out = (ROOT/args.output).resolve()
    if not out.is_relative_to(ROOT/'runs') or out == ROOT/'runs':
        raise ValueError('Output must be a dedicated directory within runs/')
    if args.phase=='dry-run':
        try:
            result = preflight(c)
            result['planned_commands'] = {str(seed):train_command(c,out/f'seed-{seed}'/'training'/'to-000050'/'attempt-001',seed,50,None) for seed in c['effective_seeds']}
            dump(ROOT/args.dry_run_report,result)
            print(json.dumps(result,ensure_ascii=False,indent=2))
            if result['missing_dependencies'] or result['missing_auditors']:
                raise RuntimeError('Missing dependencies/auditors; see dry-run report')
        except BaseException as exc:
            dump(ROOT/(args.dry_run_report+'.failure.json'),dict(error=repr(exc),traceback=traceback.format_exc()))
            raise
        return
    if args.phase=='run':
        out.mkdir(parents=True,exist_ok=False)
        dump(out/'config.json',c)
    else:
        if read_json(out/'config.json') != c:
            raise ValueError('Effective config differs from frozen run; do not alter protocol during resume')
    book = StageBook(out)
    if args.phase=='run':
        for seed in c['effective_seeds']:
            for target in c['training']['segment_steps']:
                book.set(f'seed{seed}/train{target}','PENDING')
            book.set(f'seed{seed}/resource_gate','PENDING')
            for target in c['training']['evaluation_steps']:
                book.set(f'seed{seed}/predict{target}','PENDING')
        for stage in ('plotting','automatic_audit','analysis_bundle'):
            book.set(stage,'PENDING')
        if 43 not in c['effective_seeds']:
            book.set('seed43_optional','NOT_REQUESTED')
    caught = None
    try:
        book.set('preflight','RUNNING')
        check = preflight(c)
        dump(out/'preflight.json',check)
        history=next_attempt(out/'preflights')
        history.mkdir()
        dump(history/'preflight.json',check)
        if check['missing_dependencies'] or check['missing_auditors']:
            raise RuntimeError('Required local dependencies/auditors missing')
        protocol_path = out/'protocol.json'
        training_names = ('main_model/train.py','main_model/training_engine.py','main_model/common.py')
        if c.get('experiment')=='classifier_dropout_02':training_names+=('main_model/dropout_protocol.py',)
        code = {n:check['code']['files'][n] for n in training_names}
        if protocol_path.exists():
            old = read_json(protocol_path)
            if old['training_code'] != code and args.phase in ('run','resume'):
                raise ValueError('Training code differs from frozen trajectory; do not silently resume')
            if old['training_runtime']!={n:check['dependencies'][n] for n in ('torch','transformers','numpy','accelerate')}:
                raise ValueError('Training runtime versions differ from frozen trajectory')
        else:
            dump(protocol_path,dict(training_code=code,initial_code=check['code'],
                training_runtime={n:check['dependencies'][n] for n in ('torch','transformers','numpy','accelerate')},
                predeclared_nodes=c['training'],manual_visual_review='PENDING_AFTER_RUN',
                optimizer='AdamW lr2e-5 wd.01 foreach=False max_grad_norm1; constant LR',
                missing_historical_nodes=check['missing_historical_nodes']))
        book.set('preflight','PASS')
        needs_gpu = args.phase in ('run','resume','evaluate')
        if args.phase in ('run','resume') and check['free_gib'] < c['minimum_free_gib_per_seed']*len(c['effective_seeds']):
            raise RuntimeError('Insufficient free disk for protected full-state checkpoints')
        from gpu_guard import ProjectGpuLock
        with ProjectGpuLock() if needs_gpu else nullcontext():
            if args.phase in ('run','resume'):
                for seed in c['effective_seeds']:
                    train_seed(c,out,seed,book)
                sample_audit(c,out)
            if args.phase=='evaluate':
                for seed in c['effective_seeds']:
                    known = checkpoints(out,seed)
                    missing=[step for step in c['training']['evaluation_steps'] if step not in known]
                    if missing:
                        raise ValueError(f'Evaluation requires saved seed{seed} nodes {missing}; no training launched')
                    for step in c['training']['evaluation_steps']:
                        evaluate(c,out,seed,step,known[step],book,force=True)
                sample_audit(c,out)
        if args.phase in ('run','resume','plot'):
            figures(c,out,book)
        if args.phase=='audit':
            figures(c,out,book,audit_only=True)
        book.set('requested_phase','PASS_PENDING_MANUAL_REVIEW',phase=args.phase)
        dump(out/'latest_outcome.json',dict(status='PASS_PENDING_MANUAL_REVIEW',phase=args.phase,
            historical_failure_file_may_remain=True))
    except BaseException as exc:
        caught = exc
        for name,item in list(book.data['stages'].items()):
            if item['status']=='RUNNING':
                book.set(name,'FAILED',error=repr(exc))
        book.set('requested_phase','FAILED',phase=args.phase,error=repr(exc))
        error=dict(phase=args.phase,error=repr(exc),traceback=traceback.format_exc())
        dump(out/'failure.json',error)
        failed=next_attempt(out/'failures')
        failed.mkdir()
        dump(failed/'failure.json',error)
        dump(out/'latest_outcome.json',dict(status='FAILED',**error))
    finally:
        try:
            bundle(c,out,book)
        except BaseException as exc:
            book.set('analysis_bundle','FAILED',error=repr(exc))
            if caught is None:
                caught = exc
                book.set('requested_phase','FAILED',phase=args.phase,error=repr(exc))
                dump(out/'latest_outcome.json',dict(status='FAILED',phase=args.phase,error=repr(exc)))
    if caught:
        raise caught
    print('Finished. Manual visual review remains pending. See stages.json and bundles/.',flush=True)


if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        if not isinstance(exc,SystemExit) or exc.code:
            dump(ROOT/'.local/b512_prepare/last_launcher_failure.json',
                 dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
