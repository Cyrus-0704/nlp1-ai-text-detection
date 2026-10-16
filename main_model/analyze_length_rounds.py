"""CPU-only analysis/curation of completed fixed-node length rounds.

Never loads weights, starts inference, tunes weights/thresholds, or reads test.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import numpy as np
from sklearn.metrics import (brier_score_loss, confusion_matrix, f1_score,
                             log_loss, roc_auc_score)
from threadpoolctl import threadpool_limits

from common import read_rows
from round_support import ROOT, csv_out, digest, dump, predictions, read_json


def portable(value):
    if isinstance(value, dict):
        return {k: portable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [portable(v) for v in value]
    if isinstance(value, str):
        s = value.replace('\\', '/')
        prefix = ROOT.as_posix() + '/'
        return s[len(prefix):] if s.startswith(prefix) else s
    return value


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finalize_handoff(out):
    """Use checkout-stable LF text hashes without changing rendered geometry."""
    fig = out/'figures_seed43'
    original_path = out/'delivery_text_normalization.json'
    if not original_path.exists():
        dump(original_path,dict(rule='CRLF to LF in curated JSON/TXT/MD/SVG only; CSV LFS bytes and PDF/PNG unchanged',
            original_runtime_figure_files=read_json(fig/'figure_manifest.json')['files'],
            note='Original runtime bundle and its manifest untouched; delivery manifest records checkout-stable bytes'))
    for path in out.rglob('*'):
        if path.is_file() and path.suffix in ('.json','.txt','.md','.svg'):
            path.write_bytes(path.read_bytes().replace(b'\r\n',b'\n'))
    meta = read_json(fig/'figure_manifest.json')
    for name in meta['files']:
        path = fig/name
        meta['files'][name] = dict(bytes=path.stat().st_size,sha256=digest(path))
    meta['delivery_normalization'] = 'LF text; unchanged PDF/PNG; see ../delivery_text_normalization.json'
    (fig/'figure_manifest.json').write_bytes((json.dumps(meta,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    files = {p.relative_to(out).as_posix():dict(bytes=p.stat().st_size,sha256=digest(p))
             for p in sorted(out.rglob('*')) if p.is_file() and p != out/'manifest.json'}
    meta = dict(scope='Necessary B dev handoff; weights, raw text, environments and full logs excluded',
        files=files,total_files=len(files),total_bytes=sum(x['bytes'] for x in files.values()),
        git_delivery='See PR4 merge status; publication is not recipient acceptance')
    (out/'manifest.json').write_bytes((json.dumps(meta,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run42', type=Path, default=ROOT/'runs/b_v2_length512_round')
    parser.add_argument('--run43', type=Path, default=ROOT/'runs/b_v2_length512_seed43')
    parser.add_argument('--c-predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New directory only')
    args = parser.parse_args()
    threadpool_limits(1)
    out = args.output.resolve()
    require(not out.exists(), 'Output already exists; use a new directory')
    configs = {s: read_json(r/'config.json') for s, r in ((42,args.run42),(43,args.run43))}
    require(configs[42]['effective_seeds'] == [42] and configs[43]['effective_seeds'] == [43],
            'Expected separate completed seed42/43 rounds')
    for key in configs[42]:
        if key != 'effective_seeds':
            require(configs[42][key] == configs[43][key], f'Cross-seed config differs: {key}')
    config = configs[42]
    for key, sha in [('train_file','train_sha256'),('dev_file','dev_sha256'),
                     ('data_manifest','manifest_sha256'),('evaluation_script','evaluator_sha256')]:
        require(digest(ROOT/config[key]) == config[sha], f'Frozen input changed: {key}')
    rows = read_rows(ROOT/config['dev_file'], labeled=True)
    ids = [r['sample_id'] for r in rows]
    y = np.array([r['label'] for r in rows])
    short = np.array([len(r['text'].split()) < 100 for r in rows])
    groups, gi = np.unique([r['group_id'] for r in rows], return_inverse=True)
    require(len(rows) == 15086 and short.sum() == 1841, 'Unexpected fixed-dev size')
    require(digest(args.c_predictions) == '031662988c4459b0cc30f1a8b45b89f396568b8b4b844789a88fc4362adfcabd',
            'C probability file differs from reviewed handoff')
    paths = {b['name']:ROOT/b['run']/'dev_predictions.csv' for b in config['baselines']}
    audits, copies, resources = {}, [], []
    for seed, run in ((42,args.run42),(43,args.run43)):
        stages = read_json(run/'stages.json')['stages']
        require(all(v['status'].startswith('PASS') or
                    (k == 'seed43_optional' and seed == 42 and v['status'] == 'NOT_REQUESTED')
                    for k,v in stages.items()), 'Incomplete/failed runtime stage')
        archive = run/'bundles/attempt-001_analysis_bundle.zip'
        require(digest(archive) == stages['analysis_bundle']['sha256'], 'Bundle SHA differs')
        with zipfile.ZipFile(archive) as z:
            manifest = json.loads(z.read('manifest.json'))['files']
            for name, item in manifest.items():
                require(hashlib.sha256(z.read(name)).hexdigest() == item['sha256'], f'Bundle member differs: {name}')
        artifacts = read_json(run/'artifact_manifest.json')['files']
        for name, item in artifacts.items():
            require(digest(run/name) == item['sha256'], f'Run artifact differs: {name}')
        protocol = read_json(run/'protocol.json')
        require(protocol['training_code'] == read_json(args.run42/'protocol.json')['training_code'], 'Training core changed across seeds')
        for name, sha in protocol['training_code'].items():
            require(digest(ROOT/name) == sha, f'Training core differs now: {name}')
        audit = read_json(run/'figures/attempt-001/automatic_audit.json')
        require(audit['status'] == 'PASS' and all(r['exit_code'] == 0 for r in audit['records']), 'Figure audit failed')
        audits[str(seed)] = dict(bundle_sha256=digest(archive), bundle_files_verified=len(manifest),
            artifact_files_verified=len(artifacts), automatic_figure_checks=len(audit['records']),
            training_code=protocol['training_code'], independent_reload={})
        for step in (500,1000,2000):
            folder = run/f'seed-{seed}/evaluation/step-{step:06d}/attempt-001'
            summary = read_json(folder/'run_summary.json')
            require(summary['status'] == 'PASS' and summary['evaluated_optimizer_step'] == step, 'Node incomplete')
            require(summary['a_evaluation'] == dict(script_sha256=config['evaluator_sha256'],short_by='word',threshold=.5,short_threshold=100), 'Evaluation protocol differs')
            require(summary['prediction_rows'] == len(rows) and summary['reload_max_abs_probability_difference'] == 0,
                    'Saved/reloaded predictions inconsistent')
            require(summary['inference_precision'] == 'fp32' and summary['inference_batch_size'] == 8, 'Inference settings differ')
            cp = run/f'seed-{seed}/training/to-{step:06d}/attempt-001/checkpoints/step-{step:06d}'
            require(digest(cp/'resume_manifest.json') == summary['checkpoint_manifest_sha256'], 'Checkpoint manifest differs')
            name = f'L512_s{seed}_u{step}'
            paths[name] = folder/'dev_predictions.csv'
            copies.append((name,folder))
            resources.append(dict(run=name,**{k:summary[k] for k in ('optimizer_updates','attempted_updates',
                'fp16_skipped_updates','processed_samples','training_seconds','peak_allocated_mib','peak_reserved_mib',
                'before_save_inference_seconds','independent_reload_and_inference_seconds')}))
            audits[str(seed)]['independent_reload'][str(step)] = 0
    paths['C_TFIDF'] = args.c_predictions
    p = {n:np.array(predictions(path,ids)) for n,path in paths.items()}
    # Reuse C's already declared weight: no new search, no checkpoint selection.
    for name in ('L256_s43_u2000','L512_s42_u2000','L512_s43_u2000'):
        p['C07_B03_'+name] = .7*p['C_TFIDF']+.3*p[name]

    def metrics(prob, mask=None):
        mask = np.ones(len(y),bool) if mask is None else mask
        yy, pp = y[mask], prob[mask]
        cm = confusion_matrix(yy,pp >= .5,labels=[0,1])
        both = len(np.unique(yy)) == 2
        sm = short[mask]
        aa = float(f1_score(yy,pp >= .5,average='macro')) if both else None
        ss = float(f1_score(yy[sm],pp[sm] >= .5,average='macro')) if sm.any() and len(np.unique(yy[sm])) == 2 else None
        return dict(samples=int(mask.sum()),macro_f1_all=aa,macro_f1_short_temporary=ss,
            weighted_score_temporary=.7*aa+.3*ss if aa is not None and ss is not None else None,
            human_recall=float(cm[0,0]/cm[0].sum()) if cm[0].sum() else None,
            machine_recall=float(cm[1,1]/cm[1].sum()) if cm[1].sum() else None,
            human_false_positive=int(cm[0,1]),machine_false_negative=int(cm[1,0]),
            errors=int(cm[0,1]+cm[1,0]),roc_auc=float(roc_auc_score(yy,pp)) if both else None,
            log_loss=float(log_loss(yy,pp,labels=[0,1])),brier=float(brier_score_loss(yy,pp)))

    all_metrics = {n:metrics(prob) for n,prob in p.items()}
    names = list(p)
    group_cm = np.zeros((len(groups),len(names),8),dtype=np.int64)
    for j,name in enumerate(names):
        idx = y*2+(p[name] >= .5)
        np.add.at(group_cm[:,j,:4],(gi,idx),1)
        np.add.at(group_cm[:,j,4:],(gi[short],idx[short]),1)

    def f1cm(cm):
        tn,fp,fn,tp = np.moveaxis(cm,-1,0)
        return (np.divide(2*tn,2*tn+fp+fn,out=np.zeros_like(tn,dtype=float),where=(2*tn+fp+fn)!=0)
            +np.divide(2*tp,2*tp+fp+fn,out=np.zeros_like(tp,dtype=float),where=(2*tp+fp+fn)!=0))/2

    rng = np.random.default_rng(20261016)
    boot = np.zeros((1000,len(names),3))
    for rep in range(1000):
        cm = group_cm[rng.integers(0,len(groups),len(groups))].sum(axis=0)
        aa,ss = f1cm(cm[:,:4]),f1cm(cm[:,4:])
        boot[rep] = np.stack([aa,ss,.7*aa+.3*ss],axis=1)
    pairs = [('L512_s42_u2000','L256_s42_u2000'),('L512_s43_u2000','L256_s43_u2000'),
        ('L512_s43_u2000','L512_s42_u2000'),('L512_s43_u2000','L512_s43_u1000'),
        ('L512_s43_u2000','C_TFIDF'),('C07_B03_L512_s43_u2000','C_TFIDF'),
        ('C07_B03_L512_s43_u2000','C07_B03_L256_s43_u2000'),
        ('C07_B03_L512_s43_u2000','C07_B03_L512_s42_u2000')]
    comparisons = []
    for new,old in pairs:
        d = boot[:,names.index(new)]-boot[:,names.index(old)]
        stats = {key:dict(delta=all_metrics[new][key]-all_metrics[old][key],ci95=np.quantile(d[:,k],[.025,.975]).tolist())
            for k,key in enumerate(('macro_f1_all','macro_f1_short_temporary','weighted_score_temporary'))}
        old_ok,new_ok = (p[old]>=.5)==y,(p[new]>=.5)==y
        comparisons.append(dict(new=new,old=old,metrics=stats,old_only_correct=int((old_ok&~new_ok).sum()),
            new_only_correct=int((~old_ok&new_ok).sum()),both_wrong=int((~old_ok&~new_ok).sum())))
    out.mkdir(parents=True)
    csv_out(out/'metrics.csv',[dict(run=n,**m) for n,m in all_metrics.items()])
    csv_out(out/'resources.csv',resources)
    dump(out/'paired_comparisons.json',dict(unit='dev group_id',groups=len(groups),replicates=1000,seed=20261016,
        comparisons=comparisons,limitation='Conditional fitted-model and same-dev selection intervals; not seed uncertainty or independent generalization; no multiplicity correction'))
    slices = []
    for field in ('domain','generator'):
        for value in sorted({r[field] for r in rows}):
            mask = np.array([r[field] == value for r in rows])
            for n in ('L512_s42_u2000','L512_s43_u2000','C_TFIDF','C07_B03_L512_s43_u2000'):
                slices.append(dict(field=field,value=value,run=n,**metrics(p[n],mask)))
    csv_out(out/'slices.csv',slices)
    quality = []
    for name,prob in p.items():
        pp = np.clip(prob,np.finfo(float).eps,1-np.finfo(float).eps)
        nll = -np.where(y==1,np.log(pp),np.log1p(-pp))
        quality.append(dict(run=name,confident_human_errors_p_ge_099=int(((y==0)&(prob>=.99)).sum()),
            confident_machine_errors_p_le_001=int(((y==1)&(prob<=.01)).sum()),
            human_nll=float(nll[y==0].mean()),machine_nll=float(nll[y==1].mean())))
    csv_out(out/'confidence_diagnostics.csv',quality)
    for name,folder in copies:
        shutil.copy2(folder/'dev_predictions.csv',out/(name+'_dev_predictions.csv'))
        for src,suffix in [('run_summary.json','summary.json'),('arguments.json','arguments.json')]:
            dump(out/(name+'_'+suffix),portable(read_json(folder/src)))
        text = (folder/'a_evaluation_report.txt').read_text(encoding='utf-8')
        (out/(name+'_a_evaluation_word.txt')).write_text(text.replace(str(ROOT)+'\\','').replace(ROOT.as_posix()+'/',''),encoding='utf-8')
    for seed,run in ((42,args.run42),(43,args.run43)):
        for file in ('config.json','sampling_audit.json','protocol.json','stages.json'):
            dump(out/f'seed{seed}_{file}',portable(read_json(run/file)))
    dump(out/'source_manifest.json',read_json(ROOT/config['model_dir']/'source_manifest.json'))
    name = 'C07_B03_L512_s43_u2000'
    csv_out(out/(name+'_dev_predictions.csv'),[dict(sample_id=i,p_machine=float(v)) for i,v in zip(ids,p[name])])
    predictions(out/(name+'_dev_predictions.csv'),ids)
    command = [sys.executable,'-X','utf8',config['evaluation_script'],'--gold',config['dev_file'],
        '--pred',str(out/(name+'_dev_predictions.csv')),'--threshold','0.5','--short-by','word','--short-threshold','100']
    report = subprocess.check_output(command,cwd=ROOT,text=True,encoding='utf-8')
    (out/'fixed_fusion_a_evaluation_word.txt').write_text(report.replace(str(ROOT)+'\\','').replace(ROOT.as_posix()+'/',''),encoding='utf-8')
    # Keep the new seed's complete editable figure set; no duplicate seed42/baseline exports.
    fig = out/'figures_seed43'
    fig.mkdir()
    for source in (args.run43/'figures/attempt-001').iterdir():
        if source.is_file() and source.suffix in ('.png','.pdf','.svg','.csv','.json'):
            if source.suffix == '.json':
                dump(fig/source.name,portable(read_json(source)))
            else:
                shutil.copy2(source,fig/source.name)
    dump(out/'audit.json',dict(status='PASS_CPU_POSTRUN_CHECK',data_version='internal-v2',dev_rows=len(rows),
        short_rows=int(short.sum()),groups=len(groups),train_sha256=config['train_sha256'],dev_sha256=config['dev_sha256'],
        seeds=audits,inputs={n:dict(path=portable(str(path.resolve())),sha256=digest(path)) for n,path in paths.items()},
        c_weight=.7,b_weight=.3,new_weight_search=False,gpu_inference=False,weights_loaded=False,
        original_run_files_modified=False,manual_visual_review='SEE_VISUAL_REVIEW_JSON'))
    finalize_handoff(out)
    print(json.dumps(dict(metrics=all_metrics,comparisons=comparisons),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
