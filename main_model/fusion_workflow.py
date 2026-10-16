"""Foreground preparation and acceptance of the existing fixed B/C candidate."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
import traceback
import zipfile

import numpy as np
from round_support import ROOT,StageBook,digest,dump,foreground,next_attempt,predictions,read_json,git_record
from common import read_rows,write_predictions
from fusion_inference import c_probabilities,verify_bundle

C_SHA='6d231819e61d24b2f9b8ee91b07b72f6e7c0f8c26d93c500d419d791211c46f8'
C_REF_SHA='031662988c4459b0cc30f1a8b45b89f396568b8b4b844789a88fc4362adfcabd'
DEV_SHA='1d80e7b6045e0914f224dcbc3afe5f56569f5bd995f742607389c26316476600'
B_HANDOFF=ROOT/'handoffs/B/b_v2_length512_two_seed'
B_EXPORT=ROOT/'runs/b_v2_length512_seed43/seed-43/evaluation/step-002000/attempt-001/checkpoint'
C_MODEL=ROOT/'handoffs/C/c_v2_round1/baseline.joblib'
C_REF=ROOT/'handoffs/C/c_v2_round1/baseline_dev_predictions.csv'


def preflight():
    if git_record()['branch']!='work/b-model':raise ValueError('Keep B on work/b-model')
    if digest(ROOT/'data/processed/v2/dev.jsonl')!=DEV_SHA:raise ValueError('Fixed dev changed')
    if digest(C_MODEL)!=C_SHA or digest(C_REF)!=C_REF_SHA:raise ValueError('C LFS file/hash mismatch')
    s=read_json(B_HANDOFF/'L512_s43_u2000_summary.json')
    if s['status']!='PASS' or s['reload_max_abs_probability_difference']!=0:raise ValueError('B reference not accepted')
    cfg=read_json(B_EXPORT/'config.json')
    effective=cfg.get('classifier_dropout') if cfg.get('classifier_dropout') is not None else cfg['hidden_dropout_prob']
    if effective!=.1 or cfg['detection_max_length']!=512:raise ValueError('Expected original 0.1 B candidate')
    return dict(status='PASS_CPU_PREFLIGHT',code=git_record(),C_sha256=C_SHA,dev_sha256=DEV_SHA,
        free_gib=shutil.disk_usage(ROOT).free/2**30,
        official_contract='Only probability output specified; .5/word<100 are internal conventions, not blockers')


def cpu_check(out,book,previous=None):
    if previous is not None:
        previous=previous.resolve();old=read_json(previous/'C_cpu_acceptance.json')
        if old['status']!='PASS' or old['model_sha256']!=C_SHA or old['dev_sha256']!=DEV_SHA or old['reference_sha256']!=C_REF_SHA:
            raise ValueError('CPU acceptance cache does not match frozen inputs')
        import importlib.metadata
        if any(importlib.metadata.version(n)!=v for n,v in old['packages'].items()):raise ValueError('CPU runtime changed')
        import ast,hashlib
        def c_function_sha(path):
            tree=ast.parse(path.read_text(encoding='utf-8'))
            function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='c_probabilities')
            return hashlib.sha256(ast.dump(function,include_attributes=False).encode()).hexdigest()
        if c_function_sha(previous/'inference_bundle/fusion_inference.py')!=c_function_sha(ROOT/'main_model/fusion_inference.py') or digest(previous/'inference_bundle/common.py')!=digest(ROOT/'main_model/common.py'):
            raise ValueError('C inference implementation changed; cannot reuse CPU acceptance')
        old['C_function_canonical_sha256']=c_function_sha(ROOT/'main_model/fusion_inference.py')
        stage=read_json(previous/'stages.json')['stages']['C_full_dev_cpu']
        original=previous/stage['directory']/'C_dev_predictions.csv'
        if digest(original)!=old['prediction_sha256']:raise ValueError('Cached CPU predictions changed')
        old['reused_from']=str(previous.relative_to(ROOT))
        dump(out/'C_cpu_acceptance.json',old)
        book.set('C_full_dev_cpu','PASS',result=old,reused=True)
        return
    folder=next_attempt(out/'cpu_check');folder.mkdir(parents=True)
    book.set('C_full_dev_cpu','RUNNING',directory=str(folder.relative_to(out)))
    rows=read_rows(ROOT/'data/processed/v2/dev.jsonl',labeled=True)
    ids=[r['sample_id'] for r in rows]
    expected=predictions(C_REF,ids)
    import time,importlib.metadata,warnings
    tick=time.perf_counter()
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter('always');actual=c_probabilities(C_MODEL,rows)
    np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-8)
    write_predictions(folder/'C_dev_predictions.csv',rows,actual)
    result=dict(status='PASS',rows=len(rows),max_abs_probability_difference=float(np.max(np.abs(actual-expected))),
        rtol=1e-6,atol=1e-8,seconds=time.perf_counter()-tick,model_sha256=C_SHA,dev_sha256=DEV_SHA,
        warnings=[str(w.message) for w in observed],reference_sha256=C_REF_SHA,
        prediction_sha256=digest(folder/'C_dev_predictions.csv'),
        packages={n:importlib.metadata.version(n) for n in ('numpy','scipy','scikit-learn','joblib')})
    dump(folder/'result.json',result);dump(out/'C_cpu_acceptance.json',result)
    book.set('C_full_dev_cpu','PASS',directory=str(folder.relative_to(out)),result=result)


def prepare(out,book):
    destination=out/'inference_bundle'
    if destination.exists():raise FileExistsError('Bundle exists; never replace it')
    if shutil.disk_usage(ROOT).free < 2*2**30:raise RuntimeError('Need at least2 GiB for inference bundle')
    book.set('package','RUNNING')
    inv=read_json(B_EXPORT.parent/'prediction_manifest.json')['export_files']
    for name,item in inv.items():
        if digest(B_EXPORT/name)!=item['sha256']:raise ValueError('B saved inference export changed: '+name)
    destination.mkdir();(destination/'B').mkdir()
    allowed=('config.json','model.safetensors','tokenizer.json','tokenizer_config.json','special_tokens_map.json',
             'sentencepiece.bpe.model','source_manifest.json')
    for name in allowed:
        if (B_EXPORT/name).is_file():shutil.copy2(B_EXPORT/name,destination/'B'/name)
    shutil.copy2(C_MODEL,destination/'C.joblib')
    for name in ('fusion_inference.py','common.py','gpu_guard.py','verify_fusion_output.py'):
        shutil.copy2(ROOT/'main_model'/name,destination/name)
    (destination/'references').mkdir()
    for component,path in [('C',C_REF),('B',B_HANDOFF/'L512_s43_u2000_dev_predictions.csv'),
            ('predictions',B_HANDOFF/'C07_B03_L512_s43_u2000_dev_predictions.csv')]:
        shutil.copy2(path,destination/'references'/(component+'.csv'))
    shutil.copy2(ROOT/'requirements/fusion.txt',destination/'requirements.txt')
    (destination/'INFERENCE.md').write_text('# Development candidate, pending independent acceptance\n\n'
        'Run offline from your chosen working directory:\n'
        'python PATH_TO_BUNDLE/fusion_inference.py --bundle PATH_TO_BUNDLE --input INPUT.jsonl --output NEW_OUTPUT_DIRECTORY\n'
        'Output predictions.csv: string sample_id,p_machine in input order. C=.7/B=.3; FP32/batch8/512.\n'
        'No optimizer or training environment included. Official CSV template is unspecified.\n',encoding='utf-8')
    files={p.relative_to(destination).as_posix():dict(bytes=p.stat().st_size,sha256=digest(p))
           for p in destination.rglob('*') if p.is_file()}
    dump(destination/'bundle.json',dict(status='DEVELOPMENT_CANDIDATE',fusion={'C':.7,'B':.3},length=512,
        precision='fp32',batch=8,seed=43,optimizer_updates=2000,files=files,
        source_revision='e73636d4f797dec63c3081bb6ed5c7b0bb3f2089',C_training_source='Historical fingerprint missing, disclosed by C',
        code=git_record(),independent_acceptance='PENDING_A',internal_threshold=.5,internal_short='whitespace words<100'))
    verify_bundle(destination)
    book.set('package','PASS',bytes=sum(x['bytes'] for x in files.values()),files=len(files))


def reference_check(folder,rows):
    all_rows=read_rows(ROOT/'data/processed/v2/dev.jsonl',labeled=True)
    all_ids=[r['sample_id'] for r in all_rows];ids=[r['sample_id'] for r in rows]
    records={}
    for component,path in [('C',C_REF),('B',B_HANDOFF/'L512_s43_u2000_dev_predictions.csv'),
            ('predictions',B_HANDOFF/'C07_B03_L512_s43_u2000_dev_predictions.csv')]:
        mapping=dict(zip(all_ids,predictions(path,all_ids)))
        actual=np.array(predictions(folder/(component+'.csv'),ids))
        expected=np.array([mapping[i] for i in ids])
        # B and fusion retain FP32 inference tolerance; C reference has10 significant CSV digits.
        rt,at=(1e-6,1e-8) if component=='C' else (1e-5,1e-6)
        np.testing.assert_allclose(actual,expected,rtol=rt,atol=at)
        if not np.array_equal(actual>=.5,expected>=.5):raise ValueError('Threshold decisions differ: '+component)
        records[component]=dict(max_abs_difference=float(np.max(np.abs(actual-expected))),rtol=rt,atol=at,
            reference_sha256=digest(path),threshold_decisions_equal=True)
    dump(folder/'reference_check.json',records)
    return records


def run_nodes(out,book,force=False):
    c=read_json(out/'C_cpu_acceptance.json')
    if c['status']!='PASS' or c['model_sha256']!=C_SHA or c['dev_sha256']!=DEV_SHA:raise ValueError('C CPU gate incomplete')
    bundle=out/'inference_bundle';verify_bundle(bundle)
    from gpu_guard import ProjectGpuLock
    with ProjectGpuLock():
        for name,input_file in [('smoke','smoke_dev.jsonl'),('dev','dev.jsonl')]:
            rows=read_rows(ROOT/'data/processed/v2'/input_file,labeled=True)
            stage='unified_'+name
            old=book.data['stages'].get(stage,{})
            if old.get('status')=='PASS' and not force:
                folder=out/old['directory'];meta=read_json(folder/'prediction_manifest.json')
                if meta['bundle_sha256']!=digest(bundle/'bundle.json'):raise ValueError('Completed bundle identity changed')
                for n,item in meta['files'].items():
                    if digest(folder/n)!=item['sha256']:raise ValueError('Completed predictions changed')
                reference_check(folder,rows);continue
            folder=next_attempt(out/stage)
            book.set(stage,'RUNNING',directory=str(folder.relative_to(out)))
            foreground([sys.executable,'-u',str(bundle/'fusion_inference.py'),'--bundle',str(bundle),
                '--input',str(ROOT/'data/processed/v2'/input_file),'--output',str(folder)],
                folder.parent/(folder.name+'.log'))
            checks=reference_check(folder,rows)
            foreground([sys.executable,'data_tools/evaluate.py','--gold',str(ROOT/'data/processed/v2'/input_file),
                '--pred',str(folder/'predictions.csv'),'--threshold','0.5','--short-by','word','--short-threshold','100',
                '--save-report',str(folder/'a_evaluation_word.txt')],folder/'a_evaluation.log')
            book.set(stage,'PASS',directory=str(folder.relative_to(out)),checks=checks)
    dump(out/'acceptance.json',dict(status='PASS_B_LOCAL_PENDING_A',bundle_sha256=digest(bundle/'bundle.json'),
        C_model_sha256=C_SHA,dev_sha256=DEV_SHA,source_code=git_record(),
        independent_A_acceptance=False,official_data_read=False,training=False))


def analysis_bundle(out):
    folder=next_attempt(out/'analysis');folder.mkdir(parents=True)
    for name in ('stages.json','acceptance.json','failure.json','preflight.json','C_cpu_acceptance.json'):
        if (out/name).exists():shutil.copy2(out/name,folder/name)
    for stage in ('unified_smoke','unified_dev'):
        entry=read_json(out/'stages.json')['stages'].get(stage,{})
        if entry.get('status')=='PASS':
            source=out/entry['directory'];dest=folder/stage;dest.mkdir()
            for name in ('C.csv','B.csv','predictions.csv','prediction_manifest.json','reference_check.json','a_evaluation_word.txt','stages.json'):
                shutil.copy2(source/name,dest/name)
        elif entry.get('directory'):
            source=out/entry['directory'];dest=folder/stage;dest.mkdir()
            for name in ('failure.json','stages.json'):
                if (source/name).is_file():shutil.copy2(source/name,dest/name)
            log=source.parent/(source.name+'.log')
            if log.is_file():(dest/'failure_log_tail.txt').write_text('\n'.join(log.read_text(encoding='utf-8',errors='replace').splitlines()[-60:]),encoding='utf-8')
    dump(folder/'manifest.json',dict(files={p.relative_to(folder).as_posix():dict(bytes=p.stat().st_size,sha256=digest(p))
        for p in folder.rglob('*') if p.is_file()},excluded=['weights','raw text','environment','full logs']))
    with zipfile.ZipFile(folder.with_suffix('.zip'),'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in folder.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(folder))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['dry-run','cpu-check','prepare','run','resume','evaluate','bundle'],default='dry-run')
    p.add_argument('--output',type=Path,default=ROOT/'runs/b_fusion_acceptance_final')
    p.add_argument('--reuse-cpu-from',type=Path,help='Explicit verified previous CPU result; avoids repeating C inference')
    args=p.parse_args();out=args.output.resolve()
    if not out.is_relative_to(ROOT/'runs') or out==ROOT/'runs':raise ValueError('Use a dedicated runs output')
    check=preflight()
    if args.phase=='dry-run':print(json.dumps(check,ensure_ascii=False,indent=2));return
    out.mkdir(parents=True,exist_ok=True);book=StageBook(out);dump(out/'preflight.json',check)
    try:
        if args.phase in ('cpu-check','prepare'):cpu_check(out,book,args.reuse_cpu_from)
        if args.phase=='prepare':prepare(out,book)
        if args.phase in ('run','resume','evaluate'):run_nodes(out,book,force=args.phase=='evaluate')
        book.set('requested_phase','PASS',phase=args.phase)
    except BaseException as exc:
        for name,item in list(book.data['stages'].items()):
            if item['status']=='RUNNING':book.set(name,'FAILED',error=repr(exc))
        book.set('requested_phase','FAILED',phase=args.phase,error=repr(exc))
        dump(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
    finally:analysis_bundle(out)


if __name__=='__main__':main()
