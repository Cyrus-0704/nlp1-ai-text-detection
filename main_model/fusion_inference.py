"""Standalone fixed-probability fusion. Text-only inputs; no training or evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
from common import read_rows, write_predictions, validate_probabilities


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def verify_bundle(bundle):
    bundle=Path(bundle).resolve()
    meta=json.loads((bundle/'bundle.json').read_text(encoding='utf-8'))
    if meta['fusion'] != {'C':.7,'B':.3} or meta['length']!=512 or meta['precision']!='fp32' or meta['batch']!=8:
        raise ValueError('Unexpected fixed inference configuration')
    for name,item in meta['files'].items():
        path=(bundle/name).resolve()
        if not path.is_relative_to(bundle) or not path.is_file() or digest(path)!=item['sha256']:
            raise ValueError(f'Bundle integrity failure: {name}')
    return meta


def c_probabilities(model_path,rows,batch=256):
    import joblib
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    model=joblib.load(model_path)  # Only trusted collaborator artifact, hash checked by caller.
    classes=list(model.named_steps['classifier'].classes_)
    if classes != [0,1]:raise ValueError('Expected human0/machine1 classes')
    vectorizer=model.named_steps['tfidf']
    if vectorizer.analyzer!='char' or vectorizer.ngram_range!=(2,5):
        raise ValueError('Unexpected C text transform')
    output=[]
    for start in range(0,len(rows),batch):
        output.extend(model.predict_proba([r['text'] for r in rows[start:start+batch]])[:,classes.index(1)].tolist())
    validate_probabilities(output,len(rows))
    return np.asarray(output,dtype=np.float64)


def blend(c,b):
    c=np.asarray(c,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    validate_probabilities(c,len(c));validate_probabilities(b,len(c))
    result=.7*c+.3*b
    validate_probabilities(result,len(c))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',required=True,type=Path)
    p.add_argument('--input',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path,help='New output directory')
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    args=p.parse_args()
    meta=verify_bundle(args.bundle)
    rows=read_rows(args.input)
    if not all(isinstance(r['sample_id'],str) for r in rows):raise ValueError('IDs must be strings')
    args.output.mkdir(parents=True,exist_ok=False)
    stages={}
    def stage(name,status,**extra):
        stages[name]=dict(status=status,**extra);save(args.output/'stages.json',stages)
    try:
        stage('C_prediction','RUNNING')
        tick=time.perf_counter()
        c=c_probabilities(args.bundle/'C.joblib',rows)
        write_predictions(args.output/'C.csv',rows,c)
        stage('C_prediction','PASS',seconds=time.perf_counter()-tick)
        stage('B_prediction','RUNNING')
        import torch
        from transformers import AutoModelForSequenceClassification
        from common import probabilities,load_tokenizer
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        model=AutoModelForSequenceClassification.from_pretrained(args.bundle/'B',local_files_only=True,
            attn_implementation='sdpa').to(args.device)
        model.gradient_checkpointing_disable()
        if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
        tick=time.perf_counter()
        b=probabilities(model,load_tokenizer(args.bundle/'B'),rows,512,8,'fp32')
        if args.device=='cuda':torch.cuda.synchronize()
        write_predictions(args.output/'B.csv',rows,b)
        stage('B_prediction','PASS',seconds=time.perf_counter()-tick,
              peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if args.device=='cuda' else None)
        stage('fusion','RUNNING')
        write_predictions(args.output/'predictions.csv',rows,blend(c,b))
        stage('fusion','PASS')
        save(args.output/'prediction_manifest.json',dict(status='PASS',input_sha256=digest(args.input),
            bundle_sha256=digest(args.bundle/'bundle.json'),rows=len(rows),device=args.device,
            precision=meta['precision'],batch=meta['batch'],length=512,probability_direction='machine=1',
            files={n:dict(bytes=(args.output/n).stat().st_size,sha256=digest(args.output/n))
                   for n in ('C.csv','B.csv','predictions.csv')}))
    except BaseException as exc:
        for item in stages.values():
            if item['status']=='RUNNING':item.update(status='FAILED',error=repr(exc))
        save(args.output/'stages.json',stages)
        save(args.output/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    # The caller's project is the lock scope, including independently packaged copies.
    import gpu_guard
    import sys
    gpu_guard.ROOT=Path.cwd().resolve()
    probe=argparse.ArgumentParser(add_help=False)
    probe.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    requested,_=probe.parse_known_args()
    with gpu_guard.ProjectGpuLock(inspect=requested.device=='cuda'):main()
