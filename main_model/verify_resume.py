"""Compare uninterrupted and interrupted/resumed XLM-R GPU training in fresh processes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from common import sha256, write_json

def equal(a,b):
    if isinstance(a,torch.Tensor):
        return torch.equal(a,b)
    if isinstance(a,np.ndarray):
        return np.array_equal(a,b)
    if isinstance(a,dict):
        return a.keys() == b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):
        return len(a) == len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a == b

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train-file',required=True)
    p.add_argument('--dev-file',required=True)
    p.add_argument('--data-manifest',required=True)
    p.add_argument('--model-dir',required=True)
    p.add_argument('--output-dir',required=True)
    a=p.parse_args()
    out=Path(a.output_dir).resolve()
    out.mkdir(parents=True,exist_ok=False)
    common=[sys.executable,str(Path(__file__).with_name('train.py')),
            '--train-file',a.train_file,'--dev-file',a.dev_file,'--model-dir',a.model_dir,
            '--data-manifest',a.data_manifest,'--data-kind','a-internal',
            '--max-steps','4','--gradient-accumulation','2','--save-steps','2',
            '--keep-checkpoints','1','--class-weighting','balanced','--deterministic']
    for name,extra in [('continuous',[]),('interrupted',['--stop-after-steps','2']),
                       ('resumed',['--resume-from',str(out/'interrupted/checkpoints/step-000002')])]:
        with (out/f'{name}.log').open('w',encoding='utf-8') as log:
            subprocess.run(common+['--output-dir',str(out/name)]+extra,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('Completed '+name,flush=True)
    left,right=out/'continuous/checkpoints/step-000004',out/'resumed/checkpoints/step-000004'
    model_equal=sha256(left/'model.safetensors') == sha256(right/'model.safetensors')
    x=torch.load(left/'training_state.pt',map_location='cpu',weights_only=False,mmap=True)
    y=torch.load(right/'training_state.pt',map_location='cpu',weights_only=False,mmap=True)
    checks={k:equal(x[k],y[k]) for k in ['signature','optimizer','scaler','cursor','rng']}
    checks['sample_order_hash']=x['counters']['sample_order_hash'] == y['counters']['sample_order_hash']
    checks['losses']=[r['loss'] for r in x['history']] == [r['loss'] for r in y['history']]
    checks['model_tensors_bit_identical']=model_equal
    checks['predictions_csv_identical']=sha256(out/'continuous/dev_predictions.csv') == sha256(out/'resumed/dev_predictions.csv')
    result=dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,
                scope='Same machine, software, seed, deterministic CUDA; real XLM-R; 4 updates vs 2+2 fresh-process resume',
                model_sha256=sha256(left/'model.safetensors'),internal_test_read=False,official_validation_read=False)
    write_json(out/'resume_verification.json',result)
    print(json.dumps(result,indent=2),flush=True)
    if result['status'] != 'PASS':
        raise AssertionError('Resume verification did not match uninterrupted training')

if __name__ == '__main__':
    main()
