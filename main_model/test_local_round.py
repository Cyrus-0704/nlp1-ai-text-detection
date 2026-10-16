"""Small CPU-only regressions. Never load XLM-R or run CUDA training/inference."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from common import check_mapping
from gpu_guard import ProjectGpuLock
from local_round import load_config, checkpoints
from local_round import train_seed
from round_support import ROOT, dump, predictions
from training_engine import (CursorBatches, capture_rng, restore_rng, signature,
                             load_training_state, train_updates)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Linear(2,6),torch.nn.Dropout(.3),torch.nn.Linear(6,2))

    def forward(self,input_ids,attention_mask):
        return types.SimpleNamespace(logits=self.layers(input_ids.float()))

    def save_pretrained(self,path,**kwargs):
        path = Path(path)
        path.mkdir(exist_ok=True)
        save_file(self.state_dict(),path/'model.safetensors')


class TinyTokenizer:
    def __call__(self,texts,**kwargs):
        return dict(input_ids=torch.tensor([[int(t),int(t)%3] for t in texts]),
                    attention_mask=torch.ones((len(texts),2),dtype=torch.long))

    def save_pretrained(self,path):
        dump(Path(path)/'tokenizer_config.json',{'fixture':'CPU only'})


class LocalRoundTests(unittest.TestCase):
    def test_protocol_and_optional_seed(self):
        c = load_config(ROOT/'main_model/configs/length512_v2.json')
        self.assertEqual(c['effective_seeds'],[42])
        self.assertEqual(load_config(ROOT/'main_model/configs/length512_v2.json',True)['effective_seeds'],[42,43])
        self.assertEqual(load_config(ROOT/'main_model/configs/length512_v2.json',seed43_only=True)['effective_seeds'],[43])
        with self.assertRaises(ValueError):
            load_config(ROOT/'main_model/configs/length512_v2.json',True,True)
        with tempfile.TemporaryDirectory(dir=ROOT/'.local') as temp:
            bad = copy.deepcopy(c)
            bad['dev_file']='data/processed/v2/test.jsonl'
            dump(Path(temp)/'bad.json',bad)
            with self.assertRaises(ValueError):
                load_config(Path(temp)/'bad.json')

    def test_predictions_fail_closed(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.local') as temp:
            p = Path(temp)/'p.csv'
            good = 'sample_id,p_machine\n0001,0.1\n0002,0.9\n'
            p.write_text(good,encoding='utf-8')
            self.assertEqual(predictions(p,['0001','0002']),[.1,.9])
            for text in (good.replace('0.9','nan'),good.replace('0.9','1.1'),
                         good.replace('0002','0001'),good.replace('sample_id,p_machine','sample_id,score'),
                         good.replace('0002,0.9\n','')):
                p.write_text(text,encoding='utf-8')
                with self.assertRaises(ValueError):
                    predictions(p,['0001','0002'])
            p.write_text(good,encoding='utf-8')
            with self.assertRaises(ValueError):
                predictions(p,['0002','0001'])
        with self.assertRaises(ValueError):
            check_mapping(types.SimpleNamespace(config=types.SimpleNamespace(
                label2id={'human':1,'machine':0},id2label={0:'machine',1:'human'})))

    def test_lock_rejects_second_owner_and_allows_child(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.local') as temp:
            lock = Path(temp)/'gpu.lock'
            code = "import sys;sys.path.insert(0,'main_model');from gpu_guard import ProjectGpuLock;\nwith ProjectGpuLock(sys.argv[1],inspect=False): print('OWNED')"
            with ProjectGpuLock(lock,inspect=False):
                allowed = subprocess.run([sys.executable,'-X','utf8','-c',code,str(lock)],capture_output=True,text=True,encoding='utf-8',cwd=ROOT)
                self.assertEqual(allowed.returncode,0,allowed.stderr)
                env = os.environ.copy()
                env.pop('NLP1_GPU_LOCK_OWNER',None)
                denied = subprocess.run([sys.executable,'-X','utf8','-c',code,str(lock)],capture_output=True,text=True,encoding='utf-8',cwd=ROOT,env=env)
                self.assertNotEqual(denied.returncode,0)
                self.assertIn('Another project GPU command owns the lock',denied.stderr)
            with ProjectGpuLock(lock,inspect=False):
                pass

    def test_cpu_full_state_resume_exact_and_protected(self):
        torch.set_num_threads(1)
        rows=[dict(text=str(i),sample_id=f'{i:03d}',label=i%2) for i in range(11)]
        with tempfile.TemporaryDirectory(dir=ROOT/'.local') as temp:
            root=Path(temp)
            source=root/'source'
            source.mkdir()
            dump(source/'source_manifest.json',{'test':'tiny CPU only'})
            data=root/'train.json'
            data.write_text('fixture',encoding='utf-8')
            args=argparse.Namespace(train_file=str(data),dev_file=str(data),model_dir=str(source),
                seed=42,max_length=2,batch_size=1,gradient_accumulation=2,learning_rate=2e-5,
                class_weighting='none',deterministic=True,max_steps=4,stop_after_steps=None,
                save_steps=2,keep_checkpoints=1,protect_checkpoints=True,progress_every=100,
                strict_resume_code=True,resume_from=None)
            def fresh():
                random.seed(42)
                np.random.seed(42)
                torch.manual_seed(42)
                return TinyModel()
            full=fresh()
            (root/'full').mkdir()
            total=train_updates(full,TinyTokenizer(),rows,args,root/'full',device='cpu')
            split=fresh()
            (root/'part').mkdir()
            args.stop_after_steps=2
            train_updates(split,TinyTokenizer(),rows,args,root/'part',device='cpu')
            cp=root/'part/checkpoints/step-000002'
            rng_expected=load_training_state(cp,signature(args,[1.,1.]))['rng']
            # Simulate arbitrarily RNG-consuming evaluation between separate segments.
            torch.rand(17)
            np.random.random(9)
            random.random()
            loaded=TinyModel()
            loaded.load_state_dict(load_file(cp/'model.safetensors'))
            args.resume_from=str(cp)
            args.stop_after_steps=None
            (root/'resume').mkdir()
            resumed=train_updates(loaded,TinyTokenizer(),rows,args,root/'resume',device='cpu')
            for k,v in full.state_dict().items():
                self.assertTrue(torch.equal(v,loaded.state_dict()[k]),k)
            for k in ('optimizer_updates','attempted_updates','processed_samples','sample_order_hash','training_loss'):
                self.assertEqual(total[k],resumed[k],k)
            a=load_training_state(root/'full/checkpoints/step-000004',signature(args,[1.,1.]))
            b=load_training_state(root/'resume/checkpoints/step-000004',signature(args,[1.,1.]))
            self.assertEqual(a['cursor'],b['cursor'])
            self.assertEqual(a['scaler'],b['scaler'])
            for item in a['optimizer']['state']:
                for k,v in a['optimizer']['state'][item].items():
                    if torch.is_tensor(v):
                        self.assertTrue(torch.equal(v,b['optimizer']['state'][item][k]))
            self.assertTrue(torch.equal(a['rng']['torch'],b['rng']['torch']))
            self.assertEqual(a['rng']['python'],b['rng']['python'])
            np.testing.assert_array_equal(a['rng']['numpy'][1],b['rng']['numpy'][1])
            self.assertEqual(a['rng']['cuda'],[])
            self.assertTrue((root/'full/checkpoints/step-000002').exists(),'Protected node was deleted')
            bad=copy.deepcopy(signature(args,[1.,1.]))
            bad['max_length']=3
            with self.assertRaises(ValueError):
                load_training_state(cp,bad)
            (cp/'tokenizer_config.json').write_text('tamper',encoding='utf-8')
            with self.assertRaises(ValueError):
                load_training_state(cp,signature(args,[1.,1.]))

    def test_foreground_schedule_failure_then_resume_no_retraining(self):
        from round_support import StageBook
        with tempfile.TemporaryDirectory(dir=ROOT/'.local') as temp:
            out=Path(temp)
            c=load_config(ROOT/'main_model/configs/length512_v2.json')
            known={}
            trained=[]
            fail=[True]
            def fake_train(cmd,log,echo=False):
                target=int(cmd[cmd.index('--stop-after-steps')+1]) if '--stop-after-steps' in cmd else 2000
                if target==1000 and fail[0]:
                    raise RuntimeError('Injected interruption; no GPU used')
                trained.append(target)
                folder=Path(cmd[cmd.index('--output-dir')+1])
                known[target]=folder/'checkpoints'/f'step-{target:06d}'
            with patch('local_round.checkpoints',return_value=known), \
                 patch('local_round.foreground',side_effect=fake_train), \
                 patch('local_round.verify_checkpoint',return_value={}), \
                 patch('local_round.recover_summary',return_value=dict(optimizer_updates=50,max_observed_sequence_length=512,peak_allocated_mib=1)), \
                 patch('local_round.evaluate') as evaluator, \
                 patch('gpu_guard.require_idle_project',return_value={}):
                with self.assertRaises(RuntimeError):
                    train_seed(c,out,42,StageBook(out))
                self.assertEqual(trained,[50,500])
                self.assertEqual(evaluator.call_count,1)
                fail[0]=False
                train_seed(c,out,42,StageBook(out))
                self.assertEqual(trained,[50,500,1000,1500,2000])
                self.assertEqual([x.args[3] for x in evaluator.call_args_list],[500,500,1000,2000])


if __name__=='__main__':
    unittest.main(verbosity=2)
