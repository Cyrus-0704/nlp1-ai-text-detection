"""CPU-only regressions for fixed fusion, portable packages and head-only dropout."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

import numpy as np
import torch
from transformers import XLMRobertaConfig,XLMRobertaForSequenceClassification
from dropout_protocol import apply_classifier_dropout
from fusion_inference import blend,verify_bundle
from fusion_workflow import reference_check
from local_round import load_config,train_command,main
from round_support import ROOT,dump


class FusionDropoutTests(unittest.TestCase):
    def test_tiny_unified_inference_cpu_roundtrip(self):
        # Deliberately tiny synthetic correctness fixture, never a competition score.
        import joblib
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from tokenizers import Tokenizer,models,pre_tokenizers
        from transformers import PreTrainedTokenizerFast
        from fusion_inference import main as predict
        from common import LABEL2ID,ID2LABEL
        with tempfile.TemporaryDirectory(dir=ROOT/'.local') as temp:
            root=Path(temp);bundle=root/'bundle';bundle.mkdir()
            bdir=bundle/'B';bdir.mkdir()
            config=XLMRobertaConfig(vocab_size=8,hidden_size=8,num_hidden_layers=1,num_attention_heads=2,
                intermediate_size=16,num_labels=2,label2id=LABEL2ID,id2label=ID2LABEL)
            XLMRobertaForSequenceClassification(config).save_pretrained(bdir)
            tok=Tokenizer(models.WordLevel({'<unk>':0,'<pad>':1,'human':2,'machine':3,'text':4},unk_token='<unk>'))
            tok.pre_tokenizer=pre_tokenizers.Whitespace()
            PreTrainedTokenizerFast(tokenizer_object=tok,unk_token='<unk>',pad_token='<pad>').save_pretrained(bdir)
            c=Pipeline([('tfidf',TfidfVectorizer(analyzer='char',ngram_range=(2,5),dtype=np.float32)),
                ('classifier',LogisticRegression())]).fit(['human text','human prose','machine text','machine output'],[0,0,1,1])
            joblib.dump(c,bundle/'C.joblib')
            dump(bundle/'bundle.json',dict(fusion={'C':.7,'B':.3},length=512,precision='fp32',batch=8,
                files={p.relative_to(bundle).as_posix():dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                       for p in bundle.rglob('*') if p.is_file()}))
            rows=[dict(sample_id='0001',text='human text'),dict(sample_id='0002',text='machine text')]
            input_path=root/'input.jsonl';input_path.write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf-8')
            argv=['fusion_inference.py','--bundle',str(bundle),'--input',str(input_path),'--output',str(root/'output'),'--device','cpu']
            # Exercise the actual independent CLI and its CPU lock path on a host
            # with no requirement for nvidia-smi/GPU inventory.
            import subprocess
            subprocess.check_call([sys.executable,str(ROOT/'main_model/fusion_inference.py'),*argv[1:]],cwd=root)
            from round_support import predictions,read_json
            cc=predictions(root/'output/C.csv',['0001','0002']);bb=predictions(root/'output/B.csv',['0001','0002'])
            ff=predictions(root/'output/predictions.csv',['0001','0002'])
            np.testing.assert_allclose(ff,.7*np.array(cc)+.3*np.array(bb),rtol=0,atol=0)
            self.assertEqual(read_json(root/'output/prediction_manifest.json')['device'],'cpu')

    def test_head_only_change_and_save_reload(self):
        config=XLMRobertaConfig(vocab_size=32,hidden_size=16,num_hidden_layers=1,num_attention_heads=2,
            intermediate_size=32,num_labels=2,hidden_dropout_prob=.1,attention_probs_dropout_prob=.1,classifier_dropout=.1)
        model=XLMRobertaForSequenceClassification(config)
        weights={n:p.clone() for n,p in model.state_dict().items()}
        with self.assertRaises(ValueError):apply_classifier_dropout(model,.2,resuming=True)
        record=apply_classifier_dropout(model,.2)
        self.assertEqual(model.classifier.dropout.p,.2)
        self.assertTrue(all(m.p==.1 for m in model.roberta.modules() if isinstance(m,torch.nn.Dropout)))
        self.assertTrue(all(torch.equal(p,weights[n]) for n,p in model.state_dict().items()))
        with tempfile.TemporaryDirectory(dir=ROOT/'.local') as temp:
            model.save_pretrained(temp)
            reloaded=XLMRobertaForSequenceClassification.from_pretrained(temp,local_files_only=True)
            self.assertEqual(reloaded.classifier.dropout.p,.2)
            apply_classifier_dropout(reloaded,.2,resuming=True)
        self.assertEqual(record['attention'],.1)

    def test_dropout_fixed_schedule_and_both_seeds(self):
        c=load_config(ROOT/'main_model/configs/dropout02_v2.json')
        old=load_config(ROOT/'main_model/configs/length512_v2.json')
        self.assertEqual(c['training'],old['training'])
        self.assertEqual(c['effective_seeds'],[42,43])
        cmd=train_command(c,Path('unused'),42,50,None)
        self.assertIn('--classifier-dropout',cmd)
        self.assertNotIn('--resume-from',cmd)
        self.assertEqual(cmd[cmd.index('--model-dir')+1],old['model_dir'])
        with self.assertRaises(ValueError):load_config(ROOT/'main_model/configs/dropout02_v2.json',seed43_only=True)

    def test_dropout_training_blocked_without_actual_acceptance(self):
        c=load_config(ROOT/'main_model/configs/dropout02_v2.json')
        # Patch all disk reads only after config parsing: acceptance must be checked
        # before any output directory, training child, or GPU stage is created.
        original=__import__('local_round').read_json
        def read(path):
            if str(path).endswith('dropout02_v2.json'):return c
            return dict(status='FAILED')
        argv=['local_round.py','--config','main_model/configs/dropout02_v2.json','--phase','run','--output','runs/never_created_test']
        with patch('local_round.read_json',side_effect=read),patch.object(sys,'argv',argv),patch('local_round.foreground') as child:
            with self.assertRaises(ValueError):main()
            child.assert_not_called()
        self.assertFalse((ROOT/'runs/never_created_test').exists())

    def test_dropout_waits_for_A_after_local_pass(self):
        c=load_config(ROOT/'main_model/configs/dropout02_v2.json')
        def read(path):
            if str(path).endswith('dropout02_v2.json'):return c
            return dict(status='PASS_B_LOCAL_PENDING_A',bundle_sha256=c['integration_bundle_sha256'])
        def exists(path):return str(path).endswith('acceptance.json')
        argv=['local_round.py','--config','main_model/configs/dropout02_v2.json','--phase','run','--output','runs/never_created_test']
        with patch('local_round.read_json',side_effect=read),patch.object(Path,'is_file',exists),patch.object(sys,'argv',argv),patch('local_round.foreground') as child:
            with self.assertRaisesRegex(ValueError,'A independent'):main()
            child.assert_not_called()

    def test_blend_direction_and_invalid_values(self):
        np.testing.assert_allclose(blend([0,.8],[1,.2]),[.3,.62])
        for a,b in [([np.nan],[.2]),([.1],[2]),([.1,.2],[.3])]:
            with self.assertRaises(ValueError):blend(a,b)

    def test_bundle_tampering_and_path_escape(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.local') as temp:
            folder=Path(temp);file=folder/'asset';file.write_bytes(b'checked')
            meta=dict(fusion={'C':.7,'B':.3},length=512,precision='fp32',batch=8,
                files={'asset':dict(sha256=hashlib.sha256(b'checked').hexdigest())})
            dump(folder/'bundle.json',meta);verify_bundle(folder)
            file.write_bytes(b'tampered')
            with self.assertRaises(ValueError):verify_bundle(folder)
            meta['files']={'../escape':dict(sha256='bad')};dump(folder/'bundle.json',meta)
            with self.assertRaises(ValueError):verify_bundle(folder)


if __name__=='__main__':unittest.main(verbosity=2)
