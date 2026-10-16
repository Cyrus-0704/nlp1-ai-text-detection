"""Evaluate a saved internal-dev node, then verify a separately loaded export."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

from round_support import ROOT, StageBook, dump, digest, foreground, predictions, read_json, verify_checkpoint


def main(args):
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification
    from common import read_rows, probabilities, write_predictions, load_tokenizer
    from sklearn.metrics import f1_score, log_loss, brier_score_loss
    checkpoint = Path(args.checkpoint).resolve()
    meta = verify_checkpoint(checkpoint)
    if digest(args.dev) != meta['signature']['dev_sha256']:
        raise ValueError('Checkpoint/dev hash mismatch')
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    dump(out/'evaluation_config.json',vars(args))
    book=StageBook(out)
    book.set('before_save_prediction','RUNNING')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    dev = read_rows(args.dev,labeled=True)
    ids = [r['sample_id'] for r in dev]
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,local_files_only=True,attn_implementation='sdpa').to('cuda')
    model.gradient_checkpointing_disable()
    tokenizer = load_tokenizer(checkpoint)
    tick = time.perf_counter()
    values = probabilities(model,tokenizer,dev,meta['signature']['max_length'],args.batch,'fp32')
    before_seconds = time.perf_counter()-tick
    write_predictions(out/'dev_before_save.csv',dev,values)
    book.set('before_save_prediction','PASS',seconds=before_seconds)
    book.set('model_tokenizer_export','RUNNING')
    export = out/'checkpoint'
    model.save_pretrained(export,safe_serialization=True)
    tokenizer.save_pretrained(export)
    for source, target in [('source_manifest.json','source_manifest.json'),
                           ('training_arguments.json','training_arguments.json')]:
        shutil.copy2(checkpoint/source,export/target)
    # Check label mapping in both the in-memory predictor and the independent child.
    del model
    gc.collect()
    torch.cuda.empty_cache()
    book.set('model_tokenizer_export','PASS')
    book.set('independent_reload_prediction','RUNNING')
    reload_seconds = foreground([sys.executable,'-u','main_model/predict.py','--model-dir',str(export),
        '--input',args.dev,'--output',str(out/'dev_predictions.csv'),'--batch-size',str(args.batch),
        '--precision','fp32','--max-length',str(meta['signature']['max_length'])],out/'reload_prediction.log')
    actual = predictions(out/'dev_predictions.csv',ids)
    predictions(out/'dev_before_save.csv',ids)
    np.testing.assert_allclose(actual,values,rtol=1e-5,atol=1e-6)
    book.set('independent_reload_prediction','PASS',seconds=reload_seconds,precision='fp32',batch=args.batch)
    y = np.array([r['label'] for r in dev])
    short = np.array([len(r['text'].split())<100 for r in dev])
    pred = np.array(actual)>=.5
    metric = dict(macro_f1_all=float(f1_score(y,pred,labels=[0,1],average='macro',zero_division=0)),
                  macro_f1_short_temporary=float(f1_score(y[short],pred[short],labels=[0,1],average='macro',zero_division=0)) if short.any() else None,
                  log_loss=float(log_loss(y,actual)),brier=float(brier_score_loss(y,actual)))
    metric['weighted_score_temporary'] = .7*metric['macro_f1_all']+.3*metric['macro_f1_short_temporary'] if short.any() else None
    dump(out/'internal_dev_metrics.json',metric)
    book.set('a_word_evaluation','RUNNING')
    a_seconds = foreground([sys.executable,args.evaluator,'--gold',args.dev,'--pred',str(out/'dev_predictions.csv'),
        '--threshold','0.5','--short-by','word','--short-threshold','100','--save-report',str(out/'a_evaluation_report.txt')],out/'a_evaluation.log')
    book.set('a_word_evaluation','PASS',seconds=a_seconds)
    summary = read_json(args.training_summary)
    summary.update(status='PASS',before_save_inference_seconds=before_seconds,
        independent_reload_and_inference_seconds=reload_seconds,a_evaluation_seconds=a_seconds,
        reload_max_abs_probability_difference=float(np.max(np.abs(np.array(actual)-values))),
        prediction_rows=len(ids),ordered_ids_verified=True,finite_probability_range_verified=True,
        probability_direction='softmax(logits)[:,1]; machine=1',inference_precision='fp32',inference_batch_size=args.batch,
        checkpoint_manifest_sha256=digest(checkpoint/'resume_manifest.json'),
        a_evaluation=dict(script_sha256=digest(args.evaluator),short_by='word',threshold=.5,short_threshold=100),
        evaluated_optimizer_step=meta['optimizer_step'],competition_performance_claim=False)
    dump(out/'run_summary.json',summary)
    shutil.copy2(checkpoint/'training_arguments.json',out/'arguments.json')
    shutil.copy2(checkpoint.parent.parent/'training_log.jsonl',out/'training_log.jsonl')
    dump(out/'prediction_manifest.json',dict(status='PASS',rtol=1e-5,atol=1e-6,
        files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in out.glob('*.csv')},
        export_files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in export.iterdir() if p.is_file()}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--training-summary',required=True)
    p.add_argument('--dev',required=True)
    p.add_argument('--evaluator',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--batch',type=int,default=8)
    args = p.parse_args()
    try:
        from gpu_guard import ProjectGpuLock
        with ProjectGpuLock():
            main(args)
    except BaseException as exc:
        if Path(args.output).is_dir():
            book=StageBook(args.output)
            for name,item in list(book.data['stages'].items()):
                if item['status']=='RUNNING':
                    book.set(name,'FAILED',error=repr(exc))
            dump(Path(args.output)/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
