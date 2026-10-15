"""B main-model training. Only A's internal train/dev splits are accepted."""
import argparse
import gc
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed
from common import (LABEL2ID, ID2LABEL, read_rows, validate_splits, write_json,
                    sha256, probabilities, write_predictions, load_tokenizer)
from training_engine import train_updates

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-file', required=True)
    p.add_argument('--dev-file', required=True)
    p.add_argument('--model-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--data-kind', choices=['synthetic', 'a-internal'], required=True,
                   help='Explicit provenance; never pass official validation as dev')
    p.add_argument('--data-manifest', help='A manifest; input hashes must match its declared files')
    p.add_argument('--evaluation-script', help='A internal-dev evaluator; invoked after strict prediction checks')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-steps', type=int, default=50)
    p.add_argument('--max-length', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--inference-batch-size', type=int, default=1,
                   help='Same FP32 batch size before saving and in the independent reload process')
    p.add_argument('--gradient-accumulation', type=int, default=16)
    p.add_argument('--learning-rate', type=float, default=2e-5)
    p.add_argument('--class-weighting', choices=['none','balanced'], default='none')
    p.add_argument('--save-steps', type=int, default=100)
    p.add_argument('--keep-checkpoints', type=int, default=2)
    p.add_argument('--resume-from', help='Trusted local resumable checkpoint; output-dir must be new')
    p.add_argument('--stop-after-steps', type=int, help='Controlled optimizer-boundary interruption for resume verification')
    p.add_argument('--deterministic', action='store_true', help='Require deterministic CUDA operations on this environment')
    args = p.parse_args()
    if args.save_steps < 1 or args.keep_checkpoints < 1 or (args.stop_after_steps is not None and args.stop_after_steps < 1):
        raise ValueError('Invalid checkpoint/stop settings')
    if args.deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True)
    if min(args.max_steps, args.batch_size, args.inference_batch_size, args.gradient_accumulation) < 1 or not 1 <= args.max_length <= 512 or args.learning_rate <= 0:
        raise ValueError('Invalid training hyperparameters')
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'arguments.json', vars(args))
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this validated training configuration')
    torch.set_num_threads(4)
    set_seed(args.seed)
    train = read_rows(args.train_file, labeled=True)
    dev = read_rows(args.dev_file, labeled=True)
    validate_splits(train, dev)
    provenance = None
    if args.data_manifest:
        manifest = json.loads(Path(args.data_manifest).read_text(encoding='utf-8-sig'))
        declared = list(manifest.get('splits', {}).values()) + list(manifest.get('smoke', {}).get('files', {}).values())
        for path in (args.train_file, args.dev_file):
            digest = sha256(path)
            if not any(item['sha256'] == digest and item['bytes'] == Path(path).stat().st_size for item in declared):
                raise ValueError('Input does not match A manifest')
        provenance = dict(version=manifest['version'], manifest_sha256=sha256(args.data_manifest))
        shutil.copy2(args.data_manifest, out/'data_manifest.json')
    repo = Path(__file__).resolve().parent.parent
    code_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    code_dirty = bool(subprocess.check_output(['git', 'status', '--porcelain', '--', 'main_model'], cwd=repo, text=True).strip())
    source = Path(args.model_dir)/'source_manifest.json'
    if not source.exists():
        raise ValueError('Source model requires source_manifest.json from download_model.py')
    shutil.copy2(source, out/'source_manifest.json')
    load_path = args.resume_from or args.model_dir
    tokenizer = load_tokenizer(load_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        load_path, local_files_only=True, num_labels=2, label2id=LABEL2ID,
        id2label=ID2LABEL, attn_implementation='sdpa').to('cuda')
    model.config.detection_max_length = args.max_length
    model.config.detection_inference_precision = 'fp32'
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    training_stats = train_updates(model, tokenizer, train, args, out)
    stats = dict(status='TRAINED_PENDING_RELOAD_CHECK', data_kind=args.data_kind,
                 competition_performance_claim=False, evaluation_scope='internal-development-only',
                 code_commit=code_commit, main_model_code_dirty=code_dirty, data_provenance=provenance,
                 train_rows=len(train), dev_rows=len(dev), **training_stats,
                 fp16=True, gradient_checkpointing=model.is_gradient_checkpointing,
                 dataloader_num_workers=0, inference_precision='fp32', inference_batch_size=args.inference_batch_size,
                 threshold=0.5, short_text_rule='TEMPORARY ONLY: len(text.split()) < 100; not official multilingual word counting',
                 split_overlap_checks=['sample_id', 'group_id', 'whitespace-normalized exact text'],
                 input_files={k: dict(path=str(Path(v).resolve()), sha256=sha256(v))
                              for k, v in [('train', args.train_file), ('dev', args.dev_file)]},
                 python=platform.python_version(), packages={k: importlib.metadata.version(k)
                 for k in ['torch','transformers','accelerate','numpy','scikit-learn']},
                 created_at_utc=datetime.now(timezone.utc).isoformat())
    if not training_stats['completed']:
        stats['status'] = 'STOPPED_AT_CHECKPOINT'
        write_json(out/'run_summary.json', stats)
        print(json.dumps(stats, indent=2), flush=True)
        return
    write_json(out/'run_summary.json', stats)
    # Release optimizer state before FP32 inference. No second GPU model is held concurrently.
    gc.collect()
    torch.cuda.empty_cache()
    model.gradient_checkpointing_disable()
    inference_start = time.perf_counter()
    expected = probabilities(model, tokenizer, dev, args.max_length, args.inference_batch_size, 'fp32')
    stats['before_save_inference_seconds'] = time.perf_counter()-inference_start
    write_predictions(out/'dev_before_save.csv', dev, expected)
    checkpoint = out/'checkpoint'
    model.save_pretrained(checkpoint, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint)
    shutil.copy2(out/'source_manifest.json', checkpoint/'source_manifest.json')
    shutil.copy2(out/'arguments.json', checkpoint/'training_arguments.json')
    if args.data_manifest:
        shutil.copy2(out/'data_manifest.json', checkpoint/'data_manifest.json')
    del model
    gc.collect()
    torch.cuda.empty_cache()
    command = [sys.executable, str(Path(__file__).with_name('predict.py')), '--model-dir', str(checkpoint),
               '--input', str(Path(args.dev_file).resolve()), '--output', str(out/'dev_predictions.csv'),
               '--precision', 'fp32', '--batch-size', str(args.inference_batch_size)]
    reload_start = time.perf_counter()
    with (out/'reload_prediction.log').open('w', encoding='utf-8') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    stats['independent_reload_and_inference_seconds'] = time.perf_counter()-reload_start
    # Evaluate internal dev only, never called from the final inference entry point.
    import csv
    from sklearn.metrics import f1_score
    with (out/'dev_predictions.csv').open(encoding='utf-8', newline='') as f:
        actual_rows = list(csv.DictReader(f))
    if [r['sample_id'] for r in actual_rows] != [str(r['sample_id']) for r in dev]:
        raise AssertionError('Independent process ID mismatch')
    actual = np.array([float(r['p_machine']) for r in actual_rows])
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    y = np.array([r['label'] for r in dev])
    pred = (actual >= 0.5).astype(int)
    mask = np.array([len(r['text'].split()) < 100 for r in dev])
    metrics = dict(evaluator='B auxiliary sklearn calculation; A evaluation reported separately when supplied', data_kind=args.data_kind,
                   competition_performance_claim=False, evaluation_scope='internal-development-only', threshold=0.5,
                   macro_f1_all=float(f1_score(y, pred, labels=[0,1], average='macro', zero_division=0)),
                   short_text_rule=stats['short_text_rule'], official_score=None)
    metrics['macro_f1_short_TEMPORARY'] = float(f1_score(y[mask], pred[mask], labels=[0,1], average='macro', zero_division=0)) if mask.any() else None
    write_json(out/'internal_dev_metrics.json', metrics)
    if args.evaluation_script:
        eval_command = [sys.executable, str(Path(args.evaluation_script).resolve()),
                        '--gold', str(Path(args.dev_file).resolve()), '--pred', str(out/'dev_predictions.csv'),
                        '--threshold', '0.5', '--short-by', 'word', '--short-threshold', '100',
                        '--save-report', str(out/'a_evaluation_report.txt')]
        with (out/'a_evaluation.log').open('w', encoding='utf-8') as log:
            subprocess.run(eval_command, stdout=log, stderr=subprocess.STDOUT, check=True)
        stats['a_evaluation'] = dict(script_sha256=sha256(args.evaluation_script),
                                    report='a_evaluation_report.txt', short_by='word', threshold=0.5,
                                    short_threshold=100, rule_status='temporary; not official')
    stats.update(status='PASS', reload_check=f'independent subprocess; FP32, batch={args.inference_batch_size}, rtol=1e-5, atol=1e-6',
                 reload_max_abs_probability_difference=float(np.max(np.abs(actual-np.array(expected)))),
                 prediction_rows=len(actual), ordered_ids_verified=True, probability_direction='softmax(logits)[:,1]; machine=1',
                 finite_loss_verified=True, finite_probability_range_verified=True)
    write_json(out/'run_summary.json', stats)
    print(json.dumps(stats, indent=2), flush=True)

if __name__ == '__main__':
    main()
