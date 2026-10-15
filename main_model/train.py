"""B main-model training. Only A's internal train/dev splits are accepted."""
import argparse
import gc
import importlib.metadata
import json
import math
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

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-file', required=True)
    p.add_argument('--dev-file', required=True)
    p.add_argument('--model-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--data-kind', choices=['synthetic', 'a-internal'], required=True,
                   help='Explicit provenance; never pass official validation as dev')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-steps', type=int, default=50)
    p.add_argument('--max-length', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--gradient-accumulation', type=int, default=16)
    p.add_argument('--learning-rate', type=float, default=2e-5)
    args = p.parse_args()
    if min(args.max_steps, args.batch_size, args.gradient_accumulation) < 1 or not 1 <= args.max_length <= 512 or args.learning_rate <= 0:
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
    source = Path(args.model_dir)/'source_manifest.json'
    if not source.exists():
        raise ValueError('Source model requires source_manifest.json from download_model.py')
    shutil.copy2(source, out/'source_manifest.json')
    tokenizer = load_tokenizer(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir, local_files_only=True, num_labels=2, label2id=LABEL2ID,
        id2label=ID2LABEL, attn_implementation='sdpa').to('cuda')
    model.config.detection_max_length = args.max_length
    model.config.detection_inference_precision = 'fp32'
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    def collate(rows):
        inputs = tokenizer([r['text'] for r in rows], truncation=True,
                           max_length=args.max_length, padding=True, return_tensors='pt')
        inputs['labels'] = torch.tensor([r['label'] for r in rows], dtype=torch.long)
        return inputs
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                        num_workers=0, pin_memory=True)
    iterator = iter(loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01, foreach=False)
    scaler = torch.amp.GradScaler('cuda')
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    total_loss, updates, attempts, seen, gpu_batches = 0., 0, 0, 0, 0
    with (out/'training_log.jsonl').open('w', encoding='utf-8', buffering=1) as log:
        while updates < args.max_steps:
            model.train()
            losses = []
            for _ in range(args.gradient_accumulation):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)
                batch = {k: v.to('cuda', non_blocking=True) for k, v in batch.items()}
                if batch['input_ids'].device.type != 'cuda' or next(model.parameters()).device.type != 'cuda':
                    raise RuntimeError('GPU placement failed')
                gpu_batches += 1
                seen += len(batch['labels'])
                with torch.autocast('cuda', dtype=torch.float16):
                    loss = model(**batch).loss
                value = float(loss.detach())
                if not math.isfinite(value):
                    raise RuntimeError('Nonfinite loss')
                losses.append(value)
                scaler.scale(loss / args.gradient_accumulation).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            attempts += 1
            skipped = scaler.get_scale() < scale_before
            if not skipped:
                updates += 1
                total_loss += float(np.mean(losses))
            if attempts > args.max_steps + 30:
                raise RuntimeError('Too many FP16 skipped updates')
            entry = dict(optimizer_step=updates, attempt=attempts, loss=float(np.mean(losses)),
                         skipped=skipped, grad_norm=float(grad_norm) if torch.isfinite(grad_norm) else None,
                         elapsed_seconds=time.perf_counter()-start, scaler_scale=scaler.get_scale())
            log.write(json.dumps(entry, allow_nan=False)+'\n')
            print(json.dumps(entry, allow_nan=False), flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-start
    stats = dict(status='TRAINED_PENDING_RELOAD_CHECK', data_kind=args.data_kind,
                 competition_performance_claim=False if args.data_kind == 'synthetic' else None,
                 train_rows=len(train), dev_rows=len(dev), optimizer_updates=updates,
                 attempted_updates=attempts, fp16_skipped_updates=attempts-updates,
                 training_loss=total_loss/updates, training_seconds=elapsed,
                 optimizer_updates_per_second=updates/elapsed, samples_per_second=seen/elapsed,
                 processed_samples=seen, gpu_batches=gpu_batches, gpu=torch.cuda.get_device_name(0),
                 peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                 peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                 fp16=True, gradient_checkpointing=model.is_gradient_checkpointing,
                 dataloader_num_workers=0, inference_precision='fp32', inference_batch_size=1,
                 threshold=0.5, short_text_rule='TEMPORARY ONLY: len(text.split()) < 100; not official multilingual word counting',
                 split_overlap_checks=['sample_id', 'group_id', 'whitespace-normalized exact text'],
                 input_files={k: dict(path=str(Path(v).resolve()), sha256=sha256(v))
                              for k, v in [('train', args.train_file), ('dev', args.dev_file)]},
                 python=platform.python_version(), packages={k: importlib.metadata.version(k)
                 for k in ['torch','transformers','accelerate','numpy','scikit-learn']},
                 created_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(out/'run_summary.json', stats)
    # Release optimizer state before FP32 inference. No second GPU model is held concurrently.
    del optimizer, scaler, batch, loss
    gc.collect()
    torch.cuda.empty_cache()
    model.gradient_checkpointing_disable()
    expected = probabilities(model, tokenizer, dev, args.max_length, 1, 'fp32')
    write_predictions(out/'dev_before_save.csv', dev, expected)
    checkpoint = out/'checkpoint'
    model.save_pretrained(checkpoint, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint)
    shutil.copy2(out/'source_manifest.json', checkpoint/'source_manifest.json')
    shutil.copy2(out/'arguments.json', checkpoint/'training_arguments.json')
    del model
    gc.collect()
    torch.cuda.empty_cache()
    command = [sys.executable, str(Path(__file__).with_name('predict.py')), '--model-dir', str(checkpoint),
               '--input', str(Path(args.dev_file).resolve()), '--output', str(out/'dev_predictions.csv'),
               '--precision', 'fp32', '--batch-size', '1']
    with (out/'reload_prediction.log').open('w', encoding='utf-8') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
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
    metrics = dict(evaluator='B temporary fallback; A evaluator not delivered', data_kind=args.data_kind,
                   competition_performance_claim=args.data_kind != 'synthetic', threshold=0.5,
                   macro_f1_all=float(f1_score(y, pred, labels=[0,1], average='macro', zero_division=0)),
                   short_text_rule=stats['short_text_rule'], official_score=None)
    metrics['macro_f1_short_TEMPORARY'] = float(f1_score(y[mask], pred[mask], labels=[0,1], average='macro', zero_division=0)) if mask.any() else None
    write_json(out/'internal_dev_metrics.json', metrics)
    stats.update(status='PASS', reload_check='independent subprocess; FP32, batch=1, rtol=1e-5, atol=1e-6',
                 reload_max_abs_probability_difference=float(np.max(np.abs(actual-np.array(expected)))),
                 prediction_rows=len(actual), ordered_ids_verified=True, probability_direction='softmax(logits)[:,1]; machine=1',
                 finite_loss_verified=True, finite_probability_range_verified=True)
    write_json(out/'run_summary.json', stats)
    print(json.dumps(stats, indent=2), flush=True)

if __name__ == '__main__':
    main()
