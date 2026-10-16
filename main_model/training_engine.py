"""Optimizer-boundary checkpoints and deterministic sample cursors for single-GPU B runs."""
import hashlib
import json
import random
import shutil
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from common import sha256, write_json


class CursorBatches:
    """Infinite seeded permutations, with the consumed position serialized exactly."""
    def __init__(self, size, batch_size, seed, state=None):
        self.size, self.batch_size, self.seed = size, batch_size, seed
        self.epoch, self.position = (state['epoch'], state['position']) if state else (0, 0)

    def state_dict(self):
        return dict(epoch=self.epoch, position=self.position)

    def __iter__(self):
        while True:
            if self.position == self.size:
                self.epoch += 1
                self.position = 0
            order = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed+self.epoch)).tolist()
            while self.position < self.size:
                end = min(self.position+self.batch_size, self.size)
                batch = order[self.position:end]
                self.position = end
                yield batch


def class_weights(rows, mode):
    counts = np.bincount([r['label'] for r in rows], minlength=2)
    return (len(rows)/(2*counts)).tolist() if mode == 'balanced' else [1., 1.]


def classification_loss(logits, labels, weights):
    # CE(weight=..., reduction='mean') cancels the class weight when batch size is 1.
    # These weights have training-set expectation 1; keep this normalization fixed.
    return (F.cross_entropy(logits.float(), labels, reduction='none') * weights[labels]).mean()


def capture_rng(cuda=True):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if cuda and torch.cuda.is_initialized() else [])


def restore_rng(rng):
    random.setstate(rng['python'])
    np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch'])
    if rng['cuda']:
        torch.cuda.set_rng_state_all(rng['cuda'])


def signature(args, weights):
    result = dict(train_sha256=sha256(args.train_file), dev_sha256=sha256(args.dev_file),
                source_manifest_sha256=sha256(Path(args.model_dir)/'source_manifest.json'),
                seed=args.seed, max_length=args.max_length, batch_size=args.batch_size,
                gradient_accumulation=args.gradient_accumulation, learning_rate=args.learning_rate,
                class_weighting=args.class_weighting, weights=weights, deterministic=args.deterministic,
                optimizer='AdamW;weight_decay=.01;foreach=False;max_grad_norm=1',
                scheduler='constant; no scheduler state', torch_version=torch.__version__)
    if getattr(args, 'strict_resume_code', False):
        result['training_code'] = {name:sha256(Path(__file__).with_name(name))
                                   for name in ('train.py', 'training_engine.py', 'common.py')}
    return result


def save_checkpoint(model, tokenizer, optimizer, scaler, cursor, counters, history, config, args, out):
    root = out/'checkpoints'
    root.mkdir(exist_ok=True)
    dest = root/f"step-{counters['updates']:06d}"
    if dest.exists():
        raise FileExistsError(f'Never overwrite a completed checkpoint: {dest}')
    pending = root/f".step-{counters['updates']:06d}.pending"
    pending.mkdir(exist_ok=False)
    model.save_pretrained(pending, safe_serialization=True)
    tokenizer.save_pretrained(pending)
    for src, dst in [('source_manifest.json','source_manifest.json'), ('arguments.json','training_arguments.json'),
                     ('data_manifest.json','data_manifest.json')]:
        if (out/src).exists():
            shutil.copy2(out/src, pending/dst)
    state = dict(format_version=1, signature=config, optimizer=optimizer.state_dict(),
                 scaler=scaler.state_dict(), cursor=cursor.state_dict(), counters=dict(counters),
                 history=list(history), rng=capture_rng(cuda=next(model.parameters()).device.type=='cuda'))
    torch.save(state, pending/'training_state.pt')
    write_json(pending/'resume_manifest.json', dict(format_version=1, optimizer_step=counters['updates'],
               signature=config, files={p.name:sha256(p) for p in pending.iterdir() if p.is_file()},
               trusted_local_state_only=True, checkpoint_at='after optimizer update and zero_grad'))
    pending.rename(dest)
    older = [] if getattr(args, 'protect_checkpoints', False) else sorted(
        p for p in root.glob('step-*') if p.is_dir() and not (p/'PROTECTED').exists())[:-args.keep_checkpoints]
    for p in older:
        if p.is_symlink() or p.resolve().parent != root.resolve():
            raise RuntimeError('Checkpoint retention path escaped this run')
        shutil.rmtree(p)
    return dest


def load_training_state(path, expected):
    path = Path(path).resolve()
    meta = json.loads((path/'resume_manifest.json').read_text(encoding='utf-8'))
    if meta['signature'] != expected:
        raise ValueError('Resume settings/data mismatch; use original hyperparameters and inputs')
    for name, digest in meta['files'].items():
        if sha256(path/name) != digest:
            raise ValueError(f'Corrupt resume file: {name}')
    # Only our locally generated state with verified hashes is accepted. Never load untrusted pickle.
    return torch.load(path/'training_state.pt', map_location='cpu', weights_only=False)


def train_updates(model, tokenizer, rows, args, out, device='cuda'):
    # device='cpu' is only used by the tiny deterministic regression test.
    is_cuda = device == 'cuda'
    weights_list = class_weights(rows, args.class_weighting)
    config = signature(args, weights_list)
    state = load_training_state(args.resume_from, config) if args.resume_from else None
    cursor = CursorBatches(len(rows), args.batch_size, args.seed, state['cursor'] if state else None)
    def collate(batch_rows):
        batch = tokenizer([r['text'] for r in batch_rows], truncation=True, max_length=args.max_length,
                          padding=True, return_tensors='pt')
        batch['labels'] = torch.tensor([r['label'] for r in batch_rows], dtype=torch.long)
        return batch, [r['sample_id'] for r in batch_rows]
    loader = DataLoader(rows, batch_sampler=cursor, collate_fn=collate, num_workers=0, pin_memory=is_cuda,
                        generator=torch.Generator().manual_seed(args.seed+9000))
    iterator = iter(loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01, foreach=False)
    scaler = torch.amp.GradScaler('cuda', enabled=is_cuda)
    weights = torch.tensor(weights_list, device=device, dtype=torch.float32)
    counters = dict(updates=0, attempts=0, seen=0, gpu_batches=0, total_loss=0., training_seconds=0.,
                    max_observed_length=0, observed_tokens=0, sample_order_hash='')
    history = []
    if state:
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        counters, history = state['counters'], state['history']
        restore_rng(state['rng'])
        del state
    start_step = counters['updates']
    if args.max_steps <= start_step:
        raise ValueError('max_steps must exceed the resumed optimizer step')
    optimizer.zero_grad(set_to_none=True)
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()
    wall_start = time.perf_counter()
    last_checkpoint = None
    with (out/'training_log.jsonl').open('w',encoding='utf-8',buffering=1) as log:
        for entry in history:
            log.write(json.dumps(entry, allow_nan=False)+'\n')
        while counters['updates'] < args.max_steps:
            tick = time.perf_counter()
            model.train()
            losses, unweighted_losses, batch_ids = [], [], []
            for _ in range(args.gradient_accumulation):
                batch, ids = next(iterator)
                batch_ids.extend(ids)
                counters['max_observed_length'] = max(counters['max_observed_length'],batch['input_ids'].shape[1])
                counters['observed_tokens'] += int(batch['attention_mask'].sum())
                counters['seen'] += len(ids)
                counters['gpu_batches'] += 1
                batch = {k:v.to(device,non_blocking=is_cuda) for k,v in batch.items()}
                if next(model.parameters()).device.type != device or batch['input_ids'].device.type != device:
                    raise RuntimeError('GPU placement failed')
                labels = batch.pop('labels')
                with torch.autocast(device,dtype=torch.float16,enabled=is_cuda):
                    logits = model(**batch).logits
                    loss = classification_loss(logits, labels, weights)
                raw_loss = F.cross_entropy(logits.detach().float(), labels)
                value = float(loss.detach())
                if not np.isfinite(value):
                    raise RuntimeError('Nonfinite training loss')
                losses.append(value)
                unweighted_losses.append(float(raw_loss))
                scaler.scale(loss/args.gradient_accumulation).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            skipped = scaler.get_scale() < previous_scale
            counters['attempts'] += 1
            if not skipped:
                counters['updates'] += 1
                counters['total_loss'] += float(np.mean(losses))
            if counters['attempts'] > args.max_steps+50:
                raise RuntimeError('Too many FP16 skipped updates')
            if is_cuda:
                torch.cuda.synchronize()
            counters['training_seconds'] += time.perf_counter()-tick
            counters['peak_allocated_mib'] = max(counters.get('peak_allocated_mib',0.),
                                                torch.cuda.max_memory_allocated()/2**20 if is_cuda else 0.)
            counters['peak_reserved_mib'] = max(counters.get('peak_reserved_mib',0.),
                                               torch.cuda.max_memory_reserved()/2**20 if is_cuda else 0.)
            counters['sample_order_hash'] = hashlib.sha256((counters['sample_order_hash']+json.dumps(batch_ids)).encode()).hexdigest()
            entry = dict(optimizer_step=counters['updates'],attempt=counters['attempts'],skipped=skipped,
                         loss=float(np.mean(losses)),unweighted_loss=float(np.mean(unweighted_losses)),
                         grad_norm=float(norm) if torch.isfinite(norm) else None,scaler_scale=scaler.get_scale(),
                         elapsed_seconds=counters['training_seconds'],sample_order_hash=counters['sample_order_hash'],
                         cursor=cursor.state_dict())
            history.append(entry)
            log.write(json.dumps(entry,allow_nan=False)+'\n')
            progress_every = getattr(args, 'progress_every', 1)
            if not skipped and counters['updates'] % progress_every == 0:
                print(f"update={counters['updates']}/{args.max_steps} attempts={counters['attempts']} "
                      f"skips={counters['attempts']-counters['updates']} loss={entry['loss']:.5f} "
                      f"seen={counters['seen']} active_s={counters['training_seconds']:.1f}", flush=True)
            stop = args.stop_after_steps is not None and counters['updates'] >= args.stop_after_steps
            if not skipped and (counters['updates'] % args.save_steps == 0 or stop or counters['updates'] == args.max_steps):
                last_checkpoint = save_checkpoint(model,tokenizer,optimizer,scaler,cursor,counters,history,config,args,out)
            if stop:
                break
    if is_cuda:
        torch.cuda.synchronize()
    stats = dict(optimizer_updates=counters['updates'],attempted_updates=counters['attempts'],
                 fp16_skipped_updates=counters['attempts']-counters['updates'],
                 training_loss=counters['total_loss']/counters['updates'],training_seconds=counters['training_seconds'],
                 segment_wall_seconds=time.perf_counter()-wall_start,
                 optimizer_updates_per_second=counters['updates']/counters['training_seconds'],
                 samples_per_second=counters['seen']/counters['training_seconds'],processed_samples=counters['seen'],
                 gpu_batches=counters['gpu_batches'],gpu=torch.cuda.get_device_name(0) if is_cuda else 'CPU regression fixture',
                 max_observed_sequence_length=counters['max_observed_length'],processed_nonpadding_tokens=counters['observed_tokens'],
                 peak_allocated_mib=counters.get('peak_allocated_mib',0.),peak_reserved_mib=counters.get('peak_reserved_mib',0.),
                 resumed_from=str(args.resume_from) if args.resume_from else None,resumed_start_step=start_step,
                 class_weighting=args.class_weighting,class_weights=weights_list,loss_reduction='mean(per_example_CE * class_weight)',
                 sampler='seeded-per-epoch cursor',sampler_state=cursor.state_dict(),sample_order_hash=counters['sample_order_hash'],
                 last_resumable_checkpoint=last_checkpoint.relative_to(out).as_posix(),constant_learning_rate=True,
                 completed=counters['updates'] == args.max_steps)
    return stats
