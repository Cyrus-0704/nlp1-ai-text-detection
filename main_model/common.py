"""Shared local-only model inference and strict data contracts."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

LABEL2ID = {'human': 0, 'machine': 1}
ID2LABEL = {0: 'human', 1: 'machine'}

def load_tokenizer(path):
    from transformers import AutoConfig, AutoTokenizer
    if AutoConfig.from_pretrained(path, local_files_only=True).model_type != 'xlm-roberta':
        raise ValueError('This pipeline requires an XLM-R checkpoint')
    # Transformers 4.57.6 misidentifies locally saved 4.57.6 XLM-R as Mistral.
    # Explicit False preserves the official SentencePiece/Unigram tokenizer unchanged.
    return AutoTokenizer.from_pretrained(path, local_files_only=True, fix_mistral_regex=False)

def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def read_rows(path, labeled=False):
    path = Path(path)
    with path.open(encoding='utf-8-sig', newline='') as f:
        if path.suffix.lower() == '.jsonl':
            rows = [json.loads(line) for line in f if line.strip()]
        elif path.suffix.lower() == '.csv':
            rows = list(csv.DictReader(f))
        else:
            raise ValueError('Use UTF-8 JSONL or CSV')
    if not rows:
        raise ValueError('Empty input')
    seen = set()
    for i, r in enumerate(rows):
        required = ['sample_id', 'text'] + (['group_id', 'label', 'generator', 'domain'] if labeled else [])
        if any(k not in r for k in required):
            raise ValueError(f'Row {i}: missing required fields')
        sid = r['sample_id']
        if not isinstance(sid, (str, int)) or isinstance(sid, bool) or str(sid) == '' or str(sid) in seen:
            raise ValueError(f'Row {i}: empty/invalid/duplicate sample_id')
        seen.add(str(sid))
        if not isinstance(r['text'], str) or not r['text'].strip():
            raise ValueError(f'Row {i}: empty/non-string text')
        if labeled:
            if str(r['label']) not in ('0', '1'):
                raise ValueError('label must be human=0, machine=1')
            r['label'] = int(r['label'])
            if r['group_id'] is None or str(r['group_id']) == '':
                raise ValueError('group_id must be nonempty')
    if labeled and set(r['label'] for r in rows) != {0, 1}:
        raise ValueError('Each internal split must contain both classes')
    return rows

def validate_splits(train, dev):
    for field in ('sample_id', 'group_id', 'text'):
        # Whitespace-normalized text catches accidental duplicate originals; never re-splits data.
        key = (lambda r: ' '.join(r[field].split())) if field == 'text' else (lambda r: str(r[field]))
        if {key(r) for r in train} & {key(r) for r in dev}:
            raise ValueError(f'Train/dev overlap in {field}; ask A to correct the fixed split')

def check_mapping(model):
    if model.config.label2id != LABEL2ID or model.config.id2label != ID2LABEL:
        raise ValueError('Checkpoint label mapping is not human=0, machine=1')

def probabilities(model, tokenizer, rows, max_length, batch_size=1, precision='fp32'):
    check_mapping(model)
    device = next(model.parameters()).device
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = tokenizer([r['text'] for r in rows[start:start+batch_size]],
                              truncation=True, max_length=max_length, padding=True, return_tensors='pt').to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=precision == 'fp16'):
                logits = model(**batch).logits
            output.extend(logits.float().softmax(-1)[:, 1].cpu().tolist())
    validate_probabilities(output, len(rows))
    return output

def validate_probabilities(values, count):
    a = np.asarray(values, dtype=float)
    if a.shape != (count,) or not np.isfinite(a).all() or not ((a >= 0) & (a <= 1)).all():
        raise ValueError('Invalid prediction count/probability')

def write_predictions(path, rows, probs):
    validate_probabilities(probs, len(rows))
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() != '.csv':
        raise ValueError('Prediction output must be .csv')
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_id', 'p_machine'])
        writer.writerows((r['sample_id'], repr(float(p))) for r, p in zip(rows, probs))
    with path.open(encoding='utf-8', newline='') as f:
        actual = list(csv.DictReader(f))
    if [r['sample_id'] for r in actual] != [str(r['sample_id']) for r in rows]:
        raise ValueError('Output sample_id ordering mismatch')
    validate_probabilities([float(r['p_machine']) for r in actual], len(rows))
