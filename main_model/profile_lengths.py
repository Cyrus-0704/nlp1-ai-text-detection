"""Read-only token-length diagnostics of A's frozen internal train/dev, never test."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import numpy as np
from common import read_rows, load_tokenizer, sha256, write_json


def summarize(lengths, short):
    values = np.asarray(lengths)
    return dict(rows=len(values), token_quantiles=dict(zip(
        ['min', 'p25', 'median', 'p75', 'p90', 'p95', 'max'],
        np.quantile(values, [0, .25, .5, .75, .9, .95, 1]).tolist())),
        fraction_over_256=float(np.mean(values > 256)),
        fraction_over_512=float(np.mean(values > 512)),
        short_whitespace_words_lt100_temporary=int(np.sum(short)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-manifest', required=True)
    p.add_argument('--model-dir', required=True)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    manifest_path = Path(args.data_manifest)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    tokenizer = load_tokenizer(args.model_dir)
    # Diagnostics count the entire sequence, including special tokens. This
    # suppresses model-limit warnings; it does not change saved tokenizer/model.
    tokenizer.model_max_length = 10**12
    report = dict(data_version=manifest['version'], manifest_sha256=sha256(manifest_path),
        tokenizer_sha256=sha256(Path(args.model_dir)/'tokenizer.json'),
        scope='Frozen internal train/dev only; read-only; no text export',
        tokens_include_special_tokens=True, truncation=False,
        short_rule='TEMPORARY len(text.split()) < 100; not official', splits={})
    started = time.perf_counter()
    for name in ('train', 'dev'):
        expected = manifest['splits'][name]
        path = manifest_path.parent/expected['path']
        if path.stat().st_size != expected['bytes'] or sha256(path) != expected['sha256']:
            raise ValueError(f'{name} differs from A manifest')
        rows = read_rows(path, labeled=True)
        lengths = []
        for start in range(0, len(rows), 64):
            encoded = tokenizer([r['text'] for r in rows[start:start+64]],
                padding=False, truncation=False, return_length=True,
                return_attention_mask=False, return_token_type_ids=False)
            lengths.extend(encoded['length'])
        labels = np.array([r['label'] for r in rows])
        short = np.array([len(r['text'].split()) < 100 for r in rows])
        values = np.asarray(lengths)
        report['splits'][name] = dict(all=summarize(values, short),
            by_label={str(c): summarize(values[labels == c], short[labels == c]) for c in (0, 1)})
        print(json.dumps(dict(split=name, **report['splits'][name]['all'])), flush=True)
        del rows
    report['elapsed_seconds'] = time.perf_counter()-started
    write_json(out/'length_profile.json', report)


if __name__ == '__main__':
    main()
