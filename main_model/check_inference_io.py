"""Offline format checks for new inputs/predictions; no model or reference data."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON field')
        result[key] = value
    return result


def read_records(path):
    path = Path(path)
    with path.open(encoding='utf-8-sig', newline='') as stream:
        if path.suffix.lower() == '.jsonl':
            for line in stream:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line, object_pairs_hook=unique_object)
                except (ValueError, RecursionError):
                    raise ValueError('Invalid JSON object or duplicate field') from None
                if not isinstance(row, dict):
                    raise ValueError('Each JSONL record must be an object')
                yield row
        elif path.suffix.lower() == '.csv':
            reader = csv.DictReader(stream, strict=True)
            fields = reader.fieldnames
            if not fields or any(not name.strip() for name in fields) or len(set(fields)) != len(fields):
                raise ValueError('Empty or duplicate CSV header')
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError('CSV row width differs from header')
                yield row
        else:
            raise ValueError('Input must be UTF-8 JSONL or CSV')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def check(input_path, predictions_path=None):
    ids, seen = [], set()
    for row in read_records(input_path):
        sid = row.get('sample_id')
        if not isinstance(sid, str) or not sid.strip() or sid in seen:
            raise ValueError('Input sample_id must be a nonempty unique string')
        if not isinstance(row.get('text'), str) or not row['text'].strip():
            raise ValueError('Input text must be a nonempty string')
        ids.append(sid)
        seen.add(sid)
    if not ids:
        raise ValueError('Empty input')
    result = dict(status='PASS_INPUT_FORMAT',rows=len(ids),input_sha256=digest(input_path),
                  scope='Format only; no model quality, provenance or probability-direction verification')
    if predictions_path is not None:
        if Path(predictions_path).suffix.lower() != '.csv':
            raise ValueError('Predictions must be CSV')
        count = 0
        for index, row in enumerate(read_records(predictions_path)):
            if list(row) != ['sample_id', 'p_machine']:
                raise ValueError('Expected exactly sample_id,p_machine columns in that order')
            if index >= len(ids) or row['sample_id'] != ids[index]:
                raise ValueError('Prediction IDs/count/order differ from input')
            try:
                probability = float(row['p_machine'])
            except ValueError:
                raise ValueError('Prediction probability must be numeric') from None
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError('Prediction probability must be finite and in [0,1]')
            count += 1
        if count != len(ids):
            raise ValueError('Prediction row count differs from input')
        result.update(status='PASS_INPUT_OUTPUT_FORMAT',predictions_sha256=digest(predictions_path),
                      ordered_ids=True,finite_probability_range=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--predictions', type=Path, help='Optional CSV to check against input')
    parser.add_argument('--output', required=True, type=Path, help='New JSON report; parent must exist')
    args = parser.parse_args()
    if args.output.exists():
        parser.exit(1, 'FAIL: report already exists; choose a new path\n')
    try:
        result = check(args.input, args.predictions)
        # Exclusive creation also protects a file created by another process during validation.
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    except (ValueError, OSError, csv.Error) as exc:
        parser.exit(1, f'FAIL: {exc}\n')
    print(result['status'])


if __name__ == '__main__':
    main()
