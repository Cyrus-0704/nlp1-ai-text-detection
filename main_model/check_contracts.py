"""Small failure-path checks for team handoff contracts; no external data."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
from common import (validate_splits, validate_probabilities, check_mapping, read_rows,
                    write_predictions, LABEL2ID, ID2LABEL)

def rejects(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError('Invalid contract was accepted')

def main():
    a = [dict(sample_id='a', group_id='g', text='one')]
    rejects(lambda: validate_splits(a, [dict(sample_id='b', group_id='g', text='two')]))
    rejects(lambda: validate_splits(a, [dict(sample_id='b', group_id='h', text=' one ')]))
    for bad in ([float('nan')], [1.1], [-0.1], []):
        rejects(lambda: validate_probabilities(bad, 1))
    check_mapping(SimpleNamespace(config=SimpleNamespace(label2id=LABEL2ID, id2label=ID2LABEL)))
    rejects(lambda: check_mapping(SimpleNamespace(config=SimpleNamespace(label2id={'human':1,'machine':0}, id2label=ID2LABEL))))
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
        root = Path(tmp)
        (root/'input.csv').write_text('sample_id,text\n0002,hello\n0001,world\n', encoding='utf-8')
        rows = read_rows(root/'input.csv')
        write_predictions(root/'output.csv', rows, [0.2, 0.8])
        assert (root/'output.csv').read_text(encoding='utf-8').splitlines() == ['sample_id,p_machine','0002,0.2','0001,0.8']
        (root/'duplicate.jsonl').write_text('{"sample_id":"x","text":"one"}\n{"sample_id":"x","text":"two"}\n', encoding='utf-8')
        rejects(lambda: read_rows(root/'duplicate.jsonl'))
    print('PASS: group/text leakage, invalid probabilities, reversed mapping, duplicate IDs rejected; CSV leading zeros and row order preserved.')

if __name__ == '__main__':
    main()
