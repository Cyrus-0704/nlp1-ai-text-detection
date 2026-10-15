"""Infrastructure-only synthetic fixtures. Never a competition benchmark."""
import json
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent/'synthetic'
    root.mkdir(exist_ok=True)
    for split, n in [('train', 1000), ('dev', 200)]:
        path = root/f'{split}.jsonl'
        if path.exists():
            raise FileExistsError(path)
        with path.open('w', encoding='utf-8') as f:
            for i in range(n):
                label = i % 2
                # Intentionally artificial styles: useful only to exercise labels and I/O.
                text = (f'I visited the library today and wrote note {split} {i}. The rain started before lunch.'
                        if label == 0 else
                        f'This synthetic overview {split} {i} presents three structured considerations. First, define the objective. Second, examine the evidence. Finally, summarize the outcome.')
                r = dict(sample_id=f'SYNTHETIC-{split}-{i:04}', group_id=f'SYNTHETIC-{split}-group-{i}',
                         text=text, label=label, generator='synthetic_fixture', domain='SYNTHETIC_NOT_COMPETITION')
                f.write(json.dumps(r)+'\n')
    (root/'NOTICE.txt').write_text('Synthetic infrastructure fixtures only. No competition data or performance claims.\n', encoding='utf-8')

if __name__ == '__main__':
    main()
