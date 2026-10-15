"""Verify source/saved tokenizers match on multilingual and truncation fixtures."""
from pathlib import Path
import json
from transformers import AutoTokenizer
from common import load_tokenizer, write_json

def main():
    root = Path(__file__).resolve().parent.parent
    source = root/'models/xlm-roberta-base'
    checkpoint = root/'runs/b_synthetic_smoke_50/checkpoint'
    original = AutoTokenizer.from_pretrained(source, local_files_only=True)
    loaded = load_tokenizer(checkpoint)
    texts = ['Hello world! 123', '这是中文分词核验。', 'Bonjour, été café.',
             'مرحبا بالعالم', '日本語の文章です。', 'word '*400]
    for length in (32, 256):
        assert original(texts, truncation=True, max_length=length) == loaded(texts, truncation=True, max_length=length)
    before = json.loads(original.backend_tokenizer.to_str())
    after = json.loads(loaded.backend_tokenizer.to_str())
    for key in ('model','normalizer','pre_tokenizer','post_processor','decoder'):
        assert before[key] == after[key], key
    report = dict(status='PASS', multilingual_fixtures=len(texts), max_lengths=[32,256],
                  backend_components_identical=True,
                  note='4.57.6 local config detection falsely triggers Mistral warning for saved XLM-R. Explicit fix_mistral_regex=False preserves original backend.')
    write_json(checkpoint.parent/'tokenizer_check.json', report)
    print(json.dumps(report))

if __name__ == '__main__':
    main()
