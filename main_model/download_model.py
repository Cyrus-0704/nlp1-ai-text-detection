"""Download the official checkpoint at an immutable Hub revision."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', required=True)
    p.add_argument('--revision', default='main')
    args = p.parse_args()
    repo = 'FacebookAI/xlm-roberta-base'
    info = HfApi().model_info(repo, revision=args.revision)
    root = Path(args.output_dir)
    snapshot_download(repo, revision=info.sha, local_dir=root,
                      allow_patterns=['config.json', 'model.safetensors', 'tokenizer.json',
                                      'tokenizer_config.json', 'sentencepiece.bpe.model',
                                      'special_tokens_map.json', 'README.md'], max_workers=2)
    files = {}
    for path in root.iterdir():
        if path.is_file() and path.name != 'source_manifest.json':
            digest = hashlib.sha256()
            with path.open('rb') as f:
                for chunk in iter(lambda: f.read(8*1024*1024), b''):
                    digest.update(chunk)
            files[path.name] = {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}
    manifest = dict(repo_id=repo, revision=info.sha, source='https://huggingface.co/'+repo,
                    downloaded_at_utc=datetime.now(timezone.utc).isoformat(), files=files)
    (root/'source_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2), flush=True)

if __name__ == '__main__':
    main()
