"""Bundle inference source/shared dependency specifications and inventory a checkpoint."""
import argparse
import shutil
from pathlib import Path
from common import sha256, write_json

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    args = p.parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    source = Path(__file__).resolve().parent
    required = ['config.json','model.safetensors','tokenizer.json','tokenizer_config.json',
                'source_manifest.json','training_arguments.json']
    if not all((checkpoint/name).is_file() for name in required):
        raise ValueError('Incomplete checkpoint')
    for name in ['predict.py','common.py']:
        if (checkpoint/name).exists():
            raise FileExistsError(checkpoint/name)
        shutil.copy2(source/name, checkpoint/name)
    # Shared packages must not depend on one member's ignored environment folder.
    dependency_root = source.parent/'requirements'
    shutil.copy2(dependency_root/'train.txt', checkpoint/'requirements.txt')
    shutil.copy2(dependency_root/'base.txt', checkpoint/'base.txt')
    (checkpoint/'INFERENCE.md').write_text(
        '# Synthetic smoke checkpoint only\n\nNot a competition-ready detector. human=0, machine=1.\n'
        'Use your own Python environment. Install a suitable torch build, then python -m pip install -r requirements.txt.\n'
        'Run: python predict.py --model-dir . --input INPUT.jsonl --output OUTPUT.csv --precision fp32 --batch-size 1\n'
        'Input: sample_id,text; output: sample_id,p_machine in input order. Local-only loading.\n'
        'Training provenance: training_arguments.json and source_manifest.json; run evidence lives in the parent run directory.\n', encoding='utf-8')
    files = {str(p.relative_to(checkpoint)): dict(bytes=p.stat().st_size, sha256=sha256(p))
             for p in checkpoint.rglob('*') if p.is_file() and p.name != 'package_manifest.json'}
    write_json(checkpoint/'package_manifest.json', dict(data_kind='synthetic', competition_ready=False, files=files))
    print('PASS: standalone inference sources, dependency specifications, model/tokenizer and SHA-256 manifest packaged.')

if __name__ == '__main__':
    main()
