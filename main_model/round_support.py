"""Standard-library IO for the local foreground experiment runner."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(temp, path)


def csv_out(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w = csv.DictWriter(f,fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def git_record():
    def git(*args):
        return subprocess.check_output(['git',*args],cwd=ROOT,text=True,encoding='utf-8').strip()
    return dict(commit=git('rev-parse','HEAD'), branch=git('branch','--show-current'),
                status_porcelain=git('status','--short'),
                files={p.relative_to(ROOT).as_posix():digest(p) for p in sorted((ROOT/'main_model').rglob('*.py'))})


def foreground(command, log_path, echo=False):
    """One owned foreground child; stdout streamed to a file, no detached worker."""
    env = os.environ.copy()
    env.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               CUBLAS_WORKSPACE_CONFIG=':4096:8')
    started = time.perf_counter()
    with Path(log_path).open('x',encoding='utf-8') as log:
        log.write(json.dumps(command,ensure_ascii=False)+'\n')
        log.flush()
        proc = subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                text=True,encoding='utf-8',errors='replace',bufsize=1)
        try:
            for line in proc.stdout:
                log.write(line)
                log.flush()
                if echo and line.startswith('update='):
                    print(line.rstrip(),flush=True)
            code = proc.wait()
        except KeyboardInterrupt:
            # Ctrl+C is delivered to the foreground console group. Retain ownership
            # until the existing child exits; do not terminate an unrelated process.
            print('Interrupted; waiting for the current foreground child to exit.',flush=True)
            proc.wait()
            raise
        finally:
            proc.stdout.close()
    if code:
        raise RuntimeError(f'Child exit={code}; see {log_path}')
    return time.perf_counter()-started


class StageBook:
    def __init__(self, out):
        self.path = Path(out)/'stages.json'
        self.data = read_json(self.path) if self.path.exists() else dict(stages={},manual_visual_review='PENDING_AFTER_RUN')

    def set(self, name, status, **detail):
        previous=dict(self.data['stages'].get(name,{}))
        if 'FAIL' not in status:
            previous.pop('error',None)
        previous.update(status=status,at_utc=datetime.now(timezone.utc).isoformat(),**detail)
        self.data['stages'][name] = previous
        dump(self.path,self.data)
        print(f'{name}: {status}',flush=True)


def predictions(path, ids):
    import math
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['sample_id','p_machine']:
            raise ValueError(f'Wrong prediction columns: {path}')
        rows = list(reader)
    found = [r['sample_id'] for r in rows]
    if found != ids or len(set(found)) != len(found):
        raise ValueError(f'Prediction ID/count/order/uniqueness mismatch: {path}')
    values = [float(r['p_machine']) for r in rows]
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in values):
        raise ValueError(f'Nonfinite/out-of-range probability: {path}')
    return values


def verify_checkpoint(path):
    path = Path(path)
    meta = read_json(path/'resume_manifest.json')
    for name, value in meta['files'].items():
        if digest(path/name) != value:
            raise ValueError(f'Corrupt checkpoint: {path/name}')
    if not (path/'tokenizer_config.json').is_file():
        raise ValueError('Missing checkpoint tokenizer')
    return meta


def next_attempt(root):
    root = Path(root)
    root.mkdir(parents=True,exist_ok=True)
    for i in range(1,10000):
        p = root/f'attempt-{i:03d}'
        if not p.exists():
            return p
    raise RuntimeError('Too many attempts')
