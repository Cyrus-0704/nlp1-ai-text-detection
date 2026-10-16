"""Foreground, nonblocking project GPU lock. Never terminate or poll processes."""
import json
import os
from pathlib import Path
import secrets
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def process_snapshot():
    import psutil
    allowed = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
    conflicts, unknown = [], []
    for p in psutil.process_iter(['pid', 'name']):
        if p.pid in allowed:
            continue
        try:
            if 'python' not in (p.info['name'] or '').lower():
                continue
            cmd = p.cmdline()
            # Accommodate relative and absolute entry points and older local runners.
            joined = ' '.join(cmd).replace('\\', '/').lower()
            relevant = any(x in joined for x in ('main_model/train.py', 'main_model/predict.py',
                         'main_model/local_round.py', 'main_model/evaluate_node.py',
                         'fusion_inference.py', 'main_model/fusion_workflow.py',
                         'run_b_v2', 'run_b_round2'))
            if relevant:
                cwd = Path(p.cwd()).resolve()
                if cwd == ROOT or str(ROOT).replace('\\', '/').lower() in joined:
                    conflicts.append(dict(pid=p.pid, name=p.info['name'], entry=cmd[1:3]))
            elif any(Path(s).name.lower() in ('train.py', 'predict.py') for s in cmd):
                if Path(p.cwd()).resolve() == ROOT:
                    conflicts.append(dict(pid=p.pid, name=p.info['name'], entry=cmd[1:3]))
        except psutil.NoSuchProcess:
            continue
        except (psutil.AccessDenied, OSError) as exc:
            unknown.append(dict(pid=p.pid, reason=type(exc).__name__))
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
                            '--format=csv,noheader'], capture_output=True, text=True, timeout=15)
        gpu = dict(returncode=r.returncode, output=r.stdout.strip(), error=r.stderr.strip())
    except (OSError, subprocess.TimeoutExpired) as exc:
        gpu = dict(error=str(exc), returncode=None)
    return dict(conflicts=conflicts, inaccessible_python_processes=unknown, gpu=gpu)


def require_idle_project():
    result = process_snapshot()
    if result['conflicts'] or result['inaccessible_python_processes']:
        raise RuntimeError('Project GPU conflict or Python process inspection unavailable: '+json.dumps(result))
    if result['gpu']['returncode'] != 0:
        raise RuntimeError('GPU inventory unavailable: '+json.dumps(result['gpu']))
    return result


class ProjectGpuLock:
    """OS releases the byte/flock lock on crash; stale metadata is not ownership."""
    def __init__(self, path=None, inspect=True):
        self.path = Path(path) if path else ROOT/'.local/project_gpu.lock'
        self.inspect = inspect
        self.handle = None
        self.previous = None
        self.nested = False

    def __enter__(self):
        import psutil
        inherited = os.environ.get('NLP1_GPU_LOCK_OWNER')
        if inherited and self.path.exists():
            try:
                with self.path.open('rb') as meta:
                    meta.seek(1)
                    owner = json.loads(meta.read().decode('utf-8'))
                parent = next(p for p in psutil.Process().parents() if p.pid == owner['pid'])
                if owner['token'] == inherited and parent.create_time() == owner['created']:
                    self.nested = True
                    return self
            except (ValueError, KeyError, StopIteration, OSError, psutil.Error):
                pass
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open('a+b')
        if self.path.stat().st_size == 0:
            self.handle.write(b' ')
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            self.handle = None
            raise RuntimeError('Another project GPU command owns the lock; no process was stopped')
        try:
            if self.inspect:
                self.snapshot = require_idle_project()
            token = secrets.token_hex(24)
            self.handle.seek(1)
            self.handle.truncate()
            self.handle.write(json.dumps(dict(pid=os.getpid(), created=psutil.Process().create_time(),
                                              token=token)).encode('utf-8'))
            self.handle.flush()
            self.previous = inherited
            os.environ['NLP1_GPU_LOCK_OWNER'] = token
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *exc):
        if self.nested:
            return
        if self.handle:
            self.handle.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None
        if self.previous is None:
            os.environ.pop('NLP1_GPU_LOCK_OWNER', None)
        else:
            os.environ['NLP1_GPU_LOCK_OWNER'] = self.previous
