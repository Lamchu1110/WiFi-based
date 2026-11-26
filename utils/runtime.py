"""Atomic checkpoints, reproducible RNG and process locks (standard library + torch)."""
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch


class Tee:
    def __init__(self,terminal,file):
        self.terminal,self.file = terminal,file

    def write(self,message):
        self.terminal.write(message)
        self.file.write(message)
        self.flush()
        return len(message)

    def flush(self):
        self.terminal.flush()
        self.file.flush()


def replace_with_retry(source, target, attempts=6):
    """Preserve atomic replacement across short Windows sharing/permission locks.

    Never delete the old target or relax permissions. Persistent failures still
    propagate, with the completed temporary file retained for diagnosis.
    """
    for attempt in range(attempts):
        try:
            os.replace(source,target)
            return
        except PermissionError as exc:
            if getattr(exc,'winerror',None) not in (5,32,33) or attempt+1==attempts:
                raise
            time.sleep(.05 * 2**attempt)


def atomic_json(path,data):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w',encoding='utf-8') as file:
        json.dump(data,file,indent=2,default=str,allow_nan=False)
        file.flush()
        os.fsync(file.fileno())
    replace_with_retry(temporary,path)


def atomic_save(path,data):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('wb') as file:
        torch.save(data,file)
        file.flush()
        os.fsync(file.fileno())
    replace_with_retry(temporary,path)


def rng_state():
    state = np.random.get_state()
    return {'python':random.getstate(),'numpy':(state[0],state[1].tolist(),state[2],state[3],state[4]),
            'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available():
        for index,cuda_state in enumerate(state['cuda'][:torch.cuda.device_count()]):
            torch.cuda.set_rng_state(cuda_state.cpu(),index)


class RunLock:
    """OS lock releases on process exit/crash; the small lock file may remain."""
    def __init__(self,directory):
        self.path = Path(directory)/'.run.lock'

    def __enter__(self):
        self.file = self.path.open('a+b')
        if self.file.seek(0,2) == 0:
            self.file.write(b'0')
            self.file.flush()
        self.file.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.file.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as exc:
            self.file.close()
            raise RuntimeError(f'Another training process holds {self.path}') from exc
        return self

    def __exit__(self,*args):
        if os.name == 'nt':
            import msvcrt
            self.file.seek(0)
            msvcrt.locking(self.file.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            fcntl.flock(self.file.fileno(),fcntl.LOCK_UN)
        self.file.close()
