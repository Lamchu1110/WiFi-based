"""Capture the actual runnable source, including uncommitted files, per suite."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile

from .runtime import atomic_json


def source_inventory(root):
    entries = []
    for directory, dirs, files in os.walk(root,followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in {'.git','__pycache__','data','logs','validation','ablation_archive'}
                         and not d.startswith(('runs','.venv','torchinductor_')) and not (Path(directory)/d).is_symlink())
        for name in sorted(files):
            path = Path(directory)/name
            if path.is_symlink() or name.startswith('.env'):
                continue
            if path.suffix not in {'.py','.yaml','.yml','.toml'} and not (name.startswith('requirements') and path.suffix=='.txt'):
                continue
            content = path.read_bytes()
            entries.append({'path':path.relative_to(root).as_posix(),'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content)})
    entries.sort(key=lambda entry:entry['path'])
    fingerprint = hashlib.sha256(json.dumps(entries,sort_keys=True).encode()).hexdigest()
    return entries,fingerprint


def record_source(root, suite, resume=False):
    root,suite = Path(root).resolve(),Path(suite)
    entries,fingerprint = source_inventory(root)
    if resume:
        saved = json.loads((suite/'source_manifest.json').read_text(encoding='utf-8'))
        if saved['fingerprint'] != fingerprint:
            raise ValueError('Runnable source changed since this suite started. Restore its source_snapshot.zip or start a new suite.')
        with zipfile.ZipFile(suite/'source_snapshot.zip') as archive:
            for item in saved['files']:
                if hashlib.sha256(archive.read(item['path'])).hexdigest()!=item['sha256']:
                    raise ValueError('Source snapshot integrity check failed')
        return fingerprint
    metadata = {'fingerprint':fingerprint,'files':entries,'git_commit':None,'git_branch':None,'git_status':None}
    def git(*args):
        return subprocess.check_output(['git','-C',str(root),*args],text=True,encoding='utf-8',stderr=subprocess.DEVNULL).strip()
    try:
        metadata.update(git_commit=git('rev-parse','HEAD'),git_branch=git('branch','--show-current'),git_status=git('status','--porcelain'))
        (suite/'source.diff').write_text(git('diff','--binary','HEAD','--',*[x['path'] for x in entries]),encoding='utf-8')
    except (OSError,subprocess.CalledProcessError):
        metadata['git_note']='Git unavailable; source snapshot and hashes are authoritative'
    with zipfile.ZipFile(suite/'source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for item in entries:
            content=(root/item['path']).read_bytes()
            if hashlib.sha256(content).hexdigest()!=item['sha256']:
                raise RuntimeError('Source changed while capturing the run')
            archive.writestr(item['path'],content)
    atomic_json(suite/'source_manifest.json',metadata)
    return fingerprint
