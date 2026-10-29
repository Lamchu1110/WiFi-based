"""Validate this standalone folder on CPU and save fresh evidence."""
import json
import os
import py_compile
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    evidence = root/'validation'
    evidence.mkdir(exist_ok=True)
    for path in root.rglob('*.py'):
        if '.venv' not in path.parts:
            py_compile.compile(str(path), doraise=True)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES':'', 'PYTHONUTF8':'1',
           'OMP_NUM_THREADS':'2', 'MKL_NUM_THREADS':'2', 'PYTHONPATH':str(root)}
    results = {'timestamp':datetime.now(timezone(timedelta(hours=7))).isoformat(timespec='seconds'),
               'python':sys.version, 'device':'cpu', 'syntax':'PASS', 'tests':{}}
    for script in ('test_sanity.py','test_training_smoke.py','test_wipose_smoke.py'):
        with (evidence/(script+'.log')).open('w',encoding='utf-8') as logfile:
            process = subprocess.run([sys.executable,'-X','utf8',str(root/script)], cwd=root,
                                     env=env, stdout=logfile, stderr=subprocess.STDOUT)
        results['tests'][script] = {'exit_code':process.returncode,'result':'PASS' if process.returncode==0 else 'FAIL'}
        print(script, results['tests'][script]['result'], flush=True)
    (evidence/'summary.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    return int(any(item['exit_code'] for item in results['tests'].values()))


if __name__=='__main__':
    raise SystemExit(main())
