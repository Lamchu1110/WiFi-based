"""Measure MAT versus identical mmap reads; this is NOT GPU training throughput."""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from utils.data_loading import make_loader
from utils.runtime import atomic_json
from wipose.paper_data import paper_bundle


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--cache-dir',type=Path,default=Path('data/wipose_cache'))
    p.add_argument('--samples',type=int,default=256)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--joint-vector-layout',choices=['coordinate-major','joint-major'])
    p.add_argument('--output',type=Path,default=Path('logs/wipose_io_benchmark.json'))
    args=p.parse_args()
    if min(args.samples,args.repeats)<1 or args.workers<0: p.error('Invalid sample/repeat/worker count')
    common=dict(dataset_root=args.dataset_root,joint_vector_layout=args.joint_vector_layout,
                mode='pilot',pilot_train_samples=args.samples,pilot_test_samples=32,
                cache_dir=args.cache_dir,seed=0,split_seed=41)
    raw=paper_bundle(SimpleNamespace(**common,cache='off'))
    cached=paper_bundle(SimpleNamespace(**common,cache='mmap'))
    if raw.fingerprint!=cached.fingerprint: raise ValueError('Different data/split')
    for i in range(len(raw.train)):
        for key in raw.train[i]:
            torch.testing.assert_close(raw.train[i][key],cached.train[i][key],rtol=0,atol=0)
    loaders={name:make_loader(dataset,32,torch.device('cpu'),args.workers,
                  generator=torch.Generator().manual_seed(73))
             for name,dataset in [('mat',raw.train),('mmap',cached.train)]}
    for loader in loaders.values():
        for batch in loader: pass  # Warm workers and filesystem cache; exclude startup/build from readings.
    times={name:[] for name in loaders}
    for repeat in range(args.repeats):
        for name in (('mat','mmap') if repeat%2==0 else ('mmap','mat')):
            start=time.perf_counter()
            count=sum(len(batch['output']) for batch in loaders[name])
            elapsed=time.perf_counter()-start
            if count!=len(raw.train): raise ValueError('Incomplete loader pass')
            times[name].append(elapsed)
    medians={name:statistics.median(values) for name,values in times.items()}
    result={'scope':'warm-cache CPU data reading and collation ONLY; no forward/backward, validation or cache build',
            'tensor_equivalence':'exact for every measured sample','samples':len(raw.train),'workers':args.workers,
            'repeats':args.repeats,'seconds':times,'median_seconds':medians,
            'read_only_speedup':medians['mat']/medians['mmap'],'python':platform.python_version(),
            'torch':str(torch.__version__),'platform':platform.platform(),'split_fingerprint':raw.fingerprint}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(args.output,result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
