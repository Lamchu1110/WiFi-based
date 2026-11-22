"""Disclosed 70/30 WiPose split and lossless, process-safe preprocessing cache."""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, Dataset, Subset

from dataset_lib.training_data import Bundle, digest, file_digest
from utils.runtime import RunLock, atomic_json
from wipose.wipose_dataset import WiPoseDataset, MEAN, STD


class CachedWiPose(Dataset):
    """Read-only mmap, reopened separately by Windows spawn workers."""
    def __init__(self, directory, count):
        self.directory = Path(directory)
        self.count = count
        self._x = self._y = None

    def __len__(self):
        return self.count

    def __getstate__(self):
        return {**self.__dict__, '_x': None, '_y': None}

    def __getitem__(self, index):
        if self._x is None:
            self._x = np.load(self.directory/'csi.npy', mmap_mode='r', allow_pickle=False)
            self._y = np.load(self.directory/'pose.npy', mmap_mode='r', allow_pickle=False)
        # Copies prevent any caller/augmentation from modifying shared cached values.
        return {'input_wifi-csi': torch.from_numpy(self._x[index].copy()),
                'output': torch.from_numpy(self._y[index].copy())}


def cached_dataset(raw, cache_root, inventory_fingerprint):
    directory = Path(cache_root)/inventory_fingerprint
    directory.mkdir(parents=True, exist_ok=True)
    with RunLock(directory):
        metadata = directory/'complete.json'
        if metadata.exists():
            info = json.loads(metadata.read_text(encoding='utf-8'))
            if info['fingerprint'] != inventory_fingerprint or info['count'] != len(raw):
                raise ValueError('Cache manifest mismatch')
            # Hash arrays once per suite, not once per sample or epoch.
            for name, sha in info['sha256'].items():
                if file_digest(directory/name) != sha:
                    raise ValueError(f'Corrupted cache: {directory/name}; use a new --cache-dir')
            print(f'Cache verified: {len(raw)} samples, {directory}', flush=True)
        else:
            print(f'Building cache once from {len(raw)} MAT files: {directory}', flush=True)
            start = time.monotonic()
            x = np.lib.format.open_memmap(directory/'csi.tmp.npy', mode='w+', dtype='float32', shape=(len(raw),9,30,5))
            y = np.lib.format.open_memmap(directory/'pose.tmp.npy', mode='w+', dtype='float32', shape=(len(raw),18,3))
            try:
                for i in range(len(raw)):
                    sample = raw[i]
                    x[i] = sample['input_wifi-csi'].numpy()
                    y[i] = sample['output'].numpy()
                    if (i+1) % 5000 == 0 or i+1 == len(raw):
                        print(f'Cache {i+1}/{len(raw)} ({time.monotonic()-start:.1f}s)', flush=True)
                x.flush(); y.flush()
            finally:
                # Windows disallows renaming an open mmap.
                x._mmap.close(); y._mmap.close()
            os.replace(directory/'csi.tmp.npy', directory/'csi.npy')
            os.replace(directory/'pose.tmp.npy', directory/'pose.npy')
            atomic_json(metadata, {'fingerprint':inventory_fingerprint, 'count':len(raw),
                        'sha256':{name:file_digest(directory/name) for name in ('csi.npy','pose.npy')},
                        'preprocessing':'native loader float32, unchanged axis/normalization/XY*0.001'})
    return CachedWiPose(directory, len(raw))


def paper_bundle(args):
    parts = [WiPoseDataset(args.dataset_root, name, args.joint_vector_layout) for name in ('Train','Test')]
    raw = ConcatDataset(parts)
    inventory = []
    for part in parts:
        for name in part.file_list:
            stat = (part.directory/name).stat()
            inventory.append((part.directory.name+'/'+name, stat.st_size, stat.st_mtime_ns))
    names = [row[0] for row in inventory]
    fingerprint = digest(['paper_wipose_preprocessing_v1',inventory,args.joint_vector_layout,MEAN,STD])
    # Filename ordering and split seed are explicit implementation choices, not author-supplied IDs.
    train_ids, test_ids = train_test_split(np.arange(len(raw)), test_size=0.30,
                                          random_state=args.split_seed, shuffle=True)
    train_ids, test_ids = sorted(train_ids.tolist()), sorted(test_ids.tolist())
    split = {'protocol':'paper_reported_70_30_final_epoch', 'split_seed':args.split_seed,
             'source_inventory':inventory, 'train':[names[i] for i in train_ids],
             'val':[], 'test':[names[i] for i in test_ids],
             'paper_unknowns':['original split seed/file IDs','validation and checkpoint policy'],
             'selection':'fixed final epoch; no test-based selection'}
    if args.mode == 'pilot':
        train_ids = sorted(np.random.default_rng(args.seed).choice(train_ids,min(len(train_ids),args.pilot_train_samples),replace=False).tolist())
        test_ids = sorted(np.random.default_rng(args.seed+1).choice(test_ids,min(len(test_ids),args.pilot_test_samples),replace=False).tolist())
        split['pilot'] = {'train':[names[i] for i in train_ids],'test':[names[i] for i in test_ids]}
    dataset = raw
    cache_train_ids,cache_test_ids = train_ids,test_ids
    if args.cache != 'off':
        if args.mode == 'pilot':
            # A quick pilot must not preprocess the entire 166K-file release.
            chosen = sorted(set(train_ids+test_ids))
            remap = {original:local for local,original in enumerate(chosen)}
            dataset = cached_dataset(Subset(raw,chosen),args.cache_dir,digest([fingerprint,chosen]))
            cache_train_ids = [remap[i] for i in train_ids]
            cache_test_ids = [remap[i] for i in test_ids]
        else:
            dataset = cached_dataset(raw,args.cache_dir,fingerprint)
    return Bundle(Subset(dataset,cache_train_ids),None,Subset(dataset,cache_test_ids),'xy_conf',
                  digest([fingerprint,split]),split,1,'scaled_pixel')
