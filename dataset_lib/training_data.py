"""Validated, cached datasets and sequence-safe windows for training version 2."""
import hashlib
import io
import json
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io
import torch
import yaml
from torch.utils.data import Dataset, Subset


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Bundle:
    train: Dataset
    val: Dataset
    test: Dataset
    target_format: str
    fingerprint: str
    split: dict
    window: int


class MMFiStore:
    def __init__(self, root=None, archive=None):
        self.root = Path(root) if root else None
        self.zip = zipfile.ZipFile(archive) if archive else None
        self.zip_sequences = defaultdict(list)
        if self.zip:
            for entry in self.zip.infolist():
                parts = entry.filename.split('/')
                if len(parts) >= 5 and parts[-2] == 'wifi-csi' and parts[-1].startswith('frame') and parts[-1].endswith('.mat') and entry.file_size:
                    self.zip_sequences[tuple(parts[-5:-2])].append(entry)

    def sequence(self, scene, subject, action):
        group = f'{scene}/{subject}/{action}'
        if self.zip:
            entries = sorted(self.zip_sequences[(scene,subject,action)],key=lambda e:e.filename)
            if not entries:
                raise FileNotFoundError(f'No CSI in ZIP for {group}')
            prefix = entries[0].filename.rsplit('/wifi-csi/',1)[0]
            labels = np.load(io.BytesIO(self.zip.read(prefix+'/ground_truth.npy')),allow_pickle=False)
            records = [(int(Path(e.filename).stem[5:])-1,e.filename,e.CRC,e.file_size) for e in entries]
        else:
            directory = self.root/group
            labels = np.load(directory/'ground_truth.npy',allow_pickle=False)
            paths = sorted((directory/'wifi-csi').glob('frame*.mat'))
            records = [(int(p.stem[5:])-1,str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in paths if p.stat().st_size]
        labels = np.asarray(labels,dtype=np.float32)
        if labels.ndim != 3 or labels.shape[1:] != (17,3) or not np.isfinite(labels).all():
            raise ValueError(f'Invalid XYZ labels for {group}: {labels.shape}')
        if not records:
            raise ValueError(f'No nonempty CSI frames: {group}')
        indices = [r[0] for r in records]
        if len(set(indices)) != len(indices) or min(indices) < 0 or max(indices) >= len(labels):
            raise ValueError(f'Duplicate or unlabelled CSI indices: {group}')
        fingerprint = digest([group,hashlib.sha256(labels.tobytes()).hexdigest(),[(r[0],r[2],r[3]) for r in records]])
        return group,torch.from_numpy(labels.copy()),records,fingerprint

    def read(self, path):
        data = io.BytesIO(self.zip.read(path)) if self.zip else path
        csi = np.array(scipy.io.loadmat(data)['CSIamp'],dtype=np.float32,copy=True)
        if csi.shape != (3,114,10):
            raise ValueError(f'Invalid CSI shape {csi.shape}: {path}')
        csi[~np.isfinite(csi)] = np.nan
        for time in range(10):
            column = csi[:,:,time]
            finite = column[np.isfinite(column)]
            if not len(finite):
                raise ValueError(f'No finite CSI values: {path}, time={time}')
            column[np.isnan(column)] = finite.mean()
        span = float(csi.max()-csi.min())
        if span <= 0:
            raise ValueError(f'Constant CSI: {path}')
        return torch.from_numpy((csi-csi.min())/span)


class MMFiFrames(Dataset):
    def __init__(self, store, form):
        self.store,self.records,self.labels,self.groups,self.signatures = store,[],{},[],[]
        for subject,actions in sorted(form.items()):
            number = int(subject[1:])
            if not 1 <= number <= 40:
                raise ValueError(f'Invalid subject: {subject}')
            scene = f'E{(number-1)//10+1:02d}'
            for action in sorted(actions):
                group,labels,records,signature = store.sequence(scene,subject,action)
                self.groups.append(group)
                self.labels[group] = labels
                self.signatures.append(signature)
                self.records.extend((group,index,path) for index,path,_,_ in records)
        self.records.sort(key=lambda r:(r[0],r[1]))

    def __len__(self):
        return len(self.records)

    def __getitem__(self,index):
        group,frame,path = self.records[index]
        return {'input_wifi-csi':self.store.read(path),'output':self.labels[group][frame]}


class PoseWindows(Dataset):
    def __init__(self, frames, indices, width, stride):
        self.frames,self.width,self.windows = frames,width,[]
        grouped = defaultdict(list)
        for index in indices:
            grouped[frames.records[index][0]].append(index)
        for group,ids in sorted(grouped.items()):
            ids.sort(key=lambda i:frames.records[i][1])
            for start in range(0,len(ids)-width+1,stride):
                window = ids[start:start+width]
                numbers = [frames.records[i][1] for i in window]
                if numbers == list(range(numbers[0],numbers[0]+width)):
                    self.windows.append(tuple(window))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self,index):
        samples = [self.frames[i] for i in self.windows[index]]
        return {key:torch.stack([s[key] for s in samples]) for key in ('input_wifi-csi','output')}


def mmfi_bundle(args):
    from .mmfi import decode_config
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    if config['data_unit'] != 'frame' or config['modality'] != 'wifi-csi':
        raise ValueError('Expected frame/wifi-csi MM-Fi config')
    if not args.dataset_root and not args.dataset_zip:
        raise ValueError('Provide --dataset-root or --dataset-zip')
    forms = decode_config(config)
    store = MMFiStore(args.dataset_root,args.dataset_zip)
    train = MMFiFrames(store,forms['train_dataset']['data_form'])
    heldout = MMFiFrames(store,forms['val_dataset']['data_form'])
    if set(train.groups) & set(heldout.groups):
        raise ValueError('Train and heldout sequence overlap')
    from sklearn.model_selection import train_test_split
    if args.split_mode == 'sequence':
        if len(heldout.groups) < 2:
            raise ValueError('Sequence split requires at least two heldout actions/subjects; do not silently fall back to frame split')
        val_groups,test_groups = train_test_split(sorted(heldout.groups),test_size=0.5,random_state=41)
        val_set,test_set = set(val_groups),set(test_groups)
        val_ids = [i for i,r in enumerate(heldout.records) if r[0] in val_set]
        test_ids = [i for i,r in enumerate(heldout.records) if r[0] in test_set]
        split = {'mode':'sequence','train':sorted(train.groups),'val':sorted(val_groups),'test':sorted(test_groups)}
    else:
        val_ids,test_ids = train_test_split(np.arange(len(heldout)),test_size=0.5,random_state=41)
        val_ids,test_ids = sorted(map(int,val_ids)),sorted(map(int,test_ids))
        split = {'mode':'legacy_frame','val_indices':val_ids,'test_indices':test_ids}
    width = args.window
    stride = args.window_stride or width
    train_data = PoseWindows(train,range(len(train)),width,stride)
    val_data = PoseWindows(heldout,val_ids,width,stride)
    test_data = PoseWindows(heldout,test_ids,width,stride)
    fingerprint = digest([train.signatures,heldout.signatures,split,width,stride,'mmfi_float32_minmax_v2'])
    return Bundle(train_data,val_data,test_data,'xyz',fingerprint,split,width)


class NpzPoses(Dataset):
    def __init__(self,path,full=False):
        with np.load(path,allow_pickle=False) as data:
            self.x = torch.from_numpy(np.array(data['x'],dtype=np.float32,copy=True))
            self.y = torch.from_numpy(np.array(data['y'],dtype=np.float32,copy=True))
            self.target_format = str(data['target_format'].item())
            unit = str(data['coordinate_unit'].item())
            ordered = data['windows_ordered'] if 'windows_ordered' in data else np.array(False)
            if ordered.shape != () or ordered.dtype.kind != 'b':
                raise ValueError('windows_ordered must be a scalar boolean, not a string')
            self.ordered = bool(ordered.item())
        if self.target_format not in ('xy_conf','xyz') or unit != 'm':
            raise ValueError('Declare target_format=xy_conf/xyz and coordinate_unit=m')
        if self.x.ndim not in (4,5) or tuple(self.x.shape[-3:]) != (9,30,5):
            raise ValueError('Invalid NPZ CSI shape')
        if tuple(self.y.shape) != (*self.x.shape[:-3],18,3) or not len(self.x):
            raise ValueError('NPZ x/y shape mismatch or empty split')
        if not torch.isfinite(self.x).all() or not torch.isfinite(self.y).all():
            raise ValueError('Nonfinite NPZ data')
        if self.target_format == 'xy_conf' and (self.y[...,-1] < 0).any():
            raise ValueError('Confidence must be nonnegative')
        if self.x.ndim == 5 and (self.x.shape[1] < 2 or not self.ordered):
            raise ValueError('Windows must be ordered within-sequence, W>=2')
        if full and (self.target_format != 'xyz' or self.x.ndim != 5):
            raise ValueError('Full requires XYZ windows')

    def __len__(self):
        return len(self.x)

    def __getitem__(self,index):
        return {'input_wifi-csi':self.x[index],'output':self.y[index]}


def wipose_bundle(args,require_xyz=False):
    if args.npz_dir:
        paths = [args.npz_dir/f'{name}.npz' for name in ('train','val','test')]
        sets = [NpzPoses(p,require_xyz) for p in paths]
        if len({(s.target_format,s.x.ndim,tuple(s.x.shape[1:])) for s in sets}) != 1:
            raise ValueError('All NPZ splits must share target format and window shape')
        width = sets[0].x.shape[1] if sets[0].x.ndim == 5 else 1
        split = {'mode':'explicit_npz','files':[p.name for p in paths]}
        return Bundle(*sets,sets[0].target_format,digest([file_digest(p) for p in paths]),split,width)
    if require_xyz:
        raise ValueError('Native WiPose supplies XY+confidence, not XYZ; provide verified XYZ NPZ for 3D scenarios')
    from wipose import WiPoseDataset
    from sklearn.model_selection import train_test_split
    if not args.dataset_root:
        raise ValueError('Provide --dataset-root or --npz-dir')
    training,heldout = WiPoseDataset(args.dataset_root,'Train'),WiPoseDataset(args.dataset_root,'Test')
    val_ids,test_ids = train_test_split(np.arange(len(heldout)),test_size=0.5,random_state=41)
    split = {'mode':'native_file_no_temporal_order','train':training.file_list,
             'val':[heldout.file_list[i] for i in val_ids],'test':[heldout.file_list[i] for i in test_ids]}
    inventory = [(splitname,p.name,p.stat().st_size,p.stat().st_mtime_ns)
                 for splitname in ('Train','Test') for p in sorted((args.dataset_root/splitname).glob('*.mat'))]
    return Bundle(training,Subset(heldout,val_ids),Subset(heldout,test_ids),'xy_conf',digest([inventory,split]),split,1)
