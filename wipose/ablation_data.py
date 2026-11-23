"""Fixed outer WiPose holdout, inner validation, and lossless caching."""
import json

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, Subset

from dataset_lib.training_data import Bundle, digest
from .paper_data import cached_dataset
from .wipose_dataset import WiPoseDataset, MEAN, STD


def ablation_bundle(args):
    parts = [WiPoseDataset(args.dataset_root, name, args.joint_vector_layout) for name in ('Train','Test')]
    raw = ConcatDataset(parts)
    inventory = []
    for part in parts:
        for name in part.file_list:
            stat = (part.directory/name).stat()
            inventory.append((part.directory.name+'/'+name, stat.st_size, stat.st_mtime_ns))
    names = [row[0] for row in inventory]
    fingerprint = digest(['paper_wipose_preprocessing_v1', inventory, args.joint_vector_layout, MEAN, STD])
    if args.split_file:
        saved = json.loads(args.split_file.read_text(encoding='utf-8-sig'))
        if saved.get('val') or saved.get('pilot'):
            raise ValueError('--split-file must be a full outer train/test split with no validation or pilot subset')
        train_names, test_names = saved['train'], saved['test']
        if (len(set(train_names)) != len(train_names) or len(set(test_names)) != len(test_names)
                or set(train_names) & set(test_names) or set(train_names+test_names) != set(names)):
            raise ValueError('Imported split must contain every dataset ID exactly once')
        mapping = {name:i for i,name in enumerate(names)}
        outer_train = sorted(mapping[name] for name in train_names)
        test_ids = sorted(mapping[name] for name in test_names)
        split_origin = 'imported_file'
    else:
        outer_train, test_ids = train_test_split(np.arange(len(raw)),test_size=.30,
                                                random_state=args.split_seed,shuffle=True)
        outer_train, test_ids = sorted(outer_train.tolist()), sorted(test_ids.tolist())
        split_origin = 'sorted_ids_random_70_30'
    if args.stage == 'develop':
        train_ids, val_ids = train_test_split(outer_train,test_size=args.validation_fraction,
                                            random_state=args.validation_seed,shuffle=True)
        train_ids, val_ids = sorted(train_ids), sorted(val_ids)
    else:
        train_ids, val_ids = outer_train, []
    split = {'protocol':'wipose_ablation_v3', 'stage':args.stage, 'origin':split_origin,
             'split_seed':args.split_seed,'validation_seed':args.validation_seed,
             'outer_train':[names[i] for i in outer_train],
             'train':[names[i] for i in train_ids], 'val':[names[i] for i in val_ids],
             'test':[names[i] for i in test_ids], 'source_inventory':inventory,
             'selection':'inner validation MPJPE' if args.stage=='develop' else 'fixed final epoch',
             'test_evaluated':args.stage=='final',
             'limits':['random frame split; not cross-subject or cross-sequence',
                       'upstream fixed normalization, XY*0.001; confidence is not Z']}
    if args.mode == 'pilot':
        def take(ids, count, seed):
            return sorted(np.random.default_rng(seed).choice(ids,min(len(ids),count),replace=False).tolist()) if ids else []
        # Pilot sample membership is independent of model/shuffle seed.
        train_ids = take(train_ids,args.pilot_train_samples,args.split_seed+100)
        val_ids = take(val_ids,args.pilot_eval_samples,args.validation_seed+100)
        if args.stage == 'final':
            test_ids = take(test_ids,args.pilot_eval_samples,args.split_seed+101)
        split['pilot'] = {'train':[names[i] for i in train_ids], 'val':[names[i] for i in val_ids],
                          'test':[names[i] for i in test_ids] if args.stage=='final' else []}
    if len(train_ids) < args.batch_size or len(train_ids) % args.batch_size == 1:
        raise ValueError('Too few train samples or singleton last batch; adjust pilot size/batch size')
    dataset = raw
    mapped_train, mapped_val, mapped_test = train_ids, val_ids, test_ids
    if args.cache != 'off' and args.mode != 'preflight':
        # Development never reads held-out test tensors, even while making cache.
        chosen = sorted(set(train_ids+val_ids+(test_ids if args.stage=='final' else [])))
        mapping = {original:local for local,original in enumerate(chosen)}
        dataset = cached_dataset(Subset(raw,chosen),args.cache_dir,digest([fingerprint,chosen]))
        mapped_train = [mapping[i] for i in train_ids]
        mapped_val = [mapping[i] for i in val_ids]
        if args.stage == 'final':
            mapped_test = [mapping[i] for i in test_ids]
    return Bundle(Subset(dataset,mapped_train), Subset(dataset,mapped_val) if val_ids else None,
                  Subset(dataset,mapped_test) if args.stage=='final' else Subset(raw,test_ids),
                  'xy_conf',digest([fingerprint,split]),split,1,'scaled_pixel')
