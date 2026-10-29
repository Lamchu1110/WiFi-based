"""WiPose training: native XY/confidence, or explicit preprocessed NPZ XYZ windows."""
import argparse
import csv
import json
import random
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from model import HPEWiPoseModel, STHPE
from utils.eval import calulate_error, compute_pck_pckh_18
from utils.losses import KinematicPoseLoss
from test_sanity import Tee


class NpzPoses(Dataset):
    """Explicit train/val/test files; no inference of Z or temporal ordering."""
    def __init__(self, path, full):
        with np.load(path, allow_pickle=False) as data:
            self.x = np.array(data['x'], dtype=np.float32, copy=True)
            self.y = np.array(data['y'], dtype=np.float32, copy=True)
            self.target_format = str(data['target_format'].item())
            unit = str(data['coordinate_unit'].item())
            ordered = bool(data['windows_ordered'].item()) if 'windows_ordered' in data else False
        if self.target_format not in ('xy_conf', 'xyz') or unit != 'm':
            raise ValueError('NPZ needs target_format=xy_conf/xyz and coordinate_unit=m')
        if self.x.ndim not in (4, 5) or self.x.shape[-3:] != (9, 30, 5):
            raise ValueError(f'Invalid WiPose CSI shape: {self.x.shape}')
        if self.y.shape != (*self.x.shape[:-3], 18, 3) or not len(self.x):
            raise ValueError('x/y sample/window shapes must match and be nonempty')
        if not np.isfinite(self.x).all() or not np.isfinite(self.y).all():
            raise ValueError('Nonfinite NPZ data')
        if full and (self.target_format != 'xyz' or self.x.ndim != 5 or self.x.shape[1] < 2 or not ordered):
            raise ValueError('Full requires verified XYZ and ordered within-sequence windows W>=2')
        if not full and self.x.ndim != 4:
            raise ValueError('Baseline/TCN require individual NPZ frames, not windows')
        self.x = torch.from_numpy(self.x)
        self.y = torch.from_numpy(self.y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return {'input_wifi-csi': self.x[index], 'output': self.y[index]}


def datasets(args, full):
    if args.npz_dir:
        sets = [NpzPoses(args.npz_dir / f'{split}.npz', full) for split in ('train', 'val', 'test')]
        if len({dataset.target_format for dataset in sets}) != 1:
            raise ValueError('All splits must declare the same target_format')
        return sets, sets[0].target_format
    if full:
        raise ValueError('Native loader returns XY+confidence, not XYZ. Full needs --npz-dir with verified XYZ windows.')
    if not args.dataset_root:
        raise ValueError('Provide --dataset-root or --npz-dir')
    from sklearn.model_selection import train_test_split
    from wipose import WiPoseDataset
    training = WiPoseDataset(str(args.dataset_root), split='Train')
    heldout = WiPoseDataset(str(args.dataset_root), split='Test')
    # Sorting makes random_state=41 stable across filesystems/machines.
    training.file_list.sort()
    heldout.file_list.sort()
    val, test = train_test_split(np.arange(len(heldout)), test_size=0.5, random_state=41)
    return (training, Subset(heldout, val), Subset(heldout, test)), 'xy_conf'


def evaluate(model, loader, device, xyz):
    model.eval()
    sums = np.zeros(10, dtype=np.float64)
    count = 0
    with torch.no_grad():
        for batch in loader:
            x, gt = batch['input_wifi-csi'].float().to(device), batch['output'].float().numpy()
            pred = model(x)[0].cpu().numpy().reshape(-1,18,3 if xyz else 2)
            gt = gt.reshape(-1,18,3)
            xy, target = pred[...,:2], gt[...,:2]
            errors = calulate_error(xy, target)
            p20 = compute_pck_pckh_18(xy.transpose(0,2,1), target.transpose(0,2,1), 0.2)
            p50 = compute_pck_pckh_18(xy.transpose(0,2,1), target.transpose(0,2,1), 0.5)
            errors3 = calulate_error(pred, gt) if xyz else (0,0)
            values = np.asarray([errors[0]*1000, errors[1]*1000, p20[18], p50[18],
                                 p20[4], p50[4], p20[7], p50[7], errors3[0]*1000, errors3[1]*1000])
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite evaluation metrics')
            sums += values * len(pred)
            count += len(pred)
    if not count:
        raise ValueError('Empty evaluation split')
    keys = ['MPJPE_xy_mm','PA_MPJPE_xy_mm','PCK20','PCK50','R.Wrist_PCK20',
            'R.Wrist_PCK50','L.Wrist_PCK20','L.Wrist_PCK50','MPJPE_xyz_mm','PA_MPJPE_xyz_mm']
    result = dict(zip(keys, (sums/count).tolist()))
    if not xyz:
        result['MPJPE_xyz_mm'] = result['PA_MPJPE_xyz_mm'] = None
    return result


def execute(args, scenario, out):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    full = scenario == 'full'
    (training, validation, testing), target_format = datasets(args, full)
    if len(training) < args.batch_size or not len(validation) or not len(testing):
        raise ValueError('Insufficient train/validation/test samples')
    # Check each split before starting a long job.
    for dataset in (training, validation, testing):
        for i in (0, len(dataset)-1):
            sample = dataset[i]
            x, y = sample['input_wifi-csi'], sample['output']
            if tuple(x.shape[-3:]) != (9,30,5) or tuple(y.shape[-2:]) != (18,3):
                raise ValueError('Invalid native sample shape')
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                raise ValueError('Nonfinite data')
    metadata = {**vars(args), 'scenario': scenario, 'target_format': target_format,
                'train_samples': len(training), 'val_samples': len(validation), 'test_samples': len(testing),
                'loss': 'XYZ kinematic' if full else 'confidence-weighted XY MSE /32' if target_format == 'xy_conf' else 'XY MSE /32',
                'limitations': 'Current WiPose TCN sees T=1; no cross-pose convolution. PCK thresholds are torso-normalized.'}
    (out/'config.json').write_text(json.dumps(metadata, default=str, indent=2), encoding='utf-8')
    print(json.dumps(metadata, default=str), flush=True)
    if args.preflight_only:
        return
    device = torch.device(args.device)
    model = (HPEWiPoseModel() if scenario == 'baseline' else STHPE('wipose', use_tcn=True, direct_3d=full)).to(device)
    criterion = KinematicPoseLoss('wipose', warmup_epochs=args.warmup_epochs).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1-max(0,epoch+1-20)/31)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, drop_last=True, generator=generator)
    val_loader = DataLoader(validation, batch_size=args.batch_size)
    test_loader = DataLoader(testing, batch_size=args.batch_size)
    best = -1
    with (out/'metrics.csv').open('w', newline='', encoding='utf-8') as file:
        writer = None
        for epoch in range(args.epochs):
            model.train()
            total = count = 0
            for step, batch in enumerate(train_loader):
                x, gt = batch['input_wifi-csi'].float().to(device), batch['output'].float().to(device)
                if not torch.isfinite(x).all() or not torch.isfinite(gt).all():
                    raise ValueError('Nonfinite training sample')
                optimizer.zero_grad(set_to_none=True)
                pred = model(x)[0]
                if full:
                    loss = criterion(pred, gt, epoch=epoch)
                else:
                    weight = gt[...,2:3] if target_format == 'xy_conf' else 1.0
                    loss = torch.nn.functional.mse_loss(pred*weight, gt[...,:2]*weight)/32
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite loss')
                loss.backward()
                if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                    raise ValueError('Nonfinite gradient')
                optimizer.step()
                total += loss.item()*len(x)
                count += len(x)
                if step % 100 == 0:
                    print(f'epoch={epoch+1} step={step}/{len(train_loader)} loss={loss.item():.8f}', flush=True)
            result = {'epoch': epoch+1, 'lr': optimizer.param_groups[0]['lr'], 'train_loss': total/count,
                      **evaluate(model, val_loader, device, full)}
            if writer is None:
                writer = csv.DictWriter(file, fieldnames=list(result))
                writer.writeheader()
            writer.writerow(result)
            file.flush()
            print(json.dumps(result), flush=True)
            scheduler.step()
            checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                          'scheduler': scheduler.state_dict(), 'epoch': epoch+1, 'metrics': result}
            torch.save(checkpoint, out/'last.pt')
            if result['PCK20'] > best:
                best = result['PCK20']
                torch.save(checkpoint, out/'best.pt')
    model.load_state_dict(torch.load(out/'best.pt', map_location=device, weights_only=True)['model'])
    result = evaluate(model, test_loader, device, full)
    (out/'test_metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('TEST '+json.dumps(result), flush=True)


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='backslashreplace')
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--dataset-root', type=Path)
    source.add_argument('--npz-dir', type=Path)
    parser.add_argument('--scenario', choices=['baseline','tcn','full'], default='baseline')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--warmup-epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'runs_wipose')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.epochs <= 50 or args.batch_size < 2:
        parser.error('Require 1..50 epochs and batch-size>=2')
    if args.scenario == 'full' and not args.npz_dir:
        parser.error('Full XYZ requires --npz-dir; native third coordinate is confidence')
    stamp = datetime.now(timezone(timedelta(hours=7))).strftime('%Y%m%d_%H%M%S_%f')
    out = args.output/stamp/args.scenario
    out.mkdir(parents=True)
    with (out/'train.log').open('w', encoding='utf-8') as logfile:
        with redirect_stdout(Tee(sys.stdout, logfile)), redirect_stderr(Tee(sys.stderr, logfile)):
            print(f'Output: {out}', flush=True)
            try:
                execute(args, args.scenario, out)
            except Exception:
                traceback.print_exc()
                (out/'status.json').write_text(json.dumps({'status':'failed','error':traceback.format_exc()}), encoding='utf-8')
                return 1
            (out/'status.json').write_text(json.dumps({'status':'preflight_pass' if args.preflight_only else 'complete'}), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
