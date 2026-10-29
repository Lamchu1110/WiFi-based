"""CPU smoke tests: generated native MAT and explicit XYZ-window NPZ data."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.io


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
    root = Path(__file__).resolve().parent
    rng = np.random.default_rng(42)
    with tempfile.TemporaryDirectory(prefix='wipose_smoke_') as temporary:
        work = Path(temporary)
        native = work/'native'
        for split in ('Train','Test'):
            folder = native/split
            folder.mkdir(parents=True)
            for i in range(4):
                points = rng.normal(size=(3,18)).astype('float32')
                points[:2] *= 1000
                points[2] = rng.uniform(0.5,1,size=18)
                scipy.io.savemat(folder/f'{i:03d}.mat', {
                    'CSI': rng.normal(15,5,size=(5,30,3,3)).astype('float32'),
                    'SkeletonPoints': points})
        from wipose import WiPoseDataset
        sample = WiPoseDataset(native)[0]
        raw = scipy.io.loadmat(native/'Train/000.mat')
        from wipose.wipose_dataset import MEAN, STD
        import torch
        raw_csi = raw['CSI'].transpose(3,2,1,0).reshape(9,30,5)
        expected = torch.from_numpy(((raw_csi-np.asarray(MEAN,dtype='float32')[:,None,None]) /
                                    np.asarray(STD,dtype='float32')[:,None,None]).copy())
        torch.testing.assert_close(sample['input_wifi-csi'],expected)
        torch.testing.assert_close(sample['output'][:,:2],torch.from_numpy(raw['SkeletonPoints'][:2].T)*0.001)
        print('Native MAT normalization/XY/confidence PASS', flush=True)
        npz = work/'xyz'
        npz.mkdir()
        for split in ('train','val','test'):
            np.savez(npz/f'{split}.npz', x=rng.normal(size=(2,2,9,30,5)).astype('float32'),
                     y=rng.normal(size=(2,2,18,3)).astype('float32'),
                     target_format=np.array('xyz'), coordinate_unit=np.array('m'), windows_ordered=np.array(True))
        for scenario in ('baseline','tcn','full'):
            source = ['--npz-dir',str(npz)] if scenario=='full' else ['--dataset-root',str(native)]
            subprocess.run([sys.executable,str(root/'train_wipose.py'),*source,'--scenario',scenario,
                            '--epochs','1','--batch-size','2','--device','cpu',
                            '--output',str(root/'runs_smoke/wipose')],check=True)
            suite = max((root/'runs_smoke/wipose').iterdir(),key=lambda p:p.name)/scenario
            assert json.loads((suite/'status.json').read_text())['status']=='complete'
            for filename in ('metrics.csv','test_metrics.json','best.pt','last.pt'):
                assert (suite/filename).is_file()
        from train_wipose import NpzPoses
        bad = work/'confidence_windows.npz'
        np.savez(bad, x=np.zeros((2,2,9,30,5),dtype='float32'), y=np.zeros((2,2,18,3),dtype='float32'),
                 target_format=np.array('xy_conf'),coordinate_unit=np.array('m'),windows_ordered=np.array(True))
        try:
            NpzPoses(bad, full=True)
        except ValueError:
            print('Confidence-as-Z rejected PASS',flush=True)
        else:
            raise AssertionError('Full must reject confidence labels')
    print('WiPose training smoke PASS (synthetic data)')


if __name__=='__main__':
    os.environ.setdefault('OMP_NUM_THREADS','2')
    os.environ.setdefault('MKL_NUM_THREADS','2')
    main()
