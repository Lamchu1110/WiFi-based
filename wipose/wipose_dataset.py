"""Portable WiPose MAT loader; retain upstream normalization and XY/confidence."""
from pathlib import Path
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset

MEAN = (15.9144,15.9394,12.1088,27.6384,26.1122,21.0799,14.1105,13.8744,13.8895)
STD = (9.8100,10.2362,8.0946,11.2562,12.9910,10.1495,8.0082,7.4262,9.5949)

class WiPoseDataset(Dataset):
    def __init__(self, root_dir, split='Train'):
        self.directory = Path(root_dir)/split
        if not self.directory.is_dir():
            raise FileNotFoundError(self.directory)
        self.file_list = sorted(p.name for p in self.directory.iterdir() if p.is_file() and p.suffix.lower()=='.mat')
        if not self.file_list:
            raise ValueError(f'No MAT files: {self.directory}')
        self.mean = torch.tensor(MEAN).reshape(9,1,1)
        self.std = torch.tensor(STD).reshape(9,1,1)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.directory/self.file_list[idx]
        try:
            data = scipy.io.loadmat(path)
        except NotImplementedError:
            try:
                import mat73
            except ImportError as exc:
                raise ImportError('MATLAB 7.3 requires: pip install -r requirements-wipose.txt') from exc
            data = mat73.loadmat(str(path))
        raw = np.asarray(data['CSI'])
        if raw.ndim != 4 or raw.size != 9*30*5:
            raise ValueError(f'Unexpected CSI shape: {raw.shape}, file={path}')
        csi = torch.from_numpy(raw.transpose(3,2,1,0).reshape(9,30,5).astype(np.float32, copy=True))
        points = torch.from_numpy(np.asarray(data['SkeletonPoints']).reshape(3,18).T.astype(np.float32, copy=True))
        csi = (csi-self.mean)/self.std
        points = torch.cat((points[:,:2]*0.001, points[:,2:3]), dim=1)
        if not torch.isfinite(csi).all() or not torch.isfinite(points).all():
            raise ValueError(f'Nonfinite native WiPose sample: {path}')
        return {'input_wifi-csi': csi, 'output': points}
