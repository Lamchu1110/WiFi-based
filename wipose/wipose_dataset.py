"""WiPose MAT v5/v7.3 loader, preserving upstream CSI axes and XY scaling."""
from pathlib import Path
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset

MEAN = (15.9144,15.9394,12.1088,27.6384,26.1122,21.0799,14.1105,13.8744,13.8895)
STD = (9.8100,10.2362,8.0946,11.2562,12.9910,10.1495,8.0082,7.4262,9.5949)

def dataset_root(root_dir):
    root = Path(root_dir).expanduser().resolve()
    for candidate in (root, root/'Wi-Pose', root/'WiPose'):
        if (candidate/'Train').is_dir() and (candidate/'Test').is_dir():
            return candidate
    raise FileNotFoundError(f'Expected Train/ and Test/ under {root} (or its Wi-Pose/ child). Extract Wi-Pose.rar first.')


def load_mat(path):
    try:
        return scipy.io.loadmat(path)
    except NotImplementedError:
        try:
            import mat73
        except ImportError as exc:
            raise ImportError('MATLAB 7.3 requires: python -m pip install -r requirements-wipose.txt') from exc
        return mat73.loadmat(str(path))


def pose_array(value, field, vector_layout=None):
    points = np.asarray(value).squeeze()
    if points.shape in ((3,18), (4,18)):
        points = points.T
    elif points.shape in ((18,3), (18,4)):
        pass
    elif points.ndim == 1 and points.size in (54,72):
        columns = points.size//18
        if field == 'SkeletonPoints' or vector_layout == 'coordinate-major':
            # Verified against the actual Wi-Pose.rar distributed for HPE-Li.
            points = points.reshape(columns,18).T
        elif vector_layout == 'joint-major':
            points = points.reshape(18,columns)
        else:
            raise ValueError('Flat jointsVector has ambiguous ordering; specify --joint-vector-layout coordinate-major or joint-major after inspecting the data.')
    else:
        raise ValueError(f'{field}: expected 18 XY/confidence joints, got {points.shape}')
    if points.shape[1] == 4 and not np.allclose(points[:,2],points[:,3],rtol=1e-5,atol=1e-7):
        raise ValueError('Expected duplicated (x,y,c,c) confidence columns; inspect jointsVector schema')
    if not np.isfinite(points).all() or (points[:,2] < 0).any():
        raise ValueError('Pose coordinates/confidence must be finite; confidence must be nonnegative')
    return points[:,:3]


class WiPoseDataset(Dataset):
    def __init__(self, root_dir, split='Train', vector_layout=None):
        self.root = dataset_root(root_dir)
        self.directory = self.root/split
        self.vector_layout = vector_layout
        if not self.directory.is_dir():
            raise FileNotFoundError(self.directory)
        self.file_list = sorted(p.relative_to(self.directory).as_posix() for p in self.directory.rglob('*')
                                if p.is_file() and p.suffix.lower()=='.mat')
        if not self.file_list:
            raise ValueError(f'No MAT files: {self.directory}')
        self.mean = torch.tensor(MEAN).reshape(9,1,1)
        self.std = torch.tensor(STD).reshape(9,1,1)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.directory/self.file_list[idx]
        data = load_mat(path)
        if 'CSI' in data and 'SkeletonPoints' in data:
            csi_field,pose_field = 'CSI','SkeletonPoints'
        elif 'csi_serial' in data and 'jointsVector' in data:
            csi_field,pose_field = 'csi_serial','jointsVector'
        else:
            raise ValueError(f'Unsupported WiPose fields in {path}: {sorted(k for k in data if not k.startswith("__"))}')
        raw = np.asarray(data[csi_field])
        if raw.shape != (5,30,3,3):
            raise ValueError(f'Unexpected CSI shape: {raw.shape}, file={path}')
        if np.iscomplexobj(raw):
            raw = np.abs(raw)
        csi = torch.from_numpy(raw.transpose(3,2,1,0).reshape(9,30,5).astype(np.float32, copy=True))
        try:
            points = torch.from_numpy(pose_array(data[pose_field],pose_field,self.vector_layout).astype(np.float32,copy=True))
        except ValueError as exc:
            raise ValueError(f'{path}: {exc}') from exc
        csi = (csi-self.mean)/self.std
        points = torch.cat((points[:,:2]*0.001, points[:,2:3]), dim=1)
        if not torch.isfinite(csi).all() or not torch.isfinite(points).all():
            raise ValueError(f'Nonfinite native WiPose sample: {path}')
        return {'input_wifi-csi': csi, 'output': points}
