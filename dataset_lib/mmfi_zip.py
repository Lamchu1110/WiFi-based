"""Read only CSI/XYZ from the original MM-Fi zip without extracting 17.7 GB."""
import io
import zipfile

import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset


class MMFiZipDataset(Dataset):
    def __init__(self, archive, form):
        self.archive = archive
        self.data_list = []
        self.labels = {}
        for item in archive.infolist():
            parts = item.filename.split("/")
            if len(parts) != 6 or parts[-2] != "wifi-csi" or not parts[-1].endswith(".mat") or item.file_size == 0:
                continue
            _, scene, subject, action, _, filename = parts
            if subject not in form or action not in form[subject]:
                continue
            idx = int(filename.removeprefix("frame").removesuffix(".mat")) - 1
            if not 0 <= idx < 297:
                continue
            gt = "/".join(parts[:4]) + "/ground_truth.npy"
            if gt not in self.labels:
                label = np.load(io.BytesIO(archive.read(gt)))
                if label.shape != (297,17,3) or not np.isfinite(label).all():
                    raise ValueError(f"Invalid label: {gt}, {label.shape}")
                self.labels[gt] = torch.from_numpy(label.copy()).float()
            self.data_list.append({"scene": scene, "subject": subject, "action": action,
                                   "idx": idx, "gt_path": gt, "wifi-csi_path": item.filename})
        self.data_list.sort(key=lambda item: (item["gt_path"], item["idx"]))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        item = self.data_list[index]
        csi = scipy.io.loadmat(io.BytesIO(self.archive.read(item["wifi-csi_path"])))['CSIamp']
        if csi.shape != (3,114,10):
            raise ValueError(f"Unexpected CSI shape: {csi.shape}")
        csi = csi.astype(np.float32, copy=True)
        csi[~np.isfinite(csi)] = np.nan
        for i in range(10):
            column = csi[:,:,i]
            valid = column[np.isfinite(column)]
            if not len(valid):
                raise ValueError(f"All-nonfinite CSI slice: {item['wifi-csi_path']}")
            column[np.isnan(column)] = valid.mean()
        span = csi.max() - csi.min()
        if span <= 0:
            raise ValueError(f"Constant CSI: {item['wifi-csi_path']}")
        csi = (csi - csi.min()) / span
        return {**item, "modality": ["wifi-csi"], "input_wifi-csi": csi,
                "output": self.labels[item["gt_path"]][item["idx"]]}


def make_zip_dataset(path, config):
    from .mmfi import decode_config
    archive = zipfile.ZipFile(path)
    forms = decode_config(config)
    return (MMFiZipDataset(archive, forms["train_dataset"]["data_form"]),
            MMFiZipDataset(archive, forms["val_dataset"]["data_form"]))
