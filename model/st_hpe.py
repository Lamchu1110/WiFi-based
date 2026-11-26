"""ST-HPE using the original backbone and decoder, with optional upgrades.

4D input produces a single pose. 5D [B,W,C,F,T] input produces W poses;
W denotes pose frames and T denotes CSI samples inside each frame.
"""
import time
import torch
from torch import nn
from .HPE_no_denoiser import OriginalHPE
from .HPE_Wipose import HPEWiPoseModel
from .tcn_block import MultiScaleDilatedTCN, ShortGatedTCN, ShortResidualTCN
from .utils.SKNet import SKConv


class STHPE(nn.Module):
    architecture_version = 4

    def __init__(self, dataset="mmfi", use_tcn=True, direct_3d=True, temporal_axis="csi",
                 dsk_layout='legacy', pooling_mode='auto', temporal_block='legacy'):
        super().__init__()
        if dataset not in ("mmfi", "wipose"):
            raise ValueError("dataset must be mmfi or wipose")
        self.dataset = dataset
        self.joints = 17 if dataset == "mmfi" else 18
        self.direct_3d = direct_3d
        if temporal_axis not in ("csi", "pose"):
            raise ValueError("temporal_axis must be csi or pose")
        self.temporal_axis = temporal_axis
        self.use_tcn = use_tcn
        if dsk_layout not in ('legacy', 'corrected') or pooling_mode not in ('auto', 'standard', 'preserve_csi'):
            raise ValueError('Invalid DSK layout or pooling mode')
        if temporal_block not in ('legacy', 'short_gated', 'short_residual'):
            raise ValueError('Invalid temporal block')
        if temporal_block in ('short_gated', 'short_residual') and (dataset != 'wipose' or temporal_axis != 'csi'):
            raise ValueError('Short TCN currently supports native WiPose CSI only')
        self.dsk_layout, self.temporal_block = dsk_layout, temporal_block
        self.pooling_mode = pooling_mode
        if pooling_mode == 'auto':
            self.pooling_mode = ('preserve_csi' if dataset == 'wipose' and use_tcn
                                 and temporal_axis == 'csi' and temporal_block == 'legacy' else 'standard')
        if self.pooling_mode == 'preserve_csi' and (dataset != 'wipose' or temporal_axis != 'csi'):
            raise ValueError('preserve_csi pooling is defined for native WiPose only')
        if use_tcn and temporal_block == 'legacy' and dataset == 'wipose' and temporal_axis == 'csi' and self.pooling_mode != 'preserve_csi':
            raise ValueError('Late WiPose CSI TCN needs preserve_csi pooling; standard pooling leaves T=1')
        self.backbone = OriginalHPE() if dataset == "mmfi" else HPEWiPoseModel()
        for module in self.backbone.modules():
            if isinstance(module, SKConv):
                module.layout_mode = dsk_layout
        channels = 128 if dataset == "mmfi" else 256
        if direct_3d:
            old = self.backbone.regression.fc3
            self.backbone.regression.fc3 = nn.Linear(old.in_features, self.joints * 3)
        # Keep backbone/decoder initialization identical across XYZ ablations.
        if not use_tcn:
            self.tcn = nn.Identity()
        elif temporal_block == 'short_gated':
            self.tcn = ShortGatedTCN(64)
        elif temporal_block == 'short_residual':
            self.tcn = ShortResidualTCN(64)
        else:
            self.tcn = MultiScaleDilatedTCN(channels)

    def forward(self, x):
        start = time.perf_counter()
        if x.ndim not in (4, 5):
            raise ValueError("Expected [B,C,F,T] or [B,W,C,F,T]")
        sequence_shape = x.shape[:2] if x.ndim == 5 else None
        expected = (3, 114, 10) if self.dataset == "mmfi" else (9, 30, 5)
        if tuple(x.shape[-3:]) != expected:
            raise ValueError(f"Expected CSI shape {expected}, got {tuple(x.shape[-3:])}")
        if self.use_tcn and self.temporal_axis == "pose" and (sequence_shape is None or sequence_shape[1] < 2):
            raise ValueError("Pose TCN needs ordered [B,W,C,F,T] windows with W>=2")
        if sequence_shape is not None:
            x = x.flatten(0, 1)
        pool = torch.nn.functional.avg_pool2d
        # In WiPose CSI mode keep all five measured time samples until the TCN.
        keep_csi = self.pooling_mode == 'preserve_csi'
        pool_size = (2, 1) if keep_csi else (2, 2)
        x = self.backbone.skunit1(x)
        if self.use_tcn and self.temporal_block in ('short_gated', 'short_residual'):
            x = self.tcn(x)
        x = pool(x, pool_size)
        x = self.backbone.skunit2(x)
        if self.dataset == "wipose":
            x = self.backbone.skunit3(pool(x, pool_size))
        # Convolve measured CSI time or ordered pose time explicitly, then
        # broadcast only the learned correction to preserve decoder geometry.
        if self.use_tcn and self.temporal_axis == "pose":
            batch, width = sequence_shape
            temporal = x.mean(dim=(2, 3)).reshape(batch, width, x.shape[1]).transpose(1, 2)
            correction = (self.tcn(temporal) - temporal).transpose(1, 2).reshape(batch*width, x.shape[1], 1, 1)
            x = x + correction
        elif self.use_tcn and self.temporal_block == 'legacy':
            temporal = x.mean(dim=2)
            x = x + (self.tcn(temporal) - temporal).unsqueeze(2)
        if keep_csi:
            x = torch.nn.functional.adaptive_avg_pool2d(x, (7, 1))
        if self.dataset == "mmfi":
            x = pool(x, (2, 2))
        pose = self.backbone.regression(x).reshape(
            x.shape[0], self.joints, 3 if self.direct_3d else 2)
        if sequence_shape is not None:
            pose = pose.reshape(*sequence_shape, self.joints, pose.shape[-1])
        return pose, time.perf_counter() - start
