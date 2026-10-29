"""Kinematic objectives. Velocity is defined only over explicit pose sequences."""
import torch
import math
from torch import nn
from torch.nn import functional as F


class KinematicPoseLoss(nn.Module):
    def __init__(self, dataset="mmfi", lambda_bone=0.1, lambda_smooth=0.05,
                 lambda_sym=0.02, warmup_epochs=10, symmetry_mode="target"):
        super().__init__()
        if dataset == "mmfi":
            bones = [(0,1),(1,2),(2,3),(0,4),(4,5),(5,6),(0,7),(7,8),
                     (8,9),(9,10),(8,11),(11,12),(12,13),(8,14),(14,15),(15,16)]
            symmetric = [(0,3),(1,4),(2,5),(10,13),(11,14),(12,15)]
            self.joints = 17
        elif dataset == "wipose":
            bones = [(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),(1,8),(8,9),
                     (9,10),(1,11),(11,12),(12,13),(1,0),(0,14),(14,16),(0,15),(15,17)]
            symmetric = [(0,3),(1,4),(2,5),(6,9),(7,10),(8,11),(13,15),(14,16)]
            self.joints = 18
        else:
            raise ValueError("dataset must be mmfi or wipose")
        if not all(math.isfinite(v) and v >= 0 for v in (warmup_epochs, lambda_bone, lambda_smooth, lambda_sym)):
            raise ValueError("Weights and warmup_epochs must be nonnegative")
        if symmetry_mode not in ("target", "prior"):
            raise ValueError("symmetry_mode must be target or prior")
        self.symmetry_mode = symmetry_mode
        self.register_buffer("bones", torch.tensor(bones, dtype=torch.long))
        self.register_buffer("symmetric", torch.tensor(symmetric, dtype=torch.long))
        self.weights = (lambda_bone, lambda_smooth, lambda_sym)
        self.warmup_epochs = warmup_epochs

    def forward(self, pred, target, epoch=0, return_components=False):
        if pred.shape != target.shape or pred.ndim not in (3, 4):
            raise ValueError("Matching [B,P,3] or [B,W,P,3] poses required")
        if pred.shape[-2:] != (self.joints, 3):
            raise ValueError("Expected dataset joint count and XYZ coordinates")
        if not math.isfinite(epoch) or epoch < 0:
            raise ValueError("epoch must be nonnegative")
        pred, target = pred.float(), target.float()
        if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
            raise ValueError("Pose loss received NaN/Inf")
        def lengths(pose):
            return torch.linalg.vector_norm(
                pose[..., self.bones[:, 0], :] - pose[..., self.bones[:, 1], :], dim=-1)
        predicted, actual = lengths(pred), lengths(target)
        zero = pred.sum() * 0
        components = {
            "L_mse": F.mse_loss(pred, target),
            "L_bone": F.l1_loss(predicted, actual),
            "L_smooth": F.mse_loss(pred[:, 1:] - pred[:, :-1],
                                    target[:, 1:] - target[:, :-1])
            if pred.ndim == 4 and pred.shape[1] > 1 else zero,
            "L_sym": F.l1_loss(
                predicted[..., self.symmetric[:, 0]] - predicted[..., self.symmetric[:, 1]],
                actual[..., self.symmetric[:, 0]] - actual[..., self.symmetric[:, 1]]
                if self.symmetry_mode == "target" else torch.zeros_like(actual[..., self.symmetric[:, 0]])),
        }
        # Epoch 0 is MSE-only; ramp auxiliary terms to full strength by epoch 10.
        scale = min(float(epoch) / self.warmup_epochs, 1.0) if self.warmup_epochs else 1.0
        total = components["L_mse"] + scale * sum(
            weight * components[key] for weight, key in
            zip(self.weights, ("L_bone", "L_smooth", "L_sym")))
        return (total, components) if return_components else total
