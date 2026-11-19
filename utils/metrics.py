"""Streaming pose metrics with visibility masks and explicit temporal windows."""
import numpy as np


class PoseMetrics:
    def __init__(self, dataset, xyz=False, coordinate_unit='m'):
        if coordinate_unit not in ('m','scaled_pixel'):
            raise ValueError('Metrics expect metres or pixels scaled by 0.001')
        self.coordinate_unit = coordinate_unit
        self.xyz = xyz
        self.joints = 17 if dataset == "mmfi" else 18
        self.torso = (1, 11) if dataset == "mmfi" else (5, 8)
        self.hands = (13, 16) if dataset == "mmfi" else (4, 7)
        self.labels = ("R.Hand", "L.Hand") if dataset == "mmfi" else ("R.Wrist", "L.Wrist")
        self.sums, self.counts = {}, {}
        self.hits = {t: np.zeros(self.joints) for t in (0.2, 0.5)}
        self.pck_counts = np.zeros(self.joints)
        self.invalid_torso_frames = 0
        self.degenerate_gt_frames = 0

    def add(self, name, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.size:
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite metric: {name}")
            self.sums[name] = self.sums.get(name, 0.0) + values.sum()
            self.counts[name] = self.counts.get(name, 0) + values.size

    @staticmethod
    def aligned_error(pred, target, mask):
        p, g = pred[mask], target[mask]
        if len(p) < 2:
            return None
        p0, g0 = p-p.mean(0), g-g.mean(0)
        energy = np.square(p0).sum()
        if np.square(g0).sum() <= 1e-12:
            return None
        if energy <= 1e-12:
            aligned = np.broadcast_to(g.mean(0), g.shape)
        else:
            u, s, vt = np.linalg.svd(p0.T @ g0, full_matrices=False)
            signs = np.ones(len(s))
            if np.linalg.det(u @ vt) < 0:
                signs[-1] = -1
            rotation = (u * signs) @ vt
            scale = (s*signs).sum()/energy
            aligned = scale*p0 @ rotation + g.mean(0)
        return np.linalg.norm(aligned-g, axis=-1)

    def update(self, pred, target, confidence=None):
        pred, target = np.asarray(pred, dtype=np.float64), np.asarray(target, dtype=np.float64)
        if pred.ndim not in (3,4) or pred.shape[:-1] != target.shape[:-1] or pred.shape[-2] != self.joints:
            raise ValueError("Matching frame/window pose shapes required for metrics")
        if not np.isfinite(pred).all() or not np.isfinite(target).all():
            raise ValueError("Nonfinite metric inputs")
        mask = np.ones(target.shape[:-1], dtype=bool) if confidence is None else np.asarray(confidence) > 0
        if mask.shape != target.shape[:-1]:
            raise ValueError("Visibility shape mismatch")
        for dimensions, suffix in ((2, "xy"), (3, "xyz")):
            if dimensions > pred.shape[-1]:
                continue
            p, g = pred[..., :dimensions], target[..., :dimensions]
            self.add(f"MPJPE_{suffix}_mm", np.linalg.norm(p-g,axis=-1)[mask]*1000)
            flat_p, flat_g, flat_m = p.reshape(-1,self.joints,dimensions), g.reshape(-1,self.joints,dimensions), mask.reshape(-1,self.joints)
            for a,b,m in zip(flat_p,flat_g,flat_m):
                error = self.aligned_error(a,b,m)
                if error is None:
                    if dimensions == 2:
                        self.degenerate_gt_frames += 1
                else:
                    self.add(f"PA_MPJPE_{suffix}_mm",error*1000)
            if pred.ndim == 4 and pred.shape[1] > 1:
                vmask = mask[:,1:] & mask[:,:-1]
                delta = np.diff(p,axis=1)-np.diff(g,axis=1)
                self.add(f"velocity_{suffix}_mm_per_frame",np.linalg.norm(delta,axis=-1)[vmask]*1000)
                if pred.shape[1] > 2:
                    amask = mask[:,2:] & mask[:,1:-1] & mask[:,:-2]
                    acceleration = np.diff(p,n=2,axis=1)-np.diff(g,n=2,axis=1)
                    self.add(f"acceleration_{suffix}_mm_per_frame2",np.linalg.norm(acceleration,axis=-1)[amask]*1000)
        xy, gt = pred[...,:2], target[...,:2]
        a,b = self.torso
        torso = np.linalg.norm(gt[...,a,:]-gt[...,b,:],axis=-1)
        usable = (torso > 1e-8) & mask[...,a] & mask[...,b]
        self.invalid_torso_frames += int((~usable).sum())
        valid = mask & usable[...,None]
        distance = np.linalg.norm(xy-gt,axis=-1)/np.maximum(torso[...,None],1e-8)
        axes = tuple(range(valid.ndim-1))
        self.pck_counts += valid.sum(axis=axes)
        for threshold in self.hits:
            self.hits[threshold] += ((distance <= threshold) & valid).sum(axis=axes)

    def result(self):
        if not self.counts.get("MPJPE_xy_mm"):
            raise ValueError("No visible joints to evaluate")
        result = {}
        for suffix in ("xy","xyz"):
            for template in ("MPJPE_{}_mm","PA_MPJPE_{}_mm","velocity_{}_mm_per_frame","acceleration_{}_mm_per_frame2"):
                key = template.format(suffix)
                result[key] = float(self.sums[key]/self.counts[key]) if self.counts.get(key) else None
        for threshold,hits in self.hits.items():
            tag = int(threshold*100)
            total = self.pck_counts.sum()
            result[f"PCK{tag}"] = float(hits.sum()/total*100) if total else None
            for label,index in zip(self.labels,self.hands):
                result[f"{label}_PCK{tag}"] = float(hits[index]/self.pck_counts[index]*100) if self.pck_counts[index] else None
        result["invalid_torso_frames"] = self.invalid_torso_frames
        result["degenerate_gt_frames"] = self.degenerate_gt_frames
        if self.coordinate_unit == 'scaled_pixel':
            result = {key.replace('_mm','_px'):value for key,value in result.items()}
        return result
