"""Regression tests for real temporal dependence, metrics, and mid-epoch recovery."""
import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import Dataset

from model import STHPE
from dataset_lib.training_data import Bundle,NpzPoses,PoseWindows,MMFiStore
from training import fit,pose_objective
from utils.losses import KinematicPoseLoss
from utils.metrics import PoseMetrics
from utils.runtime import RunLock


class SmallPoses(Dataset):
    def __init__(self,count,seed):
        generator = torch.Generator().manual_seed(seed)
        self.x = torch.randn(count,2,3,114,10,generator=generator)
        self.y = torch.randn(count,2,17,3,generator=generator)

    def __len__(self):
        return len(self.x)

    def __getitem__(self,index):
        return {'input_wifi-csi':self.x[index],'output':self.y[index]}


def settings(**changes):
    values = dict(epochs=2,batch_size=2,eval_batch_size=2,lr=0.001,momentum=0.9,weight_decay=0.,
                  temporal_axis='pose',legacy_objective=False,seed=7,device='cpu',amp=False,amp_dtype='bfloat16',deterministic=True,
                  warmup_epochs=1,lambda_bone=0.1,lambda_smooth=0.05,lambda_sym=0.02,symmetry_mode='target',
                  grad_clip=1.,select_metric='MPJPE_xy_mm',select_mode='min',preflight_only=False,
                  resume=None,log_every=100,checkpoint_every=1,max_steps=0)
    values.update(changes)
    return SimpleNamespace(**values)


class ReliabilityTests(unittest.TestCase):
    def test_wipose_preserves_measured_csi_time(self):
        torch.manual_seed(10)
        model = STHPE('wipose',temporal_axis='csi').train()
        seen = []
        hook = model.tcn.register_forward_pre_hook(lambda module,inputs:seen.append(tuple(inputs[0].shape)))
        pred = model(torch.randn(2,9,30,5))[0]
        pred.square().mean().backward()
        hook.remove()
        self.assertEqual(seen,[(2,256,5)])
        weights = model.tcn.branches[0][0].weight.grad
        self.assertTrue(torch.count_nonzero(weights[:,:,0]) > 0,'Temporal off-center taps must receive gradient')

    def test_neighbor_pose_changes_center_prediction_in_eval(self):
        for dataset,shape in [('mmfi',(3,114,10)),('wipose',(9,30,5))]:
            with self.subTest(dataset=dataset):
                torch.manual_seed(3)
                model = STHPE(dataset,temporal_axis='pose').eval()
                inputs = torch.randn(1,8,*shape)
                changed = inputs.clone()
                # Only the dilation-4 branch can connect frame 4 to frame 0.
                changed[:,4] += 4
                with torch.no_grad():
                    a,b = model(inputs)[0][:,0],model(changed)[0][:,0]
                self.assertGreater((a-b).abs().max().item(),1e-10)
                independent = STHPE(dataset,use_tcn=False,temporal_axis='pose').eval()
                independent.backbone.load_state_dict(model.backbone.state_dict())
                with torch.no_grad():
                    torch.testing.assert_close(independent(inputs)[0][:,0],independent(changed)[0][:,0],rtol=0,atol=0)

    def test_corrected_xy_loss_ignores_z(self):
        pred = torch.zeros(2,17,2)
        gt = torch.ones(2,17,3)
        other = gt.clone()
        other[...,2] = 10
        torch.testing.assert_close(pose_objective(pred,gt,'xyz'),pose_objective(pred,other,'xyz'))
        self.assertNotEqual(pose_objective(pred,gt,'xyz',True).item(),pose_objective(pred,other,'xyz',True).item())

    def test_xyz_ablations_share_initial_backbone_and_decoder(self):
        torch.manual_seed(7)
        a = STHPE('mmfi',use_tcn=False,direct_3d=True)
        torch.manual_seed(7)
        b = STHPE('mmfi',use_tcn=True,direct_3d=True)
        for key,value in a.backbone.state_dict().items():
            torch.testing.assert_close(value,b.backbone.state_dict()[key],rtol=0,atol=0)

    def test_ground_truth_is_zero_kinematic_loss(self):
        gt = torch.randn(2,4,17,3,requires_grad=True)
        loss,parts = KinematicPoseLoss()(gt,gt.detach(),epoch=10,return_components=True)
        self.assertEqual(loss.item(),0)
        self.assertTrue(all(value.item()==0 for value in parts.values()))
        loss.backward()
        self.assertTrue(torch.isfinite(gt.grad).all())

    def test_metrics_alignment_degenerate_torso_and_temporal_units(self):
        rng = np.random.default_rng(4)
        gt = rng.normal(size=(2,17,3))
        rotation = np.array([[0,-1],[1,0]])
        prediction = gt[...,:2]@rotation*2+3
        metric = PoseMetrics('mmfi')
        metric.update(prediction,gt)
        self.assertLess(metric.result()['PA_MPJPE_xy_mm'],1e-8)
        self.assertGreater(metric.result()['MPJPE_xy_mm'],0)
        collapsed = PoseMetrics('mmfi')
        collapsed.update(np.ones((1,17,2)),np.zeros((1,17,3)))
        result = collapsed.result()
        self.assertIsNone(result['PCK20'])
        self.assertIsNone(result['PA_MPJPE_xy_mm'])
        self.assertEqual(result['invalid_torso_frames'],1)
        sequence = rng.normal(size=(1,4,17,3))
        pred = sequence+0.25
        temporal = PoseMetrics('mmfi',xyz=True)
        temporal.update(pred,sequence)
        self.assertAlmostEqual(temporal.result()['velocity_xyz_mm_per_frame'],0,places=9)
        pred[:,1] += 0.1
        noisy = PoseMetrics('mmfi',xyz=True)
        noisy.update(pred,sequence)
        self.assertGreater(noisy.result()['velocity_xyz_mm_per_frame'],0)

    def test_visibility_and_streaming_weighting(self):
        rng = np.random.default_rng(2)
        gt = rng.normal(size=(3,18,3))
        prediction = gt[...,:2].copy()
        confidence = np.ones((3,18))
        confidence[0,4] = 0
        prediction[0,4] += 1e9
        full = PoseMetrics('wipose')
        full.update(prediction,gt,confidence)
        self.assertEqual(full.result()['MPJPE_xy_mm'],0)
        pieces = PoseMetrics('wipose')
        pieces.update(prediction[:1],gt[:1],confidence[:1])
        pieces.update(prediction[1:],gt[1:],confidence[1:])
        for key,value in full.result().items():
            if value is None:
                self.assertIsNone(pieces.result()[key])
            else:
                self.assertAlmostEqual(value,pieces.result()[key],places=9)

    def test_windows_do_not_cross_sequences_or_missing_frames(self):
        frames = SimpleNamespace(records=[('a',0,''),('a',1,''),('a',2,''),('a',3,''),
                                          ('b',0,''),('b',1,''),('b',3,'')])
        windows = PoseWindows(frames,range(7),2,2)
        self.assertEqual(windows.windows,[(0,1),(2,3),(4,5)])

    def test_npz_rejects_string_order_flag(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'bad.npz'
            np.savez(path,x=np.zeros((2,2,9,30,5)),y=np.zeros((2,2,18,3)),
                     target_format=np.array('xyz'),coordinate_unit=np.array('m'),windows_ordered=np.array('False'))
            with self.assertRaises(ValueError):
                NpzPoses(path,True)

    def test_directory_and_zip_read_the_same_normalized_frame(self):
        import scipy.io
        import zipfile
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            action = root/'data/E01/S01/A01'
            (action/'wifi-csi').mkdir(parents=True)
            rng = np.random.default_rng(3)
            np.save(action/'ground_truth.npy',rng.normal(size=(4,17,3)).astype('float32'))
            csi = rng.normal(size=(3,114,10)).astype('float32')
            csi[0,0,0] = np.nan
            scipy.io.savemat(action/'wifi-csi/frame001.mat',{'CSIamp':csi})
            archive = root/'test.zip'
            with zipfile.ZipFile(archive,'w') as file:
                for path in action.rglob('*'):
                    if path.is_file():
                        file.write(path,'MMFi_Dataset/'+path.relative_to(root/'data').as_posix())
            directory_store,zip_store = MMFiStore(root/'data'),MMFiStore(archive=archive)
            try:
                a = directory_store.sequence('E01','S01','A01')
                b = zip_store.sequence('E01','S01','A01')
                torch.testing.assert_close(a[1],b[1],rtol=0,atol=0)
                torch.testing.assert_close(directory_store.read(a[2][0][1]),zip_store.read(b[2][0][1]),rtol=0,atol=0)
            finally:
                zip_store.zip.close()

    def test_mid_epoch_resume_matches_uninterrupted_and_rejects_changed_contract(self):
        bundle = Bundle(SmallPoses(4,1),SmallPoses(2,2),SmallPoses(2,3),'xyz','fixture-data-v2',{'mode':'fixture'},2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            continuous,resumed = root/'continuous',root/'resumed'
            continuous.mkdir()
            resumed.mkdir()
            self.assertEqual(fit(settings(),'mmfi','full',bundle,continuous),'complete')
            self.assertEqual(fit(settings(max_steps=1),'mmfi','full',bundle,resumed),'interrupted')
            saved = torch.load(resumed/'last.pt',weights_only=True)
            self.assertEqual(saved['cursor']['next_batch'],1)
            self.assertEqual(saved['cursor']['epoch'],0)
            self.assertEqual(fit(settings(resume=resumed/'last.pt'),'mmfi','full',bundle,resumed),'complete')
            a = torch.load(continuous/'last.pt',weights_only=True)
            b = torch.load(resumed/'last.pt',weights_only=True)
            for key,value in a['model'].items():
                torch.testing.assert_close(value,b['model'][key],rtol=0,atol=0)
            self.assertEqual(a['cursor'],b['cursor'])
            self.assertEqual((continuous/'metrics.csv').read_text(),(resumed/'metrics.csv').read_text())
            self.assertEqual(json.loads((continuous/'test_metrics.json').read_text()),json.loads((resumed/'test_metrics.json').read_text()))
            before = hashlib.sha256((resumed/'last.pt').read_bytes()).hexdigest()
            with self.assertRaises(ValueError):
                fit(settings(resume=resumed/'last.pt',lr=0.02),'mmfi','full',bundle,resumed)
            self.assertEqual(before,hashlib.sha256((resumed/'last.pt').read_bytes()).hexdigest())

    def test_os_lock_prevents_parallel_writers(self):
        with tempfile.TemporaryDirectory() as temporary:
            with RunLock(temporary):
                with self.assertRaises(RuntimeError):
                    with RunLock(temporary):
                        pass
            with RunLock(temporary):
                pass

    def test_cli_resume_uses_the_checkpoint_directory_after_copy(self):
        import subprocess
        import sys
        import shutil
        root = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            data = work/'data'
            data.mkdir()
            rng = np.random.default_rng(5)
            for split,count in [('train',4),('val',2),('test',2)]:
                np.savez(data/f'{split}.npz',x=rng.normal(size=(count,2,9,30,5)).astype('float32'),
                         y=rng.normal(size=(count,2,18,3)).astype('float32'),
                         target_format=np.array('xyz'),coordinate_unit=np.array('m'),windows_ordered=np.array(True))
            common = [sys.executable,'-X','utf8',str(root/'train_wipose.py'),'--npz-dir',str(data),
                      '--epochs','1','--batch-size','2','--device','cpu']
            result = subprocess.run(common+['--scenario','full','--max-steps','1','--output',str(work/'runs')],
                                    text=True,encoding='utf-8',stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            self.assertEqual(result.returncode,0,result.stdout)
            previous = next((work/'runs').glob('*/full'))
            migrated = work/'renamed-run'
            shutil.copytree(previous,migrated)
            result = subprocess.run(common+['--resume',str(migrated/'last.pt')],text=True,encoding='utf-8',
                                    stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            self.assertEqual(result.returncode,0,result.stdout)
            self.assertEqual(json.loads((migrated/'status.json').read_text())['status'],'complete')
            self.assertTrue((migrated/'test_metrics.json').exists())
            self.assertEqual(json.loads((previous/'status.json').read_text())['status'],'interrupted')


if __name__=='__main__':
    os.environ.setdefault('OMP_NUM_THREADS','2')
    os.environ.setdefault('MKL_NUM_THREADS','2')
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
