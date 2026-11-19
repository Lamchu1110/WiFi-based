"""WiPose schema, physical-unit labels, fair splits, optimizer and suite checks."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import scipy.io
import torch

from model import HPEWiPoseModel, STHPE
from run_wipose_comparison import make_parser, prepare_bundle, training_settings, write_comparison
from training import fit
from utils.metrics import PoseMetrics
from wipose.wipose_dataset import MEAN, STD, WiPoseDataset


def fixture(root, hdf=False):
    raw=np.arange(5*30*3*3,dtype=np.float64).reshape(5,30,3,3)/100
    points=np.stack([np.arange(18)*7+30,np.arange(18)*11+40,np.linspace(.2,1,18)])
    for split,count in [('Train',8),('Test',8)]:
        directory=root/split
        directory.mkdir(parents=True,exist_ok=True)
        for i in range(count):
            path=directory/f'walk_{i//4:03}-frame{i%4:03}.mat'
            labels=points.copy();labels[:2]+=i
            if hdf:
                with h5py.File(path,'w',userblock_size=512) as archive:
                    for name,value in [('CSI',raw),('SkeletonPoints',labels.reshape(54,1))]:
                        item=archive.create_dataset(name,data=value.T)
                        item.attrs['MATLAB_class']=np.bytes_('double')
                with path.open('r+b') as stream:
                    header=b'MATLAB 7.3 MAT-file, Platform: TEST, Created for regression'
                    stream.write(header.ljust(116,b' ')+b'\0'*8+b'\0\2IM')
            else:
                scipy.io.savemat(path,{'CSI':raw,'SkeletonPoints':labels})
    return raw,points


class WiPoseTrialTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='wipose_trial_test_')
        self.root=Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_v5_and_v73_match_upstream_csi_axes_and_labels(self):
        raw,points=fixture(self.root/'v5')
        fixture(self.root/'v73',hdf=True)
        expected=(raw.transpose(3,2,1,0).reshape(9,30,5)-np.array(MEAN)[:,None,None])/np.array(STD)[:,None,None]
        for directory in ('v5','v73'):
            sample=WiPoseDataset(self.root/directory)[0]
            torch.testing.assert_close(sample['input_wifi-csi'],torch.tensor(expected,dtype=torch.float32))
            torch.testing.assert_close(sample['output'][:,:2],torch.tensor(points[:2].T*.001,dtype=torch.float32))
            torch.testing.assert_close(sample['output'][:,2],torch.tensor(points[2],dtype=torch.float32))

    def test_public_schema_and_ambiguous_vector_are_explicit(self):
        raw,points=fixture(self.root)
        path=self.root/'Train/walk_000-frame000.mat'
        xycc=np.column_stack([points.T,points[2]])
        scipy.io.savemat(path,{'csi_serial':raw,'jointsVector':xycc})
        sample=WiPoseDataset(self.root)[0]
        torch.testing.assert_close(sample['output'][:,2],torch.tensor(points[2],dtype=torch.float32))
        scipy.io.savemat(path,{'csi_serial':raw,'jointsVector':xycc.reshape(-1)})
        with self.assertRaisesRegex(ValueError,'ambiguous ordering'):
            WiPoseDataset(self.root)[0]
        explicit=WiPoseDataset(self.root,vector_layout='joint-major')[0]
        torch.testing.assert_close(explicit['output'],sample['output'])

    def test_reject_wrong_axis_order_and_negative_confidence(self):
        raw,points=fixture(self.root)
        path=self.root/'Train/walk_000-frame000.mat'
        scipy.io.savemat(path,{'CSI':raw.transpose(1,0,2,3),'SkeletonPoints':points})
        with self.assertRaisesRegex(ValueError,'CSI shape'):
            WiPoseDataset(self.root)[0]
        points[2,0]=-1
        scipy.io.savemat(path,{'CSI':raw,'SkeletonPoints':points})
        with self.assertRaisesRegex(ValueError,'nonnegative'):
            WiPoseDataset(self.root)[0]

    def test_pilot_subsets_and_root_detection_are_reproducible(self):
        fixture(self.root/'Wi-Pose')
        args=make_parser().parse_args(['--dataset-root',str(self.root),'--pilot-train-samples','4','--pilot-eval-samples','2'])
        a,b=prepare_bundle(args),prepare_bundle(args)
        self.assertEqual(a.fingerprint,b.fingerprint)
        self.assertEqual(a.split,b.split)
        self.assertEqual((len(a.train),len(a.val),len(a.test)),(4,2,2))
        self.assertFalse(set(a.split['val']) & set(a.split['test']))
        self.assertEqual(a.coordinate_unit,'scaled_pixel')

    def test_native_metrics_are_pixels_not_physical_mm(self):
        target=np.zeros((1,18,3)); target[0,:,0]=np.arange(18)*.01
        predicted=target[:,:,:2].copy();predicted[:,:,1]+=.003
        metric=PoseMetrics('wipose',coordinate_unit='scaled_pixel')
        metric.update(predicted,target,np.ones((1,18)))
        result=metric.result()
        self.assertAlmostEqual(result['MPJPE_xy_px'],3)
        self.assertNotIn('MPJPE_xy_mm',result)

    def test_xy_baseline_matches_original_and_tcn_has_same_initial_backbone(self):
        torch.manual_seed(7);original=HPEWiPoseModel().eval()
        torch.manual_seed(7);baseline=STHPE('wipose',False,False,'csi').eval()
        torch.manual_seed(7);improved=STHPE('wipose',True,False,'csi').eval()
        for key,value in original.state_dict().items():
            torch.testing.assert_close(value,baseline.backbone.state_dict()[key],rtol=0,atol=0)
            torch.testing.assert_close(value,improved.backbone.state_dict()[key],rtol=0,atol=0)
        data=torch.randn(2,9,30,5)
        with torch.inference_mode():
            torch.testing.assert_close(original(data)[0],baseline(data)[0],rtol=0,atol=0)

    def test_adamw_mid_epoch_resume_preserves_weights_and_scheduler(self):
        fixture(self.root/'data')
        args=make_parser().parse_args(['--dataset-root',str(self.root/'data'),'--batch-size','4','--eval-batch-size','2','--device','cpu'])
        args.epochs=1
        bundle=prepare_bundle(args)
        full,resume=self.root/'continuous',self.root/'resumed'
        full.mkdir();resume.mkdir()
        settings=training_settings(args)
        settings.log_every=100
        self.assertEqual(fit(settings,'wipose','baseline',bundle,full),'complete')
        interrupted=training_settings(args);interrupted.max_steps=1
        self.assertEqual(fit(interrupted,'wipose','baseline',bundle,resume),'interrupted')
        continued=training_settings(args,resume/'last.pt')
        self.assertEqual(fit(continued,'wipose','baseline',bundle,resume),'complete')
        a,b=[torch.load(p/'last.pt',weights_only=True,map_location='cpu') for p in (full,resume)]
        for key in a['model']:
            torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
        self.assertEqual(a['scheduler'],b['scheduler'])
        self.assertEqual(a['contract']['optimizer'],'adamw')
        self.assertEqual(a['optimizer']['param_groups'][0]['weight_decay'],.01)

    def test_suite_exports_two_completed_results_and_resumes_without_retraining(self):
        fixture(self.root/'data')
        root=Path(__file__).resolve().parent
        command=[sys.executable,str(root/'run_wipose_comparison.py'),'--dataset-root',str(self.root/'data'),
                 '--batch-size','4','--eval-batch-size','2','--device','cpu','--output',str(self.root/'runs')]
        result=subprocess.run(command,capture_output=True,text=True,encoding='utf-8')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        suite=next((self.root/'runs').iterdir())
        comparison=json.loads((suite/'comparison.json').read_text())
        self.assertEqual(comparison['status'],'complete')
        self.assertEqual([r['scenario'] for r in comparison['rows']],['baseline','tcn'])
        self.assertEqual([r['coordinate_unit'] for r in comparison['rows']],['px','px'])
        self.assertTrue((suite/'comparison.csv').is_file())
        before=[(suite/name/'last.pt').read_bytes() for name in ('baseline','tcn')]
        resumed=subprocess.run([sys.executable,str(root/'run_wipose_comparison.py'),'--resume-suite',str(suite)],
                               capture_output=True,text=True,encoding='utf-8')
        self.assertEqual(resumed.returncode,0,resumed.stdout+resumed.stderr)
        self.assertEqual(before,[(suite/name/'last.pt').read_bytes() for name in ('baseline','tcn')])
        saved_split=(suite/'split.json').read_bytes()
        extra=self.root/'data/Test/extra.mat'
        extra.write_bytes(next((self.root/'data/Test').glob('*.mat')).read_bytes())
        changed=subprocess.run([sys.executable,str(root/'run_wipose_comparison.py'),'--resume-suite',str(suite)],
                               capture_output=True,text=True,encoding='utf-8')
        self.assertEqual(changed.returncode,1)
        self.assertIn('Dataset changed',changed.stdout+changed.stderr)
        self.assertEqual(saved_split,(suite/'split.json').read_bytes())
        self.assertEqual(before,[(suite/name/'last.pt').read_bytes() for name in ('baseline','tcn')])
        # A comparison cannot silently mix different recipes.
        config=suite/'tcn/config.json'
        value=json.loads(config.read_text());value['contract']['lr']=0.1
        config.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'different training/data contracts'):
            write_comparison(suite)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(testRunner=unittest.TextTestRunner(stream=sys.stdout,verbosity=2))
