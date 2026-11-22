"""Paper recipe, leakage guards, cache fidelity, Windows workers and resume."""
import gc
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import scipy.io
import torch

from run_wipose_paper import make_parser,settings,compare
from test_wipose_trial import fixture
from training import fit,pose_objective,evaluate
from utils.data_loading import make_loader
from wipose.paper_data import paper_bundle


def populated(root):
    raw,points=fixture(root,hdf=True)
    for split in ('Train','Test'):
        for i in range(8,64):
            scipy.io.savemat(root/split/f'walk_extra_{i:03}.mat',{'CSI':raw+i*.01,'SkeletonPoints':points+i*.001})


class PaperTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='wipose_paper_')
        self.root=Path(self.temp.name)
        populated(self.root/'data')

    def tearDown(self):
        gc.collect()
        self.temp.cleanup()

    def args(self,*extra):
        args=make_parser().parse_args(['--dataset-root',str(self.root/'data'),
              '--cache-dir',str(self.root/'cache'),'--device','cpu','--pilot-train-samples','64',
              '--pilot-test-samples','32',*extra])
        args.epochs=1
        return args

    def test_reported_settings_and_plain_mse(self):
        value=settings(self.args())
        self.assertEqual((value.optimizer,value.lr,value.momentum,value.batch_size),('sgd',.001,.9,32))
        self.assertEqual((value.weight_decay,value.schedule,value.selection_policy),(0.,'constant','final'))
        target=torch.randn(2,18,3)
        target[:,:,2]=0  # Confidence must not rescale the paper MSE objective.
        pred=torch.randn(2,18,2,requires_grad=True)
        loss=pose_objective(pred,target,'xy_conf',objective_mode='paper_mse')
        torch.testing.assert_close(loss,((pred-target[:,:,:2])**2).mean(),rtol=0,atol=0)
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_split_seventy_thirty_disjoint_and_independent_of_model_seed(self):
        a=paper_bundle(self.args('--mode','train','--cache','off'))
        b=paper_bundle(self.args('--mode','train','--cache','off','--seed','8'))
        self.assertIsNone(a.val)
        self.assertEqual((len(a.train),len(a.test)),(89,39))
        self.assertEqual(a.split,b.split)
        self.assertFalse(set(a.split['train']) & set(a.split['test']))
        self.assertEqual(len(set(a.split['train']) | set(a.split['test'])),128)
        self.assertTrue(any(n.startswith('Test/') for n in a.split['train']))

    def test_cache_is_exact_immutable_and_does_not_change_split(self):
        a=paper_bundle(self.args('--cache','off'))
        b=paper_bundle(self.args())
        self.assertEqual(a.fingerprint,b.fingerprint)
        for raw,cached in ((a.train,b.train),(a.test,b.test)):
            for i in range(len(raw)):
                for key in raw[i]:
                    torch.testing.assert_close(raw[i][key],cached[i][key],rtol=0,atol=0)
        b.train[0]['input_wifi-csi'].zero_()
        torch.testing.assert_close(a.train[0]['input_wifi-csi'],b.train[0]['input_wifi-csi'],rtol=0,atol=0)

    def test_corrupted_cache_rejected(self):
        paper_bundle(self.args())
        path=next((self.root/'cache').glob('*/csi.npy'))
        with path.open('r+b') as stream:
            stream.seek(-4,2);stream.write(b'bad!')
        with self.assertRaisesRegex(ValueError,'Corrupted cache'):
            paper_bundle(self.args())

    def test_source_change_invalidates_cache_and_fingerprint(self):
        first=paper_bundle(self.args('--cache','off'))
        path=next((self.root/'data/Train').glob('*.mat'))
        stamp=path.stat()
        os.utime(path,ns=(stamp.st_atime_ns,stamp.st_mtime_ns+1000000))
        second=paper_bundle(self.args('--cache','off'))
        self.assertNotEqual(first.fingerprint,second.fingerprint)

    def test_windows_workers_preserve_cached_order(self):
        bundle=paper_bundle(self.args())
        def collect(workers):
            loader=make_loader(bundle.train,16,torch.device('cpu'),workers,
                    generator=torch.Generator().manual_seed(17))
            result=torch.cat([b['input_wifi-csi'] for b in loader])
            del loader
            return result
        torch.testing.assert_close(collect(0),collect(2),rtol=0,atol=0)

    def test_final_policy_never_selects_using_test_and_resume_is_exact(self):
        bundle=paper_bundle(self.args())
        two_epochs=self.args()
        two_epochs.epochs=2
        full,resumed=self.root/'full',self.root/'resumed'
        full.mkdir();resumed.mkdir()
        def checked_evaluate(*args,**kwargs):
            self.assertIs(args[1],bundle.test)
            out=full if not (full/'test_metrics.json').exists() else resumed
            checkpoint=torch.load(out/'last.pt',weights_only=True)
            self.assertEqual(checkpoint['cursor']['epoch'],2)
            return evaluate(*args,**kwargs)
        with patch('training.evaluate',side_effect=checked_evaluate) as spy:
            self.assertEqual(fit(settings(two_epochs),'wipose','baseline',bundle,full),'complete')
            self.assertEqual(spy.call_count,1)
            interrupted=settings(two_epochs);interrupted.max_steps=1
            self.assertEqual(fit(interrupted,'wipose','baseline',bundle,resumed),'interrupted')
            self.assertEqual(spy.call_count,1)
            continued=settings(two_epochs,resumed/'last.pt')
            continued.workers=2
            self.assertEqual(fit(continued,'wipose','baseline',bundle,resumed),'complete')
            self.assertEqual(spy.call_count,2)
        a,b=[torch.load(p/'last.pt',weights_only=True) for p in (full,resumed)]
        self.assertEqual(a['cursor'],b['cursor'])
        self.assertEqual(a['scheduler'],b['scheduler'])
        self.assertEqual(a['optimizer']['param_groups'][0]['lr'],.001)
        for key in a['model']:
            torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
        self.assertTrue((full/'final.pt').exists())
        self.assertFalse((full/'best.pt').exists())
        result=json.loads((full/'test_metrics.json').read_text())
        self.assertEqual(result['evaluated_epoch'],2)
        self.assertNotIn('best_epoch',result)
        self.assertNotIn('PCK20',(full/'metrics.csv').read_text())

    def test_suite_resume_and_comparison_contract(self):
        script=str(Path(__file__).resolve().parent/'run_wipose_paper.py')
        command=[sys.executable,script,'--dataset-root',str(self.root/'data'),'--cache-dir',str(self.root/'cache'),
                 '--device','cpu','--pilot-train-samples','64','--pilot-test-samples','32',
                 '--output',str(self.root/'runs'),'--max-steps','1']
        result=subprocess.run(command,capture_output=True,text=True,encoding='utf-8')
        self.assertEqual(result.returncode,130,result.stdout+result.stderr)
        suite=next((self.root/'runs').iterdir())
        self.assertFalse((suite/'baseline/test_metrics.json').exists())
        result=subprocess.run([sys.executable,script,'--resume-suite',str(suite)],capture_output=True,text=True,encoding='utf-8')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        data=json.loads((suite/'comparison.json').read_text())
        self.assertEqual(data['status'],'complete')
        self.assertEqual([r['val_samples'] for r in data['rows']],[0,0])
        before=[(suite/name/'final.pt').read_bytes() for name in ('baseline','tcn')]
        result=subprocess.run([sys.executable,script,'--resume-suite',str(suite)],capture_output=True,text=True,encoding='utf-8')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(before,[(suite/name/'final.pt').read_bytes() for name in ('baseline','tcn')])
        path=suite/'tcn/config.json';value=json.loads(path.read_text());value['contract']['lr']=.1
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'different training/data contracts'):
            compare(suite)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(testRunner=unittest.TextTestRunner(stream=sys.stdout,verbosity=2))
