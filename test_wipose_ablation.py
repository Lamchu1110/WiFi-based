"""Semantic tensor, gradient, data-isolation, resume and evidence regression tests."""
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from model import STHPE
from model.utils.SKNet import SKConv
from run_wipose_ablation import make_parser, settings, compare
from test_wipose_paper import populated
from training import fit
from utils.experiment_archive import record_source
from wipose.ablation_config import VARIANTS
from wipose.ablation_data import ablation_bundle
from wipose.wipose_dataset import WiPoseDataset


def model_for(name):
    scenario,layout,pooling,block=VARIANTS[name]
    torch.manual_seed(11)
    return STHPE('wipose',use_tcn=scenario=='tcn',direct_3d=False,dsk_layout=layout,pooling_mode=pooling,temporal_block=block)


class FixedBranch(nn.Module):
    def __init__(self, values):
        super().__init__();self.register_buffer('values',values)

    def forward(self,x):
        return self.values.expand(len(x),-1,-1,-1)


class ArchitectureTests(unittest.TestCase):
    def test_corrected_attention_preserves_channel_frequency_positions(self):
        values=[torch.arange(30,dtype=torch.float32).reshape(1,3,5,2)+100*i for i in range(4)]
        for mode in ('freq','freq-time','freq-chan'):
            module=SKConv(3,3,5,2,mode,r=1,layout_mode='corrected').eval()
            module.convs=nn.ModuleList(FixedBranch(value) for value in values)
            for layer in module.fcs:
                nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
            result=module(torch.zeros(2,3,5,2))
            expected=torch.stack(values,dim=1).mean(dim=1).expand(2,-1,-1,-1)
            torch.testing.assert_close(result,expected,rtol=0,atol=0)
            # Known nonuniform branch attention must still act on the same locations.
            logits=torch.tensor([-1.,0.,1.,2.])
            for i,layer in enumerate(module.fcs):nn.init.constant_(layer.bias,float(logits[i]))
            expected=(torch.stack(values,dim=1)*logits.softmax(0).reshape(1,4,1,1,1)).sum(1).expand(2,-1,-1,-1)
            torch.testing.assert_close(module(torch.zeros(2,3,5,2)),expected)

    def test_legacy_backbone_forward_and_initialization_are_preserved(self):
        x=torch.randn(2,9,30,5)
        baseline=model_for('baseline_legacy').eval()
        torch.testing.assert_close(baseline(x)[0],baseline.backbone(x)[0],rtol=0,atol=0)
        for name in VARIANTS:
            model=model_for(name)
            for key,value in baseline.backbone.state_dict().items():
                torch.testing.assert_close(value,model.backbone.state_dict()[key],rtol=0,atol=0)

    def test_gate_identity_and_learning_beyond_first_update(self):
        baseline=model_for('baseline_corrected')
        gated=model_for('tcn_gated')
        x=torch.randn(4,9,30,5);target=torch.randn(4,18,2)
        for train in (False,True):
            baseline.train(train);gated.train(train)
            torch.manual_seed(31);expected=baseline(x)[0]
            torch.manual_seed(31);actual=gated(x)[0]
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        optimizer=torch.optim.SGD(gated.parameters(),lr=.1)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss=(gated(x)[0]-target).square().mean();loss.backward()
            self.assertTrue(torch.isfinite(gated.tcn.alpha.grad))
            if step==0:self.assertGreater(float(gated.tcn.alpha.grad.abs()),0)
            else:self.assertGreater(sum(float(p.grad.abs().sum()) for p in gated.tcn.branches.parameters()),0)
            optimizer.step()

    def test_all_variants_backward_and_corrected_mmfi_shape(self):
        for name in VARIANTS:
            model=model_for(name);pose=model(torch.randn(2,9,30,5))[0]
            self.assertEqual(tuple(pose.shape),(2,18,2))
            pose.square().mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        model=STHPE('mmfi',use_tcn=True,direct_3d=True,dsk_layout='corrected')
        pose=model(torch.randn(2,3,114,10))[0]
        self.assertEqual(tuple(pose.shape),(2,17,3));pose.square().mean().backward()


class RunTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='wipose_v3_');self.root=Path(self.temp.name)
        populated(self.root/'data')

    def tearDown(self):
        gc.collect();self.temp.cleanup()

    def args(self,*extra):
        args=make_parser().parse_args(['--dataset-root',str(self.root/'data'),'--cache-dir',str(self.root/'cache'),
            '--mode','train','--device','cpu','--batch-size','8',*extra])
        args.epochs=1
        return args

    def test_fixed_outer_split_inner_validation_and_no_holdout_reads(self):
        args=self.args('--cache','off')
        raw=ablation_bundle(args)
        self.assertEqual((len(raw.train),len(raw.val),len(raw.test)),(80,9,39))
        for a,b in [('train','val'),('train','test'),('val','test')]:
            self.assertFalse(set(raw.split[a])&set(raw.split[b]))
        heldout=set(raw.split['test'])
        original=WiPoseDataset.__getitem__
        def guard(dataset,index):
            name=dataset.directory.name+'/'+dataset.file_list[index]
            self.assertNotIn(name,heldout,'Development touched test tensors')
            return original(dataset,index)
        with patch.object(WiPoseDataset,'__getitem__',guard):
            cached=ablation_bundle(self.args())
            for a,b in [(raw.train,cached.train),(raw.val,cached.val)]:
                for i in range(len(a)):
                    for key in a[i]:torch.testing.assert_close(a[i][key],b[i][key],rtol=0,atol=0)
            out=self.root/'preflight';out.mkdir()
            options=settings(args,'baseline_corrected',0);options.preflight_only=True
            fit(options,'wipose','baseline',raw,out)
        final=ablation_bundle(self.args('--stage','final','--cache','off','--batch-size','16'))
        self.assertEqual(final.split['test'],raw.split['test'])
        self.assertEqual(final.split['train'],raw.split['outer_train'])
        args.seeds=[99]
        self.assertEqual(ablation_bundle(args).split,raw.split)

    def test_imported_split_and_invalid_overlap(self):
        original=ablation_bundle(self.args('--stage','final','--cache','off','--batch-size','16'))
        path=self.root/'outer.json';path.write_text(json.dumps(original.split))
        imported=ablation_bundle(self.args('--split-file',str(path),'--cache','off'))
        self.assertEqual(imported.split['test'],original.split['test'])
        wrong=dict(original.split);wrong['train']=wrong['train']+[wrong['test'][0]]
        path.write_text(json.dumps(wrong))
        with self.assertRaisesRegex(ValueError,'exactly once'):ablation_bundle(self.args('--split-file',str(path)))

    def test_singleton_final_batch_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'singleton last batch'):
            ablation_bundle(self.args('--stage','final','--cache','off'))

    def test_development_never_evaluates_test_and_resume_is_exact(self):
        args=self.args();bundle=ablation_bundle(args)
        class Forbidden:
            def __len__(self):return 39
            def __getitem__(self,index):raise AssertionError('Held-out test was read')
        bundle.test=Forbidden()
        full,resumed=self.root/'full',self.root/'resumed';full.mkdir();resumed.mkdir()
        options=settings(args,'tcn_gated',0)
        self.assertEqual(fit(options,'wipose','tcn',bundle,full),'complete')
        interrupted=settings(args,'tcn_gated',0);interrupted.max_steps=1
        self.assertEqual(fit(interrupted,'wipose','tcn',bundle,resumed),'interrupted')
        continued=settings(args,'tcn_gated',0,resumed/'last.pt')
        self.assertEqual(fit(continued,'wipose','tcn',bundle,resumed),'complete')
        a,b=[torch.load(p/'last.pt',weights_only=True) for p in (full,resumed)]
        self.assertEqual(a['cursor'],b['cursor'])
        for key in a['model']:torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
        self.assertFalse((full/'test_metrics.json').exists())
        self.assertTrue((full/'validation_metrics.json').exists())
        with np.load(full/'validation_predictions.npz') as data:self.assertEqual(data['prediction'].shape,(9,18,2))
        changed=settings(args,'tcn_corrected',0,resumed/'last.pt')
        with self.assertRaisesRegex(ValueError,'contract mismatch'):fit(changed,'wipose','tcn',bundle,resumed)

    def test_source_snapshot_includes_uncommitted_and_rejects_changes(self):
        source=self.root/'source';source.mkdir();suite=self.root/'suite';suite.mkdir()
        (source/'model.py').write_text('x=1\n');(source/'.env').write_text('excluded\n')
        first=record_source(source,suite)
        self.assertEqual(record_source(source,suite,resume=True),first)
        manifest=json.loads((suite/'source_manifest.json').read_text())
        self.assertEqual([item['path'] for item in manifest['files']],['model.py'])
        (source/'model.py').write_text('x=2\n')
        with self.assertRaisesRegex(ValueError,'source changed'):record_source(source,suite,resume=True)

    def test_cli_final_resume_aggregation_and_completed_preservation(self):
        script=str(Path(__file__).parent/'run_wipose_ablation.py')
        command=[sys.executable,script,'--dataset-root',str(self.root/'data'),'--cache-dir',str(self.root/'cache'),
            '--output',str(self.root/'runs'),'--device','cpu','--stage','final','--batch-size','8',
            '--pilot-train-samples','32','--pilot-eval-samples','16',
            '--variants','baseline_corrected','tcn_gated','--seeds','0','1','--max-steps','1']
        env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        value=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',env=env)
        self.assertEqual(value.returncode,130,value.stdout+value.stderr)
        suite=next((self.root/'runs').iterdir())
        value=subprocess.run([sys.executable,script,'--resume-suite',str(suite)],capture_output=True,text=True,encoding='utf-8',env=env)
        self.assertEqual(value.returncode,0,value.stdout+value.stderr)
        result=compare(suite)
        self.assertEqual(result['status'],'complete');self.assertEqual(len(result['paired_differences']),2)
        self.assertTrue(all(row['runs']==2 for row in result['mean_and_sample_std']))
        files=list(suite.glob('seed_*/*/final.pt'))
        before=[p.read_bytes() for p in files]
        value=subprocess.run([sys.executable,script,'--resume-suite',str(suite)],capture_output=True,text=True,encoding='utf-8',env=env)
        self.assertEqual(value.returncode,0,value.stdout+value.stderr)
        self.assertEqual(before,[p.read_bytes() for p in files])
        path=suite/'seed_0/tcn_gated/config.json';config=json.loads(path.read_text());config['contract']['lr']=.1
        path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError,'different training'):compare(suite)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(testRunner=unittest.TextTestRunner(stream=sys.stdout,verbosity=2))
