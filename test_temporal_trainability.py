"""Regression checks for baseline preservation, adapter learning, and monitoring."""
import json
from pathlib import Path
import tempfile
import unittest

import torch

from model import STHPE
from model.tcn_block import ShortResidualTCN
from utils.trainability import TemporalMonitor
import test_wipose_ablation as helpers
from training import fit
from run_wipose_ablation import settings
from wipose.ablation_data import ablation_bundle

model_for = helpers.model_for


class TemporalTests(unittest.TestCase):
    def test_identity_matches_each_layout_in_train_and_eval(self):
        x = torch.randn(4,9,30,5)
        for layout, baseline in [('legacy','baseline_legacy'),('corrected','baseline_corrected')]:
            for train in (False,True):
                original = model_for(baseline).train(train)
                added = model_for('tcn_residual_'+layout).train(train)
                torch.manual_seed(123);a=original(x)[0]
                torch.manual_seed(123);b=added(x)[0]
                torch.testing.assert_close(a,b,rtol=0,atol=0)
                a.square().mean().backward();b.square().mean().backward()
                for (name,p),(other,q) in zip(original.backbone.named_parameters(),added.backbone.named_parameters()):
                    self.assertEqual(name,other)
                    self.assertEqual(p.grad is None,q.grad is None)
                    if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)

    def test_projection_updates_first_then_temporal_trunk_with_sgd_recipe(self):
        torch.manual_seed(41)
        model=model_for('tcn_residual_legacy').train()
        x=torch.randn(4,9,30,5)
        target=torch.rand(4,18,2)*.4  # realistic scaled-pixel order, not unit-normal labels
        opt=torch.optim.SGD(model.parameters(),lr=.001,momentum=.9)
        initial={n:p.detach().clone() for n,p in model.tcn.named_parameters()}
        for step in range(3):
            opt.zero_grad(set_to_none=True)
            (model(x)[0]-target).square().mean().backward()
            projection=model.tcn.fusion[-1]
            self.assertGreater(float(projection.weight.grad.norm()),0)
            trunk=sum(float(p.grad.abs().sum()) for p in model.tcn.branches.parameters())
            self.assertEqual(trunk,0) if step==0 else self.assertGreater(trunk,0)
            opt.step()
        self.assertGreater(float(model.tcn.fusion[-1].weight.norm()),0)
        self.assertTrue(any(not torch.equal(p,initial[n]) for n,p in model.tcn.branches.named_parameters(prefix='branches')))

    def test_monitor_does_not_change_outputs_rng_or_parameter_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            models=[model_for('tcn_residual_corrected').train() for _ in range(2)]
            opts=[torch.optim.SGD(m.parameters(),lr=.001,momentum=.9) for m in models]
            monitor=TemporalMonitor(models[1].tcn,Path(directory)/'trace.jsonl',1)
            x=torch.randn(4,9,30,5);target=torch.rand(4,18,2)*.4
            for step in range(3):
                for i,(model,opt) in enumerate(zip(models,opts)):
                    torch.manual_seed(200+step);opt.zero_grad(set_to_none=True)
                    if i:monitor.begin(step+1,1)
                    (model(x)[0]-target).square().mean().backward()
                    if i:monitor.gradients()
                    opt.step()
                    if i:monitor.finish()
                for k,v in models[0].state_dict().items():
                    torch.testing.assert_close(v,models[1].state_dict()[k],rtol=0,atol=0)
            monitor.close()
            rows=[json.loads(line) for line in (Path(directory)/'trace.jsonl').read_text().splitlines()]
            self.assertGreater(rows[0]['groups']['projection']['gradient_norm'],0)
            self.assertGreater(rows[0]['groups']['projection']['changed_elements'],0)
            self.assertIsNone(rows[0]['groups']['projection']['relative_change_from_initial'])
            self.assertEqual(rows[0]['forward_before_update']['residual_norm'],0)
            self.assertGreater(rows[-1]['forward_before_update']['residual_norm'],0)

    def test_shape_contract_and_finite_input_gradients(self):
        # GroupNorm also aggregates spatial statistics; verify shape/finite gradients,
        # not a false claim of strictly independent frequency positions.
        model=ShortResidualTCN(4,2)
        for shape in [(2,4,7,5),(2,4,7,2)]:
            x=torch.randn(*shape,requires_grad=True);y=model(x)
            self.assertEqual(y.shape,x.shape)
            torch.testing.assert_close(y,x,rtol=0,atol=0)
            y.sum().backward();self.assertTrue(torch.isfinite(x.grad).all())
        with self.assertRaises(ValueError):model(torch.randn(2,4,7,1))


class ResidualResumeTests(helpers.RunTests):
    # Inherit data-isolation and source-integrity checks, add the new architecture.
    def test_new_adapter_resume_matches_uninterrupted_with_journal(self):
        args=self.args('--trainability-every','1');bundle=ablation_bundle(args)
        full,resumed=self.root/'residual_full',self.root/'residual_resume'
        full.mkdir();resumed.mkdir()
        self.assertEqual(fit(settings(args,'tcn_residual_legacy',0),'wipose','tcn',bundle,full),'complete')
        start=settings(args,'tcn_residual_legacy',0);start.max_steps=2
        self.assertEqual(fit(start,'wipose','tcn',bundle,resumed),'interrupted')
        self.assertEqual(fit(settings(args,'tcn_residual_legacy',0,resumed/'last.pt'),'wipose','tcn',bundle,resumed),'complete')
        a,b=[torch.load(p/'last.pt',weights_only=True) for p in (full,resumed)]
        self.assertEqual(a['cursor'],b['cursor'])
        for k,v in a['model'].items():torch.testing.assert_close(v,b['model'][k],rtol=0,atol=0)
        rows=[json.loads(s) for s in (resumed/'trainability.jsonl').read_text().splitlines()]
        self.assertEqual(len({r['attempt'] for r in rows}),2)
        self.assertEqual([r['step'] for r in rows],list(range(1,len(rows)+1)))


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
