"""Sample temporal adapter gradients and actual updates, without extra forwards.

The append-only journal includes an attempt ID so resume replay is explicit.
No random draws, input mutation, optimizer changes, or validation/test reads.
"""
import json
import math
from pathlib import Path
import uuid

import torch


class TemporalMonitor:
    def __init__(self, module, path, every=100):
        self.module, self.path, self.every = module, Path(path), every
        self.parameters = dict(module.named_parameters())
        self.initial = {k:p.detach().cpu().clone() for k,p in self.parameters.items()}
        self.attempt = uuid.uuid4().hex
        self.capture = False
        self.handle = module.register_forward_hook(self._hook) if self.parameters and every else None

    def begin(self, step, epoch, last_batch=False):
        self.capture = bool(self.handle and (step <= 5 or step % self.every == 0 or last_batch))
        if self.capture:
            self.step, self.epoch = step, epoch
            self.before = {k:p.detach().cpu().clone() for k,p in self.parameters.items()}
            self.forward = {}

    def _hook(self, module, inputs, output):
        if not self.capture:
            return
        x = inputs[0].detach().float()
        residual = output.detach().float()-x
        input_norm = float(torch.linalg.vector_norm(x))
        residual_norm = float(torch.linalg.vector_norm(residual))
        self.forward = {'input_norm':input_norm,'residual_norm':residual_norm,
                        'residual_input_ratio':residual_norm/input_norm if input_norm else None}

    def gradients(self):
        if self.capture:
            self.grads = {k:p.grad.detach().cpu().clone() if p.grad is not None else None
                          for k,p in self.parameters.items()}

    def finish(self):
        if not self.capture:
            return
        groups = {}
        for name,p in self.parameters.items():
            group = 'gate' if name == 'alpha' else 'projection' if name.startswith('fusion.2.') else 'trunk'
            row = groups.setdefault(group,dict(parameters=0,gradient_sq=0.,update_sq=0.,
                initial_sq=0.,change_sq=0.,changed_elements=0,missing_gradient_tensors=0))
            current = p.detach().cpu()
            initial, before, gradient = self.initial[name],self.before[name],self.grads[name]
            row['parameters'] += p.numel()
            row['initial_sq'] += float(initial.double().square().sum())
            row['change_sq'] += float((current.double()-initial.double()).square().sum())
            row['update_sq'] += float((current.double()-before.double()).square().sum())
            row['changed_elements'] += int((current != before).sum())
            if gradient is None:
                row['missing_gradient_tensors'] += 1
            else:
                row['gradient_sq'] += float(gradient.double().square().sum())
        for row in groups.values():
            for key in ['gradient','update','initial','change']:
                row[key+'_norm'] = math.sqrt(row.pop(key+'_sq'))
            row['relative_change_from_initial'] = row['change_norm']/row['initial_norm'] if row['initial_norm'] else None
        record = {'attempt':self.attempt,'epoch':self.epoch,'step':self.step,
                  'scope':'sampled actual training step; gradient after clipping, before optimizer; update after optimizer',
                  'forward_before_update':self.forward,'groups':groups}
        if hasattr(self.module,'alpha'):
            record['alpha_after_update'] = float(self.module.alpha.detach())
        with self.path.open('a',encoding='utf-8') as f:
            f.write(json.dumps(record,allow_nan=False)+'\n')
        self.capture = False

    def close(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
