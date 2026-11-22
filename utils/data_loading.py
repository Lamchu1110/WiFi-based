"""Fixed sample ordering with optional persistent CPU data workers."""
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


class OrderedSampler(Sampler):
    def __init__(self):
        self.indices = []

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def make_loader(dataset,batch_size,device,workers=0,sampler=None,drop_last=False,generator=None):
    options = {'num_workers':workers, 'pin_memory':device.type=='cuda',
               'generator':generator, 'worker_init_fn':seed_worker}
    if workers:
        options.update(persistent_workers=True,prefetch_factor=2)
    return DataLoader(dataset,batch_size=batch_size,sampler=sampler,shuffle=False,
                      drop_last=drop_last,**options)
