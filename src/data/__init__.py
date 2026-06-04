import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import PairedImageDataset, SCCIDDataset

_dataset_registry = {}


def register_dataset(name):
    def decorator(cls):
        _dataset_registry[name] = cls
        return cls
    return decorator


def create_dataset(name, **kwargs) -> PairedImageDataset:
    if name not in _dataset_registry:
        raise KeyError(f"Dataset '{name}' not found. Registered: {list(_dataset_registry.keys())}")
    return _dataset_registry[name](**kwargs)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloader(name, batch_size, num_workers, phase, **kwargs):
    kwargs["phase"] = phase
    dataset = create_dataset(name, **kwargs)
    shuffle = phase == "train"
    g = torch.Generator()
    g.manual_seed(42)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      worker_init_fn=seed_worker, generator=g)


@register_dataset("derainh")
class DerainHDataset(PairedImageDataset):
    pass


@register_dataset("sccid")
class SCCID(SCCIDDataset):
    pass
