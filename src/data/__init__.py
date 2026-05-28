from torch.utils.data import DataLoader

from .dataset import PairedImageDataset

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


def create_dataloader(name, batch_size, num_workers, phase, **kwargs):
    kwargs["phase"] = phase
    dataset = create_dataset(name, **kwargs)
    shuffle = phase == "train"
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)


@register_dataset("default")
class DefaultDataset(PairedImageDataset):
    pass
