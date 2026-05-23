from torch.utils.data import DataLoader

from .dataset import LQGTDataset

_dataset_registry = {}


def register_dataset(name):
    def decorator(cls):
        _dataset_registry[name] = cls
        return cls
    return decorator


def create_dataset(name, **kwargs) -> LQGTDataset:
    if name not in _dataset_registry:
        raise KeyError(f"Dataset '{name}' not found. Registered: {list(_dataset_registry.keys())}")
    return _dataset_registry[name](kwargs)


def create_dataloader(name, batch_size, num_workers, phase, **kwargs):
    dataset_opt = {
        "phase": phase,
        "data_type": kwargs.get("data_type", "img"),
        "dataroot_GT": kwargs.get("dataroot_GT", ""),
        "dataroot_LQ": kwargs.get("dataroot_LQ", ""),
        "LR_size": kwargs.get("patch_size", 64),
        "GT_size": kwargs.get("patch_size", 64),
        "scale": kwargs.get("scale", 1),
        "use_flip": kwargs.get("use_flip", True),
        "use_rot": kwargs.get("use_rot", True),
        "color": kwargs.get("color", "RGB"),
    }
    dataset = create_dataset(name, **dataset_opt)
    shuffle = phase == "train"
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


# Register built-in datasets
@register_dataset("rain100h")
class Rain100HDataset(LQGTDataset):
    pass


@register_dataset("rain100l")
class Rain100LDataset(LQGTDataset):
    pass


@register_dataset("default")
class DefaultDataset(LQGTDataset):
    pass
