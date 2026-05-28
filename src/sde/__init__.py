from .base import BaseSDE
from .goub import GOUB
from .ve_bridge import VEBridge
from .vp_bridge import VPBridge

_sde_registry = {
    "goub": GOUB,
    "ve": VEBridge,
    "vp": VPBridge,
}


def create_sde(sde_type: str, **kwargs) -> BaseSDE:
    if sde_type not in _sde_registry:
        raise KeyError(f"Unknown SDE type '{sde_type}'. Available: {list(_sde_registry.keys())}")
    return _sde_registry[sde_type](**kwargs)
