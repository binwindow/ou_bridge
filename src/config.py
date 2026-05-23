from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class SDEConfig:
    lambda_square: float = 30.0
    T: int = 100
    schedule: str = "cosine"  # "cosine" | "linear" | "constant"
    eps: float = 0.005


@dataclass
class NetworkConfig:
    architecture: str = "ConditionalUNet"
    in_nc: int = 3
    out_nc: int = 3
    nf: int = 64
    depth: int = 4


@dataclass
class TrainConfig:
    total_iterations: int = 200000
    batch_size: int = 8
    patch_size: int = 64
    lr: float = 1e-4
    min_lr: float = 1e-6
    optimizer: str = "AdamW"
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.99
    ema_beta: float = 0.999
    ema_update_every: int = 1
    use_amp: bool = True
    use_ddp: bool = False
    seed: int = 42
    num_val: int = 15
    log_every: int = 100
    loss_type: str = "l1"


@dataclass
class DataConfig:
    dataset: str = "rain100h"
    data_type: str = "img"  # "img" | "lmdb"
    dataroot_GT: str = ""
    dataroot_LQ: str = ""
    num_workers: int = 4
    use_flip: bool = True
    use_rot: bool = True
    color: str = "RGB"


@dataclass
class ExperimentConfig:
    exp_name: str = "default"
    sde: SDEConfig = field(default_factory=SDEConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output_root: str = "outputs"

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        return cls(
            exp_name=d.get("exp_name", "default"),
            sde=SDEConfig(**d.get("sde", {})),
            network=NetworkConfig(**d.get("network", {})),
            train=TrainConfig(**d.get("train", {})),
            data=DataConfig(**d.get("data", {})),
            output_root=d.get("output_root", "outputs"),
        )

    @classmethod
    def from_json(cls, path: str) -> "ExperimentConfig":
        with open(path, "r") as f:
            d = json.load(f)
        return cls.from_dict(d)

    @property
    def exp_dir(self) -> str:
        return f"{self.output_root}/{self.exp_name}"

    @property
    def log_dir(self) -> str:
        return f"{self.exp_dir}/log"

    @property
    def ckpt_dir(self) -> str:
        return f"{self.exp_dir}/ckpt"

    @property
    def samples_dir(self) -> str:
        return f"{self.exp_dir}/samples"

    @property
    def plt_dir(self) -> str:
        return f"{self.exp_dir}/plt_fig"

    @property
    def test_dir(self) -> str:
        return f"{self.exp_dir}/test"
