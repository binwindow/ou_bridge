import copy
import torch
import torch.nn as nn

from .modules import ConditionalUNet, MatchingLoss


class EMA:
    """Hand-written EMA, no external library dependency."""

    def __init__(self, model: nn.Module, beta: float = 0.999):
        self.beta = beta
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    def update(self, model: nn.Module):
        with torch.no_grad():
            for ema_p, model_p in zip(self.ema_model.parameters(), model.parameters()):
                ema_p.data.lerp_(model_p.data, 1.0 - self.beta)

    def state_dict(self) -> dict:
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict: dict):
        self.ema_model.load_state_dict(state_dict)

    def to(self, device):
        self.ema_model = self.ema_model.to(device)
        return self

    def train(self):
        self.ema_model.train()

    def eval(self):
        self.ema_model.eval()


class DenoisingModel:
    """Wraps ConditionalUNet + EMA. Manages feed_data, optimize_parameters, test."""

    def __init__(self, config, device: torch.device):
        self.device = device

        net_cfg = config.network
        self.model = ConditionalUNet(
            in_nc=net_cfg.in_nc,
            out_nc=net_cfg.out_nc,
            nf=net_cfg.nf,
            depth=net_cfg.depth,
        ).to(device)
        self.model.train()

        # EMA
        self.ema = EMA(self.model, beta=config.train.ema_beta)
        self.ema.to(device)

        # Loss
        self.loss_fn = MatchingLoss(loss_type=config.train.loss_type, is_weighted=False).to(device)

        # State buffers
        self.state = None
        self.condition = None
        self.state_0 = None
        self.output = None

    def feed_data(self, state, LQ, GT=None):
        self.state = state.to(self.device)
        self.condition = LQ.to(self.device)
        if GT is not None:
            self.state_0 = GT.to(self.device)

    def optimize_parameters(self, step: int, timesteps, sde):
        """Single training step. sde is the GOUB instance (pure functional)."""
        timesteps = timesteps.to(self.device)

        # Predict noise using the model (model signature: model(xt, cond, t))
        noise = self.model(self.state, self.condition, timesteps.squeeze())
        score = sde.get_score_from_noise(noise, timesteps)

        # Maximum likelihood: match predicted reverse step to optimum reverse step
        xt_1_expectation = sde.reverse_sde_step_mean(self.state, self.condition, score, timesteps)
        xt_1_optimum = sde.reverse_optimum_step(self.state, self.state_0, self.condition, timesteps)
        loss = self.loss_fn(xt_1_expectation, xt_1_optimum)

        return loss

    def test(self, sde, use_ema=False):
        """Run reverse SDE for validation. Returns output tensor."""
        model = self.model
        if use_ema:
            model = self.ema.ema_model
        model.eval()
        with torch.no_grad():
            output = sde.reverse_sde(self.state, self.condition, model, save_states=False)
        model.train()
        self.output = output
        return output

    def get_parameter_info(self) -> dict:
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            "architecture": "ConditionalUNet",
            "total_params": total,
            "trainable_params": trainable,
        }
