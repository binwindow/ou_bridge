"""GOUB (Generalized Ornstein-Uhlenbeck Bridge) SDE.

Pure functional: mu and model passed explicitly."""
import math
import torch
import os

from .base import BaseSDE


def _constant_theta_schedule(timesteps, v=1.):
    timesteps = timesteps + 1
    return torch.ones(timesteps, dtype=torch.float32)


def _linear_theta_schedule(timesteps):
    timesteps = timesteps + 1
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def _cosine_theta_schedule(timesteps, s=0.008):
    timesteps = timesteps + 2
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:-1]
    return betas


class GOUB(BaseSDE):
    def __init__(self, lambda_square=30, T=100, schedule='cosine', eps=0.005, device=None):
        self.T = T
        self.device = device
        self.lambda_square = lambda_square / 255 if lambda_square >= 1 else lambda_square
        self.schedule = schedule
        self._initialize()

    def _initialize(self):
        L, T = self.lambda_square, self.T
        eps = getattr(self, '_eps', 0.005)
        dt = 1 / T

        if self.schedule == 'cosine':
            thetas = _cosine_theta_schedule(T)
        elif self.schedule == 'linear':
            thetas = _linear_theta_schedule(T)
        elif self.schedule == 'constant':
            thetas = _constant_theta_schedule(T)
        else:
            raise ValueError(f'Unknown schedule: {self.schedule}')

        sigmas = torch.sqrt(L ** 2 * 2 * thetas)
        thetas_cumsum = torch.cumsum(thetas, dim=0) - thetas[0]
        self.dt = -1 / thetas_cumsum[-1] * math.log(eps)
        sigma_bars = torch.sqrt(L ** 2 * (1 - torch.exp(-2 * thetas_cumsum * self.dt)))

        self.thetas = thetas.to(self.device)
        self.sigmas = sigmas.to(self.device)
        self.thetas_cumsum = thetas_cumsum.to(self.device)
        self.sigma_bars = sigma_bars.to(self.device)

        self.sigma_t_T = torch.sqrt(
            self.lambda_square ** 2 * (1 - torch.exp(-2 * (self.thetas_cumsum[-1] - self.thetas_cumsum) * self.dt)))
        self.f_sigmas = self.sigma_bars * self.sigma_t_T / self.sigma_bars[-1]

    # ---- marginal process coefficients ----

    def m(self, t):
        return torch.exp(-self.thetas_cumsum[t] * self.dt) * self.sigma_t_T[t] ** 2 / self.sigma_bars[-1] ** 2

    def n(self, t):
        return ((1 - torch.exp(-self.thetas_cumsum[t] * self.dt)) * self.sigma_t_T[t] ** 2 +
                torch.exp(-2 * (self.thetas_cumsum[-1] - self.thetas_cumsum[t]) * self.dt)
                * self.sigma_bars[t] ** 2) / self.sigma_bars[-1] ** 2

    def f_mean(self, x0, mu, t):
        return self.m(t) * x0 + self.n(t) * mu

    def f_sigma(self, t):
        return self.f_sigmas[t]

    # ---- SDE components ----

    def _mask_t100(self, t, tensor):
        """Zero out tensor where t == 100 (last timestep edge case)."""
        mask = (t == 100)
        if isinstance(mask, bool):
            if mask:
                return torch.zeros_like(tensor)
            return tensor
        if mask.any():
            mask_expanded = mask.expand_as(tensor)
            tensor = tensor.clone()
            tensor[mask_expanded] = 0
        return tensor

    def sde_reverse_drift(self, x, mu, score, t):
        tmp = torch.exp(2 * (self.thetas_cumsum[t] - self.thetas_cumsum[-1]) * self.dt)
        drift_h = - self.sigmas[t] ** 2 * tmp / self.sigma_t_T[t] ** 2 * (x - mu)
        drift_h = self._mask_t100(t, drift_h)
        return (self.thetas[t] * (mu - x) + drift_h - self.sigmas[t] ** 2 * score) * self.dt

    def dispersion(self, x, t):
        return self.sigmas[t] * (torch.randn_like(x) * math.sqrt(self.dt)).to(self.device)

    def get_score_from_noise(self, noise, t):
        return -noise / self.f_sigma(t)

    def reverse_sde_step_mean(self, x, mu, score, t):
        return x - self.sde_reverse_drift(x, mu, score, t)

    def reverse_sde_step(self, x, mu, score, t):
        return x - self.sde_reverse_drift(x, mu, score, t) - self.dispersion(x, t)

    def reverse_optimum_step(self, xt, x0, mu, t):
        return self._r_mean_1(xt, x0, mu, t)

    def _r_mean_1(self, xt, x0, mu, t):
        f_s = self.f_sigma(t)
        f_sm1 = self.f_sigma(t - 1)
        f_m = self._f_m(t)
        f_n = self._f_n(t)
        f_s1 = self._f_sigma_1(t)
        return (f_sm1 ** 2 * f_m * (xt - f_n * mu) +
                f_s1 ** 2 * self.f_mean(x0, mu, t - 1)) / f_s ** 2

    def _f_m(self, t):
        return self.m(t) / self.m(t - 1)

    def _f_n(self, t):
        return self.n(t) - self.n(t - 1) * self.m(t) / self.m(t - 1)

    def _f_sigma_1(self, t):
        return torch.sqrt(self.f_sigma(t) ** 2 - self.f_sigma(t - 1) ** 2 * self._f_m(t) ** 2)

    # ---- BaseSDE interface ----

    def generate_random_states(self, x0, mu):
        x0, mu = x0.to(self.device), mu.to(self.device)
        batch = x0.shape[0]
        timesteps = torch.randint(1, self.T, (batch, 1, 1, 1)).long()
        state_mean = self.f_mean(x0, mu, timesteps)
        noises = torch.randn_like(state_mean)
        noisy_states = noises * self.f_sigma(timesteps) + state_mean
        return timesteps, noisy_states.to(torch.float32)

    def compute_loss(self, model, xt, x0, mu, timesteps, loss_fn):
        timesteps = timesteps.to(self.device)
        noise = model(xt, mu, timesteps.squeeze())
        score = self.get_score_from_noise(noise, timesteps)
        expected = self.reverse_sde_step_mean(xt, mu, score, timesteps)
        optimal = self.reverse_optimum_step(xt, x0, mu, timesteps)
        return loss_fn(expected, optimal)

    def sample(self, model, xt, mu, **kwargs):
        """Run reverse SDE to generate derained image."""
        x = xt.clone()
        for t in reversed(range(1, self.T + 1)):
            noise = model(x, mu, t)
            score = -noise / self.f_sigma(t) if t != 100 else 0
            x = self.reverse_sde_step(x, mu, score, t)
        return x
