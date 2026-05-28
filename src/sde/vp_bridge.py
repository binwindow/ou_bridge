"""VP (Variance-Preserving) Bridge SDE from DDBM.

Simple mode: no preconditioning, model directly predicts x0.
"""
import math
import torch
import numpy as np

from .base import BaseSDE


def _append_dims(x, target_dims):
    dims_to_append = target_dims - x.ndim
    return x[(...,) + (None,) * dims_to_append]


def _vp_logsnr(t, beta_d, beta_min):
    return -torch.log((0.5 * beta_d * (t ** 2) + beta_min * t).exp() - 1)


def _vp_logs(t, beta_d, beta_min):
    return -0.25 * t ** 2 * beta_d - 0.5 * t * beta_min


class VPBridge(BaseSDE):
    """VP-bridge: variance-preserving noise schedule."""

    def __init__(self, sigma_max=1.0, sigma_min=0.0001, sigma_data=0.5,
                 beta_d=2., beta_min=0.1, num_steps=100, device=None):
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.sigma_data = sigma_data
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.num_steps = num_steps
        self.device = device

        # Pre-compute terminal values as floats (deferred tensor creation on first use)
        self._logsnr_T_val = _vp_logsnr(torch.tensor(sigma_max), beta_d, beta_min).item()
        self._logs_T_val = _vp_logs(torch.tensor(sigma_max), beta_d, beta_min).item()

    def _logsnr_T(self, device):
        return torch.tensor(self._logsnr_T_val, device=device)

    def _logs_T(self, device):
        return torch.tensor(self._logs_T_val, device=device)

    # ---- helpers ----

    def _snr_sqrt_reciprocal(self, t):
        return (torch.exp(0.5 * self.beta_d * t ** 2 + self.beta_min * t) - 1) ** 0.5

    def _snr_sqrt_reciprocal_deriv(self, t):
        r = self._snr_sqrt_reciprocal(t)
        return 0.5 * (self.beta_min + self.beta_d * t) * (r + 1 / r)

    # ---- forward bridge ----

    def _coeffs(self, t):
        logsnr_t = _vp_logsnr(t, self.beta_d, self.beta_min)
        logs_t = _vp_logs(t, self.beta_d, self.beta_min)
        logsnr_T = self._logsnr_T(t.device)
        logs_T = self._logs_T(t.device)
        a_t = (logsnr_T - logsnr_t + logs_t - logs_T).exp()
        b_t = -torch.expm1(logsnr_T - logsnr_t) * logs_t.exp()
        std_t = (-torch.expm1(logsnr_T - logsnr_t)).sqrt() * (logs_t - logsnr_t / 2).exp()
        return a_t, b_t, std_t

    def _bridge_sample(self, x0, xT, t):
        a_t, b_t, std_t = self._coeffs(t)
        noise = torch.randn_like(x0)
        return a_t * xT + b_t * x0 + std_t * noise

    # ---- BaseSDE interface ----

    def generate_random_states(self, x0, mu):
        x0, mu = x0.to(self.device), mu.to(self.device)
        b = x0.shape[0]
        t = torch.rand(b, device=self.device) * (self.sigma_max - self.sigma_min) + self.sigma_min
        t = _append_dims(t, x0.ndim)
        x_t = self._bridge_sample(x0, mu, t)
        return t, x_t

    def compute_loss(self, model, xt, x0, mu, t, loss_fn):
        pred_x0 = model(xt, mu, t.squeeze())
        return loss_fn(pred_x0, x0)

    def sample(self, model, xt, mu, **kwargs):
        """Heun ODE sampler, following DDBM sample_heun exactly."""
        sigmas = torch.linspace(self.sigma_max, self.sigma_min, self.num_steps + 1,
                                device=xt.device)
        x = xt

        for i in range(len(sigmas) - 1):
            t_i = sigmas[i]
            t_next = sigmas[i + 1]

            denoised = model(x, mu, t_i.expand(1))
            nfe += 1
            d = self._get_d_vp(x, denoised, mu, t_i)
            dt = t_next - t_i

            if t_next == 0:
                x = x + d * dt
            else:
                x_2 = x + d * dt
                denoised_2 = model(x_2, mu, t_next.expand(1))
                nfe += 1
                d_2 = self._get_d_vp(x_2, denoised_2, mu, t_next)
                x = x + (d + d_2) / 2 * dt

        return x

    def _get_d_vp(self, x, denoised, x_T, t):
        """VP ODE derivative, matching DDBM's get_d_vp."""
        logsnr_t = _vp_logsnr(t, self.beta_d, self.beta_min)
        logs_t = _vp_logs(t, self.beta_d, self.beta_min)
        logsnr_T = self._logsnr_T(x.device)
        logs_T = self._logs_T(x.device)

        a_t = (logsnr_T - logsnr_t + logs_t - logs_T).exp()
        b_t = -torch.expm1(logsnr_T - logsnr_t) * logs_t.exp()
        std_t = (-torch.expm1(logsnr_T - logsnr_t)).sqrt() * (logs_t - logsnr_t / 2).exp()

        mu_t = a_t * x_T + b_t * denoised

        grad_logq = -(x - mu_t) / std_t ** 2 / (-torch.expm1(logsnr_T - logsnr_t))
        grad_logpxTlxt = -(x - torch.exp(logs_t - logs_T) * x_T) / std_t ** 2 / torch.expm1(logsnr_t - logsnr_T)

        sigma_t = self._snr_sqrt_reciprocal(t)
        sigma_t_deriv = self._snr_sqrt_reciprocal_deriv(t)
        s_t = (1 + sigma_t ** 2).rsqrt()
        s_t_deriv = -sigma_t * sigma_t_deriv * s_t ** 3

        f = s_t_deriv * (-logs_t).exp() * x
        gt2 = 2 * logs_t.exp() ** 2 * sigma_t * sigma_t_deriv

        d = f - gt2 * (0.5 * grad_logq - grad_logpxTlxt)
        return d
