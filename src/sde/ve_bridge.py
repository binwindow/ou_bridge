"""VE (Variance-Exploding) Bridge SDE from DDBM.

Simple mode: no preconditioning, model directly predicts x0.
"""
import math
import torch
import numpy as np

from .base import BaseSDE

# Sigma schedule builder (from Karras et al.)
def _get_sigmas_karras(n, sigma_min, sigma_max, rho=7., device='cpu'):
    ramp = torch.linspace(0, 1, n, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat([sigmas, torch.zeros(1, device=device)])


def _append_dims(x, target_dims):
    """Append trailing dims to x to match target_dims."""
    dims_to_append = target_dims - x.ndim
    return x[(...,) + (None,) * dims_to_append]


class VEBridge(BaseSDE):
    """VE-bridge: variance-exploding noise schedule.

    x_t = (t^2/sigma_max^2)*x_T + (1 - t^2/sigma_max^2)*x_0 + t*sqrt(1 - t^2/sigma_max^2)*noise

    Training: predict x_0 from x_t.
    Inference: Heun ODE sampler.
    """

    def __init__(self, sigma_max=80., sigma_min=0.002, sigma_data=0.5,
                 num_steps=100, rho=7., device=None):
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.sigma_data = sigma_data
        self.num_steps = num_steps
        self.rho = rho
        self.device = device

    # ---- forward bridge ----

    def _bridge_sample(self, x0, xT, t):
        """VE bridge: x_t from x_0 and x_T at time t (sigma)."""
        # t: (B, 1, 1, 1)
        noise = torch.randn_like(x0)
        std_t = t * torch.sqrt(1 - t ** 2 / self.sigma_max ** 2)
        mu_t = t ** 2 / self.sigma_max ** 2 * xT + (1 - t ** 2 / self.sigma_max ** 2) * x0
        return mu_t + std_t * noise

    # ---- BaseSDE interface ----

    def generate_random_states(self, x0, mu):
        """Sample sigma from log-uniform, generate bridge state. mu = x_T = LQ."""
        b = x0.shape[0]
        log_sigma = torch.rand(b, device=x0.device) * (
            math.log(self.sigma_max) - math.log(self.sigma_min)
        ) + math.log(self.sigma_min)
        sigma = log_sigma.exp()
        sigma = _append_dims(sigma, x0.ndim)
        x_t = self._bridge_sample(x0, mu, sigma)
        return sigma, x_t

    def compute_loss(self, model, xt, x0, mu, sigma, loss_fn):
        """Predict x0 from xt, weight by SNR. No preconditioning."""
        pred_x0 = model(xt, mu, sigma.squeeze())
        # SNR weighting
        snr = sigma ** -2
        weights = _append_dims(snr, xt.ndim)
        loss = loss_fn(weights * pred_x0, weights * x0)
        return loss

    def sample(self, model, xt, mu, **kwargs):
        """Heun ODE sampler (Algorithm 1 from Karras et al. 2022)."""
        sigmas = _get_sigmas_karras(self.num_steps, self.sigma_min, self.sigma_max,
                                    self.rho, device=xt.device)
        x = xt
        for i in range(len(sigmas) - 1):
            sigma_i = _append_dims(sigmas[i], x.ndim)
            sigma_next = _append_dims(sigmas[i + 1], x.ndim)

            denoised = model(x, mu, sigmas[i].expand(1))
            d = self._to_d(x, sigma_i, denoised, mu)
            dt = sigma_next - sigma_i

            if sigmas[i + 1] == 0:
                x = x + d * dt
            else:
                x_2 = x + d * dt
                denoised_2 = model(x_2, mu, sigmas[i + 1].expand(1))
                d_2 = self._to_d(x_2, sigma_next, denoised_2, mu)
                d_prime = (d + d_2) / 2
                x = x + d_prime * dt

        return x

    def _to_d(self, x, sigma, denoised, x_T):
        """VE ODE derivative."""
        grad_pxtlx0 = (denoised - x) / (sigma ** 2)
        grad_pxTlxt = (x_T - x) / (self.sigma_max ** 2 - sigma ** 2)
        gt2 = 2 * sigma
        return -0.5 * gt2 * (grad_pxtlx0 - grad_pxTlxt)
