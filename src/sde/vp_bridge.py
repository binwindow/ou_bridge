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
    """VP-bridge: variance-preserving noise schedule.

    Uses log-SNR parameterization with beta_d and beta_min.
    Training: predict x_0 from x_t.
    Inference: Heun ODE sampler.
    """

    def __init__(self, sigma_max=1.0, sigma_min=0.0001, sigma_data=0.5,
                 beta_d=2., beta_min=0.1, num_steps=100, device=None):
        self.sigma_max = sigma_max  # time scalar (not noise std)
        self.sigma_min = sigma_min
        self.sigma_data = sigma_data
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.num_steps = num_steps
        self.device = device

        # Pre-compute terminal values
        self._t = torch.linspace(0, 1, 1000, device=device)  # for schedule building

        self._logsnr_T = _vp_logsnr(torch.tensor(sigma_max), beta_d, beta_min)
        self._logs_T = _vp_logs(torch.tensor(sigma_max), beta_d, beta_min)

    # ---- forward bridge ----

    def _coeffs(self, t):
        """Compute a_t, b_t, std_t at time t."""
        logsnr_t = _vp_logsnr(t, self.beta_d, self.beta_min)
        logsnr_T = _vp_logsnr(torch.tensor(self.sigma_max, device=t.device),
                              self.beta_d, self.beta_min)
        logs_t = _vp_logs(t, self.beta_d, self.beta_min)
        logs_T = _vp_logs(torch.tensor(self.sigma_max, device=t.device),
                          self.beta_d, self.beta_min)

        a_t = (logsnr_T - logsnr_t + logs_t - logs_T).exp()
        b_t = -torch.expm1(logsnr_T - logsnr_t) * logs_t.exp()
        std_t = (-torch.expm1(logsnr_T - logsnr_t)).sqrt() * (logs_t - logsnr_t / 2).exp()
        return a_t, b_t, std_t

    def _bridge_sample(self, x0, xT, t):
        """VP bridge forward sample at time t."""
        a_t, b_t, std_t = self._coeffs(t)
        noise = torch.randn_like(x0)
        return a_t * xT + b_t * x0 + std_t * noise

    # ---- BaseSDE interface ----

    def generate_random_states(self, x0, mu):
        """Sample t from uniform, generate bridge state. mu = x_T = LQ."""
        x0, mu = x0.to(self.device), mu.to(self.device)
        b = x0.shape[0]
        t = torch.rand(b, device=self.device) * (self.sigma_max - self.sigma_min) + self.sigma_min
        t = _append_dims(t, x0.ndim)
        x_t = self._bridge_sample(x0, mu, t)
        return t, x_t

    def compute_loss(self, model, xt, x0, mu, t, loss_fn):
        """Predict x0 from xt. No preconditioning."""
        pred_x0 = model(xt, mu, t.squeeze())
        # Uniform weighting (simple mode)
        return loss_fn(pred_x0, x0)

    def sample(self, model, xt, mu, **kwargs):
        """Heun ODE sampler for VP bridge."""
        # Build sigma schedule
        sigmas = torch.linspace(self.sigma_max, self.sigma_min, self.num_steps + 1,
                                device=xt.device)
        x = xt

        # Helpers
        beta_d, beta_min = self.beta_d, self.beta_min

        def _s(t):
            vp_snr_sqrt_reciprocal = (np.e ** (0.5 * beta_d * t ** 2 + beta_min * t) - 1) ** 0.5
            return (1 + vp_snr_sqrt_reciprocal ** 2) ** -0.5

        for i in range(len(sigmas) - 1):
            sigma_i = sigmas[i].item()
            sigma_next = sigmas[i + 1].item()
            si = _append_dims(torch.tensor(sigma_i, device=x.device), x.ndim)
            s_next = _append_dims(torch.tensor(sigma_next, device=x.device), x.ndim)

            denoised = model(x, mu, torch.tensor([sigma_i], device=x.device))
            d = self._to_d(x, denoised, mu, sigma_i, sigma_next, _s)
            dt = s_next - si

            if sigma_next == 0:
                x = x + d * dt
            else:
                x_2 = x + d * dt
                denoised_2 = model(x_2, mu, torch.tensor([sigma_next], device=x.device))
                d_2 = self._to_d(x_2, denoised_2, mu, sigma_next, sigma_next, _s)
                dt_actual = s_next - si
                d_prime = (d + d_2) / 2
                x = x + d_prime * dt_actual

        return x

    def _to_d(self, x, denoised, x_T, sigma_t, sigma_next, s_fn):
        """VP ODE derivative using the s-function parameterization."""
        t_i = sigma_t.item() if isinstance(sigma_t, torch.Tensor) else sigma_t
        t_next = sigma_next.item() if isinstance(sigma_next, torch.Tensor) else sigma_next

        s_t = s_fn(t_i)
        s_next = s_fn(t_next)

        logsnr_t = _vp_logsnr(torch.tensor(t_i, device=x.device), self.beta_d, self.beta_min)
        logsnr_T = _vp_logsnr(torch.tensor(self.sigma_max, device=x.device),
                              self.beta_d, self.beta_min)
        logs_t = _vp_logs(torch.tensor(t_i, device=x.device), self.beta_d, self.beta_min)
        logs_T = _vp_logs(torch.tensor(self.sigma_max, device=x.device),
                          self.beta_d, self.beta_min)

        a_t = (logsnr_T - logsnr_t + logs_t - logs_T).exp()
        b_t = -torch.expm1(logsnr_T - logsnr_t) * logs_t.exp()
        std_t = (-torch.expm1(logsnr_T - logsnr_t)).sqrt() * (logs_t - logsnr_t / 2).exp()

        mu_t = a_t * x_T + b_t * denoised

        grad_logq = -(x - mu_t) / std_t ** 2 / (-torch.expm1(logsnr_T - logsnr_t))
        grad_logpxTlxt = -(x - torch.exp(logs_t - logs_T) * x_T) / std_t ** 2 / torch.expm1(logsnr_t - logsnr_T)

        ds_dt = s_next - s_t
        d = -grad_logq + grad_logpxTlxt
        return d * ds_dt if isinstance(ds_dt, torch.Tensor) else d * ds_dt
