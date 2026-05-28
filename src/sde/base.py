import abc


class BaseSDE(abc.ABC):
    """Abstract interface for SDE-based bridge models.

    Each SDE type (GOUB, VE-bridge, VP-bridge) implements:
      - generate_random_states: sample (timesteps, noisy_states) for training
      - compute_loss: SDE-specific training loss
      - sample: run reverse process for inference
    """

    @abc.abstractmethod
    def generate_random_states(self, x0, mu):
        """Sample random timesteps and forward-diffused states.

        Returns (timesteps, noisy_states).
        """

    @abc.abstractmethod
    def compute_loss(self, model, xt, x0, mu, timesteps, loss_fn):
        """Compute SDE-specific training loss.

        Args:
            model: denoising network, signature model(xt, cond, t)
            xt: noisy states at sampled timesteps
            x0: ground truth (target image)
            mu: condition (source / LQ image)
            timesteps: sampled timesteps
            loss_fn: loss function callable(predict, target) -> tensor

        Returns scalar loss tensor.
        """

    @abc.abstractmethod
    def sample(self, model, xt, mu, **kwargs):
        """Run reverse process to generate output image."""
