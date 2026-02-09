import torch

from bench.metric import MetricBase


class PSNRMetric(MetricBase):
    """Example of a custom metric that computes the PSNR between predictions and ground truths."""

    def __init__(self, name: str = "PSRN", value_range: tuple[int] = (0, 1)) -> None:
        super().__init__(name)
        self.max_range = value_range[1] - value_range[0]

    def get_score(self, predictions: torch.Tensor, ground_truths: torch.Tensor) -> torch.Tensor:
        """Computes the Peak Signal-to-Noise Ratio (PSNR) between predictions and ground truths.

        Args:
            predictions (torch.Tensor): Predictions from the model.
            ground_truths (torch.Tensor): Ground truths.

        Returns:
            torch.Tensor: PSNR score.

        """
        predictions, ground_truths = self._reshape(predictions, ground_truths)
        mse = torch.mean((ground_truths - predictions) ** 2, dim=(2, 3, 4))
        psnr = 20 * torch.log10(self.max_range**2 / torch.sqrt(mse))
        return psnr
