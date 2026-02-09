from __future__ import annotations

from typing import TYPE_CHECKING

from einops import rearrange
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

from bench.metric import MetricBase

if TYPE_CHECKING:
    import torch


class SSIMMetric(MetricBase):
    def __init__(self, device: torch.device) -> None:
        """
        Initialize the SSIM evaluator.

        Args:
            device (torch.device): The device to run computations on.
            net_type (str): Which network variant to use for SSIM. Defaults to "vgg".
        """
        super().__init__("SSIM")
        self.ssim_estimator = SSIM(data_range=(0.0, 1.0), reduction="elementwise_mean").to(device)

    def get_score(self, predictions: torch.Tensor, ground_truths: torch.Tensor) -> torch.Tensor:
        """
        This metric does not support a direct get_score method.
        Please call update(...) followed by compute() instead.
        """

        if predictions.dim() == 5:
            predictions_img = rearrange(predictions, "b t c h w -> (b t) c h w")
            ground_truths_img = rearrange(ground_truths, "b t c h w -> (b t) c h w")
        else:
            predictions_img = predictions
            ground_truths_img = ground_truths

        return self.ssim_estimator(predictions_img, ground_truths_img)
