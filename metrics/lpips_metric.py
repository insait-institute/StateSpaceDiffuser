from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from bench.metric import MetricBase


class LPIPSMetric(MetricBase):
    def __init__(self, device: torch.device, net_type: str = "vgg") -> None:
        """
        Initialize the LPIPS evaluator.

        Args:
            device (torch.device): The device to run computations on.
            net_type (str): Which network variant to use for LPIPS. Defaults to "vgg".
        """
        super().__init__("LPIPS")
        self.lpips_model = LearnedPerceptualImagePatchSimilarity(
            net_type=net_type,
            normalize=True,
        ).to(device)

    def __resize_image_nearest_neighbor_tensor(
        self,
        images_tensor: torch.Tensor,
        new_width: int,
        new_height: int,
    ) -> torch.Tensor:
        """
        Resize the input tensor using nearest neighbor interpolation.

        Args:
            images_tensor (torch.Tensor): Input tensor of shape (B, C, H, W).
            new_width (int): Target width.
            new_height (int): Target height.

        Returns:
            torch.Tensor: Resized tensor.
        """
        resized_tensor = F.interpolate(images_tensor, size=(new_height, new_width), mode="nearest")
        return resized_tensor

    def update(self, predictions: torch.Tensor, ground_truths: torch.Tensor) -> None:
        """
        Update the LPIPS model with a batch of prediction and ground truth videos.

        Args:
            predictions (torch.Tensor): Predicted video tensor of shape (B, C, T, H, W).
            ground_truths (torch.Tensor): Ground truth video tensor of shape (B, C, T, H, W).
        """

        # Reshape and rearrange to (B*T, C, H, W)
        if predictions.dim() == 5:
            predictions_img = rearrange(predictions, "b t c h w -> (b t) c h w")
            ground_truths_img = rearrange(ground_truths, "b t c h w -> (b t) c h w")
        else:
            predictions_img = predictions
            ground_truths_img = ground_truths

        # (Optional) resize to a fixed size if desired, e.g. 256 x 256
        # predictions = self.__resize_image_nearest_neighbor_tensor(predictions, 256, 256)
        # ground_truths = self.__resize_image_nearest_neighbor_tensor(ground_truths, 256, 256)

        # Update LPIPS metric
        self.lpips_model.update(predictions_img, ground_truths_img)

    def compute(self) -> torch.Tensor:
        """
        Compute the accumulated LPIPS score.

        Returns:
            float: The LPIPS score.
        """
        lpips_score = self.lpips_model.compute()
        return lpips_score

    def reset(self) -> None:
        """
        Reset the LPIPS model.
        """
        super().reset()
        if hasattr(self, "lpips_model"):
            self.lpips_model.reset()

    def get_score(self, predictions: torch.Tensor, ground_truths: torch.Tensor) -> torch.Tensor:
        """
        This metric does not support a direct get_score method.
        Please call update(...) followed by compute() instead.
        """
        raise NotImplementedError(
            "LPIPS metric does not support get_score method. Use update and compute instead.",
        )
