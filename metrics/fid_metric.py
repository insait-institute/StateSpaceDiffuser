from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torchvision.models import Inception_V3_Weights, inception_v3

from bench.metric import MetricBase
from metrics.components.fid import FrechetInceptionDistance


def get_inception_model():
    """
    Get the Inception V3 model for FID calculation.

    Returns:
        torch.nn.Module: The Inception V3 model.
    """
    model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model.eval()
    return model


class FIDMetric(MetricBase):
    def __init__(self, name: str = "FID", device="cpu", fid_iv3=False) -> None:
        """
        Initialize the FID evaluator.

        Args:
            device (torch.device): The device to run computations on.
            fid_iv3 (bool): Whether to use InceptionV3 for FID calculation.
        """
        super().__init__(name)
        if fid_iv3:
            self.fid_iv3 = get_inception_model()
            self.fid_model = FrechetInceptionDistance(feature=self.fid_iv3, normalize=False).to(
                device,
            )
        else:
            self.fid_model = FrechetInceptionDistance(normalize=True).to(device)

    def __resize_image_nearest_neighbor_tensor(self, images_tensor, new_width, new_height):
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

    def update(self, predictions: torch.Tensor, ground_truths: torch.Tensor) -> torch.Tensor:
        """
        Update the FID model with a batch of real and fake videos.

        Args:
            real_videos (torch.Tensor): Real video tensor.
            fake_videos (torch.Tensor): Fake video tensor.
        """
        predictions, ground_truths = self._reshape(predictions, ground_truths)
        assert ground_truths.shape == predictions.shape
        real_videos = rearrange(ground_truths, "b t c h w -> (b t) c h w")
        fake_videos = rearrange(predictions, "b t c h w -> (b t) c h w")
        assert real_videos.shape == fake_videos.shape
        min_size = (299, 299)

        real_videos = self.__resize_image_nearest_neighbor_tensor(
            real_videos,
            min_size[0],
            min_size[1],
        )
        fake_videos = self.__resize_image_nearest_neighbor_tensor(
            fake_videos,
            min_size[0],
            min_size[1],
        )

        self.fid_model.update(real_videos, real=True)
        self.fid_model.update(fake_videos, real=False)

    def compute(self) -> torch.Tensor:
        """
        Compute the FID score.

        Returns:
            torch.Tensor: The FID score.
        """
        fid_score = self.fid_model.compute()
        return fid_score

    def reset(self) -> None:
        """
        Reset the FID model.
        """
        super().reset()
        if hasattr(self, "fid_model"):
            self.fid_model.reset()

    def get_score(self, predictions: torch.Tensor, ground_truths: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "FID metric does not support get_score method. Use update and compute instead.",
        )
