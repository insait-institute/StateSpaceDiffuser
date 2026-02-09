import torch
import torch.nn.functional as F
from einops import rearrange
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchvision.models import Inception_V3_Weights, inception_v3

from .quality_metrics.fid import FrechetInceptionDistance


class FIDEvaluator:
    """
    Evaluator for Fréchet Inception Distance (FID) metric.
    """

    def __init__(self, device, fid_iv3=False) -> None:
        """
        Initialize the FID evaluator.

        Args:
            device (torch.device): The device to run computations on.
            fid_iv3 (bool): Whether to use InceptionV3 for FID calculation.
        """
        if fid_iv3:
            self.fid_iv3 = get_inception_model()
            self.fid_model = FrechetInceptionDistance(feature=self.fid_iv3, normalize=False).to(
                device
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

    def update_batch(self, real_videos, fake_videos):
        """
        Update the FID model with a batch of real and fake videos.

        Args:
            real_videos (torch.Tensor): Real video tensor.
            fake_videos (torch.Tensor): Fake video tensor.
        """
        assert real_videos.shape == fake_videos.shape
        real_videos = rearrange(real_videos, "b t c h w -> (b t) c h w")
        fake_videos = rearrange(fake_videos, "b t c h w -> (b t) c h w")
        assert real_videos.shape == fake_videos.shape
        min_size = (299, 299)

        real_videos = self.__resize_image_nearest_neighbor_tensor(
            real_videos, min_size[0], min_size[1]
        )
        fake_videos = self.__resize_image_nearest_neighbor_tensor(
            fake_videos, min_size[0], min_size[1]
        )

        self.fid_model.update(real_videos, real=True)
        self.fid_model.update(fake_videos, real=False)

    def fid(self):
        """
        Compute the FID score.

        Returns:
            float: The FID score.
        """
        fid_score = self.fid_model.compute().item()
        return fid_score

    def reset(self):
        """
        Reset the FID model.
        """
        self.fid_model.reset()


class FidelityEvaluator:
    """
    Evaluator for video fidelity metrics (PSNR and SSIM).
    """

    def __init__(self, device, method="videogpt"):
        """
        Initialize the Fidelity evaluator.

        Args:
            device (torch.device): The device to run computations on.
            method (str): The method for evaluation (default is "videogpt").
        """
        self.device = device
        self.ssim_estimator = SSIM(data_range=1.0, reduction="elementwise_mean").to(device)

    def psnr(self, real_videos, fake_videos):
        """
        Compute the Peak Signal-to-Noise Ratio (PSNR) between real and fake videos.

        Args:
            real_videos (torch.Tensor): Real video tensor.
            fake_videos (torch.Tensor): Fake video tensor.

        Returns:
            float: The PSNR value.
        """
        # change dim from (1, 3, 4) to (2, 3, 4), because the shape of
        # real_videos and fake_videos is (batch, frame, channel, height, width)
        mse = torch.mean((real_videos - fake_videos) ** 2, dim=(2, 3, 4))
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        psnr_mean = torch.mean(psnr)
        return psnr_mean.item()

    def ssim(self, real_videos, fake_videos):
        """
        Compute the Structural Similarity Index (SSIM) between real and fake videos.

        Args:
            real_videos (torch.Tensor): Real video tensor.
            fake_videos (torch.Tensor): Fake video tensor.

        Returns:
            float: The SSIM value.
        """
        ssim = self.ssim_estimator(fake_videos, real_videos).item()
        return ssim


class ControlabilityEvaluator:
    """
    Evaluator for video controlability metrics.
    """

    def __init__(self, device) -> None:
        """
        Initialize the Controlability evaluator.

        Args:
            device (torch.device): The device to run computations on.
        """
        self.device = device

    def psnr(self, real_videos, fake_videos):
        """
        Compute the Peak Signal-to-Noise Ratio (PSNR) between real and fake videos.

        Args:
            real_videos (torch.Tensor): Real video tensor.
            fake_videos (torch.Tensor): Fake video tensor.

        Returns:
            float: The PSNR value.
        """
        # change dim from (1, 3, 4) to (2, 3, 4), because the shape of
        # real_videos and fake_videos is (batch, frame, channel, height, width)
        mse = torch.mean((real_videos - fake_videos) ** 2, dim=(2, 3, 4))
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        psnr_mean = torch.mean(psnr)
        return psnr_mean.item()

    def delta_psnr(self, real_videos, fake_videos1, fake_videos2):
        """
        Compute the difference in PSNR between two sets of fake videos.

        Args:
            real_videos (torch.Tensor): Real video tensor.
            fake_videos1 (torch.Tensor): First set of fake video tensor.
            fake_videos2 (torch.Tensor): Second set of fake video tensor.

        Returns:
            float: The difference in PSNR values.
        """
        psnr1 = self.psnr(real_videos, fake_videos1)
        psnr2 = self.psnr(real_videos, fake_videos2)
        return psnr1 - psnr2


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


class Evaluator:
    """
    Main evaluator class that combines multiple evaluation metrics.
    """

    def __init__(
        self,
        device,
        eval_fid=True,
        eval_fidelity=True,
        eval_controlability=True,
        fid_iv3=False,
    ):
        """
        Initialize the main Evaluator.

        Args:
            device (torch.device): The device to run computations on.
            eval_fid (bool): Whether to evaluate FID.
            eval_fidelity (bool): Whether to evaluate fidelity metrics.
            eval_controlability (bool): Whether to evaluate controlability metrics.
            fid_iv3 (bool): Whether to use InceptionV3 for FID calculation.
        """
        self.device = device

        if eval_fid:
            self.fid_evaluator = FIDEvaluator(device, fid_iv3)
        else:
            self.fid_evaluator = None

        if eval_fidelity:
            self.fidelity_evaluator = FidelityEvaluator(device)
        else:
            self.fidelity_evaluator = None

        if eval_controlability:
            self.controlability_evaluator = ControlabilityEvaluator(device)
        else:
            self.controlability_evaluator = None

    def fid_update_batch(self, real_videos, fake_videos):
        """
        Update the FID evaluator with a batch of real and fake videos.

        Args:
            real_videos (torch.Tensor): Real video tensor.
            fake_videos (torch.Tensor): Fake video tensor.
        """
        assert self.fid_evaluator is not None, "FID evaluator is not enabled"
        assert real_videos.shape == fake_videos.shape
        self.fid_evaluator.update_batch(real_videos, fake_videos)

    def fid(self):
        """
        Compute the FID score.

        Returns:
            float: The FID score.
        """
        assert self.fid_evaluator is not None, "FID evaluator is not enabled"
        return self.fid_evaluator.fid()

    def reset(self):
        """
        Reset all enabled evaluators.
        """
        if self.fid_evaluator is not None:
            self.fid_evaluator.reset()

    def psnr(self, real_videos, fake_videos):
        """
        Compute the PSNR between real and fake videos.

        Args:
            real_videos (torch.Tensor): Real video tensor.
            fake_videos (torch.Tensor): Fake video tensor.

        Returns:
            float: The PSNR value.
        """
        assert self.fidelity_evaluator is not None, "Fidelity evaluator is not enabled"
        return self.fidelity_evaluator.psnr(real_videos, fake_videos)

    def ssim(self, real_videos, fake_videos):
        """
        Compute the SSIM between real and fake videos.

        Args:
            real_videos (torch.Tensor): Real video tensor.
            fake_videos (torch.Tensor): Fake video tensor.

        Returns:
            float: The SSIM value.
        """
        assert self.fidelity_evaluator is not None, "Fidelity evaluator is not enabled"
        return self.fidelity_evaluator.ssim(real_videos, fake_videos)

    def delta_psnr(self, real_videos, fake_videos1, fake_videos2):
        """
        Compute the difference in PSNR between two sets of fake videos.

        Args:
            real_videos (torch.Tensor): Real video tensor.
            fake_videos1 (torch.Tensor): First set of fake video tensor.
            fake_videos2 (torch.Tensor): Second set of fake video tensor.

        Returns:
            float: The difference in PSNR values.
        """
        assert self.controlability_evaluator is not None, "Controlability evaluator is not enabled"
        return self.controlability_evaluator.delta_psnr(real_videos, fake_videos1, fake_videos2)


if __name__ == "__main__":
    fid_evaluator = FIDEvaluator("cpu")

    real_videos = torch.ones(10, 3, 10, 64, 64)
    fake_videos = torch.zeros(10, 3, 10, 64, 64)

    fid_evaluator.update_batch(real_videos, fake_videos)
    print("FID different: ", fid_evaluator.fid())
    fid_evaluator.reset()
    fid_evaluator.update_batch(real_videos, real_videos)
    print("FID same: ", fid_evaluator.fid())

    print("Fidelity Evaluator")
    fidelity_evaluator = FidelityEvaluator("cpu")
    print("PSNR: ", fidelity_evaluator.psnr(real_videos, fake_videos))
    print("SSIM: ", fidelity_evaluator.ssim(real_videos, fake_videos))
