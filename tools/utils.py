import os
import random

import numpy as np
import torch
from einops import rearrange
from torch import nn


def fix_seed(seed: int) -> None:
    """
    Args :
        seed : fix the seed
    Function which allows to fix all the seed and get reproducible results
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)



def resize_video(
    video: torch.Tensor,
    target_size: tuple[int, int],
) -> torch.Tensor:
    """
    Resize a video tensor to the target size.

    Args:
        video (torch.Tensor): The input video tensor of shape (B, T, C, H, W).
        target_size (tuple[int, int]): The target size (height, width).

    Returns:
        torch.Tensor: The resized video tensor.
    """
    # Reshape the video tensor to (B * T, C, H, W)
    B, T, C, H, W = video.shape
    video_reshaped = rearrange(video, "b t c h w -> (b t) c h w")

    # Resize the video frames
    resized_video = nn.functional.interpolate(
        video_reshaped,
        size=target_size,
        mode="bicubic",
        align_corners=False,
    )

    # Reshape back to (B, C, T, H', W')
    resized_video = rearrange(
        resized_video,
        "(b t) c h w -> b t c h w",
        b=B,
        t=T,
    )

    return resized_video
