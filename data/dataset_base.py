from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
from torchvision import transforms

from bench.dataset import WorldDatasetAbstract, WorldDatasetBase


def fast_pad(image, pad_width):
    """
    Pads the image using explicit pad widths (like np.pad),
    but faster for constant zero-padding.

    Parameters:
        image (np.ndarray): The input image (H, W) or (H, W, C).
        pad_width (tuple): Padding amounts as ((top, bottom), (left, right)) or ((top, bottom), (left, right), (0, 0)).

    Returns:
        np.ndarray: The padded image.
    """
    h, w = image.shape[:2]

    top, bottom = pad_width[0]
    left, right = pad_width[1]
    if image.ndim == 3:
        c = image.shape[2]
        padded = np.zeros((top + h + bottom, left + w + right, c), dtype=image.dtype)
        padded[top : top + h, left : left + w, :] = image
    else:
        padded = np.zeros((top + h + bottom, left + w + right), dtype=image.dtype)
        padded[top : top + h, left : left + w] = image

    return padded


class TransformsGenerator:
    @staticmethod
    def pad_to_match_aspect_ratio(image: np.ndarray, target_size: Tuple[int, int]):
        height, width = image.shape[:2]
        target_width, target_height = target_size

        aspect_ratio = width / height
        target_aspect_ratio = target_width / target_height

        # Determine padding
        if aspect_ratio > target_aspect_ratio:
            # Width is larger than target, pad top and bottom
            new_width = width
            new_height = int(width / target_aspect_ratio)
            top_pad = (new_height - height) // 2
            bottom_pad = new_height - height - top_pad
            pad_width = ((top_pad, bottom_pad), (0, 0), (0, 0))
        else:
            # Height is larger than target, pad left and right
            new_height = height
            new_width = int(height * target_aspect_ratio)
            left_pad = (new_width - width) // 2
            right_pad = new_width - width - left_pad
            pad_width = ((0, 0), (left_pad, right_pad), (0, 0))

        # Pad the image
        if len(image.shape) == 3:
            # Image has channels (e.g., RGB)
            padded_image = fast_pad(image, pad_width)
            # padded_image = np.pad(image, pad_width, mode="constant", constant_values=0)
        else:
            # Grayscale image
            pad_width = pad_width[:2]

            padded_image = fast_pad(image, pad_width)
            # padded_image = np.pad(image, pad_width, mode="constant", constant_values=0)

        return padded_image

    @staticmethod
    def check_and_resize(
        target_crop: None | Iterable[int],
        target_size: None | tuple[int, int],
    ):
        """
        Creates a function that transforms input OpenCV images to the target size
        :param target_crop: [left_index, upper_index, right_index, lower_index] list representing the crop region
        :param target_size: (width, height) tuple representing the target height and width
        :return: function that transforms an OpenCV image to the target size
        """

        # Creates the transformation function
        def transform(image: np.ndarray):
            if target_crop is not None:
                left, upper, right, lower = target_crop
                image = image[upper:lower, left:right]
            if target_size is not None and not all(
                dim == size for dim, size in zip(image.shape[:2], target_size)
            ):
                image = TransformsGenerator.pad_to_match_aspect_ratio(
                    image,
                    target_size,
                )
                image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

            return image

        return transform

    @staticmethod
    def to_float_tensor(tensor):
        return tensor / 1.0

    @staticmethod
    def get_evaluation_transforms_config(config) -> Tuple:
        return TransformsGenerator.get_evaluation_transforms(
            config.data.crop,
            config.observation_space[config.enc_cnn_keys[0]],
        )

    @staticmethod
    def get_evaluation_transforms(crop_size, observation_space) -> Tuple:
        """
        Obtains the transformations to use for the evaluation scripts
        :param config: The evaluation configuration file
        :return: reference_transformation, generated transformation to use for the reference and the generated datasets
        """

        reference_resize_transform = TransformsGenerator.check_and_resize(
            crop_size,
            observation_space,
        )
        generated_resize_transform = TransformsGenerator.check_and_resize(
            crop_size,
            observation_space,
        )

        # Do not normalize data for evaluation
        reference_transform = transforms.Compose(
            [
                reference_resize_transform,
                transforms.ToTensor(),
                TransformsGenerator.to_float_tensor,
            ],
        )
        generated_transform = transforms.Compose(
            [
                generated_resize_transform,
                transforms.ToTensor(),
                TransformsGenerator.to_float_tensor,
            ],
        )

        return reference_transform, generated_transform

    @staticmethod
    def get_final_transforms_config(config) -> Dict[str, transforms.Compose]:
        """
        Obtains the transformations to use for training and evaluation

        :param config: The configuration file
        :type config: Config
        :param device: The device to use for computation
        :type device: torch.device
        :return: A dictionary containing the transformations for different stages
        :rtype: Dict[str, transforms.Compose]
        """

        return TransformsGenerator.get_final_transforms(
            config.observation_space[config.encoder.enc_cnn_keys[0]][1:],
            config.data.crop,
        )

    @staticmethod
    def get_final_transforms(
        observation_space,
        crop_size,
    ) -> Dict[str, transforms.Compose]:
        """
        Obtains the transformations to use for training and evaluation

        :param config: The configuration file
        :type config: Config
        :param device: The device to use for computation
        :type device: torch.device
        :return: A dictionary containing the transformations for different stages
        :rtype: Dict[str, transforms.Compose]
        """
        # resize_transform = TransformsGenerator.check_and_resize(config.data.crop,
        #                                                         config.observation_space[config.encoder.enc_cnn_keys[0]][1:])

        resize_transform = TransformsGenerator.check_and_resize(
            crop_size,
            observation_space,
        )
        transform = transforms.Compose(
            [
                resize_transform,
                transforms.ToTensor(),
                # TransformsGenerator.to_float_tensor,
                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ],
        )

        return {
            "train": transform,
            "validation": transform,
            "test": transform,
        }