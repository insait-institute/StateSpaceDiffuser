import concurrent
from typing import Any, Optional

import cv2
import numpy as np
import torch
from dev_utils.logger import getLogger

from data.dataset_base import TransformsGenerator, WorldDatasetBase
from data.dataset_retro import DatasetOutputFormat, EnvironmentDataset

log = getLogger(__name__)


class MiniGridDataset(WorldDatasetBase):
    """A wrapper around the CSGO DatasetHandler."""

    def __init__(
        self,
        dataset_name: str,
        data_root_dpath: str = "/data",
        split_type: str = "instance",
        split: str = "all",
        seq_len: int = 50,
        seq_step: int = 16,
        image_size: int | tuple[int, int] = 150,
        n_actions: int = 4,
        cache_dpath: str = "cache",
        central_seq_len: Optional[int] = None,
        repeat: int = 1,
    ) -> None:
        super().__init__(n_actions=n_actions)
        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        self.repeat = repeat

        transforms = TransformsGenerator.get_final_transforms(image_size, None)
        dataset_dpath = f"{data_root_dpath}/{dataset_name}/{dataset_name}"

        if central_seq_len is None:
            central_seq_len = seq_len
        self.centered_seq_len = central_seq_len

        self.dataset_handler = EnvironmentDataset(
            dataset_dpath,
            seq_length_input=seq_len - 1,
            seq_step=seq_step,
            split_type=split_type,
            split=split,
            transform=transforms["train"],
            format=DatasetOutputFormat.IVG,
            enable_cache=True,
            cache_dpath=f"{cache_dpath}/{dataset_name}",
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = index % len(self.dataset_handler)
        data = self.dataset_handler.__getitem__(index)

        # Resize input_frames to 30x30 for frames_hq
        if self.centered_seq_len is not None:
            start_idx = (len(data["input_frames"]) - self.centered_seq_len) // 2
            end_idx = start_idx + self.centered_seq_len
            data["input_frames"] = data["input_frames"][start_idx:end_idx]
            data["actions"] = data["actions"][start_idx:end_idx]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            frames = list(
                executor.map(
                    lambda frame: cv2.resize(
                        frame.transpose(1, 2, 0),
                        (30, 30),
                        interpolation=cv2.INTER_AREA,
                    ).transpose(2, 0, 1),
                    data["input_frames"],
                ),
            )

        output = {
            "observations": torch.tensor(np.array(frames)) * 2 - 1,
            "actions": torch.tensor(data["actions"]),
            "extras": {"frames_hq": torch.tensor(data["input_frames"]) * 2 - 1},
        }

        return output

    def __len__(self) -> int:
        return self.dataset_handler.__len__() * self.repeat


class ExtendedMiniGridDataset(MiniGridDataset):
    def __init__(self, seq_len_ext, **kwargs):
        super().__init__(**kwargs)
        assert self.centered_seq_len % 2 == 1, "seq_len must be odd"
        assert (seq_len_ext - self.centered_seq_len) % 2 == 0, "seq_len_ext must be even"
        self.seq_len_ext = seq_len_ext

    def extend(self, data: torch.Tensor, diff):
        left_img = data[[1]]
        right_img = data[[-2]]
        # copy across dim 0 diff/2 times
        left_img = left_img.repeat(diff // 2, *[1] * (left_img.dim() - 1))
        right_img = right_img.repeat(diff // 2, *[1] * (right_img.dim() - 1))
        # concatenate the images by inserting the copies at index 1 and -2
        input_frames = torch.cat(
            [
                data[:1],
                left_img,
                data[1:],
            ],
            dim=0,
        )

        # Insert right padding before the last frame (index -2)
        insert_pos = len(input_frames) - 1  # second-to-last index
        input_frames = torch.cat(
            [
                input_frames[:insert_pos],
                right_img,
                input_frames[insert_pos:],
            ],
            dim=0,
        )

        return input_frames

    def __getitem__(self, index: int) -> dict[str, Any]:
        data = super().__getitem__(index)

        diff = self.seq_len_ext - self.centered_seq_len
        if diff > 0:
            data["observations"] = self.extend(data["observations"], diff)
            data["extras"]["frames_hq"] = self.extend(data["extras"]["frames_hq"], diff)
            data["actions"] = self.extend(data["actions"], diff)

        return data
