from pathlib import Path
from typing import Any

import torch

from bench.dataset import WorldDatasetBase
from data.csgo.dataset import (
    CSGOHdf5Dataset,
    DatasetHandler,
    DiamondCSGODataset,
)

N_KEYS = 11  # number of keyboard outputs, w,s,a,d,space,ctrl,shift,1,2,3,r
N_KEYS_WITH_REVERSE = 18  # number of keyboard outputs, w,s,a,d,space,ctrl,shift,1,2,3,r
N_CLICKS = 2  # number of mouse buttons, left, right
N_MOUSE_X = 23
N_MOUSE_Y = 15


class CSGODataset(WorldDatasetBase):
    """A wrapper around the CSGO DatasetHandler."""

    @property
    def n_actions(self) -> int:
        return 51

    def __init__(
        self,
        dataset_name: str,
        data_root_dpath: str = "/data",
        split: str = "train",
        seq_len: int = 50,
        step_size: int = 16,
        enable_high_res: bool = True,
        enable_cache_in_ram: bool = False,
        enable_manager: bool = False,
    ) -> None:
        high_res_dataset = None
        if enable_high_res:
            high_res_dataset = CSGOHdf5Dataset(f"{data_root_dpath}/csgo/full_res")

        directory = f"{data_root_dpath}/csgo/low_res/{split}"
        dataset = DiamondCSGODataset(
            directory=directory,
            dataset_full_res=high_res_dataset,
            name=f"{split}_dataset",
            cache_in_ram=enable_cache_in_ram,
            use_manager=enable_manager,
        )
        dataset.load_from_default_path()

        self.dataset_handler = DatasetHandler(seq_len=seq_len, step_size=step_size, dataset=dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        segment = self.dataset_handler.__getitem__(index)

        output = {
            "observations": segment.obs,
            "actions": segment.act,
            "extras": {"frames_hq": segment.info.get("full_res", None)},
        }
        return output

    def __len__(self) -> int:
        return self.dataset_handler.__len__()


def mirror_actions(x: torch.Tensor) -> torch.Tensor:
    """
    Flip the last (action) dimension of an input tensor.

    Parameters
    ----------
    x : torch.Tensor           # shape (b, t, d) with odd d

    Returns
    -------
    torch.Tensor               # same shape, actions mirrored around the centre
    """
    # Create a reversed index once, on the correct device & dtype
    rev_idx = torch.arange(x.size(-1) - 1, -1, -1, device=x.device, dtype=torch.long)

    # Advanced indexing is faster than torch.flip for large tensors
    return x.index_select(-1, rev_idx).detach()


class MirroredCSGODataset(CSGODataset):
    def __init__(
        self,
        dataset_name: str,
        data_root_dpath: str = "/data",
        split: str = "train",
        seq_len: int = 50,
        step_size: int = 16,
        enable_high_res: bool = True,
        enable_cache_in_ram: bool = False,
        enable_manager: bool = False,
    ) -> None:
        assert seq_len % 2 == 1, "seq_len must be odd for reverse dataset"
        self.half_seq_len = (seq_len + 1) // 2
        super().__init__(
            dataset_name=dataset_name,
            data_root_dpath=data_root_dpath,
            split=split,
            seq_len=self.half_seq_len,
            step_size=step_size,
            enable_high_res=enable_high_res,
            enable_cache_in_ram=enable_cache_in_ram,
            enable_manager=enable_manager,
        )
        self.num_actions = 60

    def get_reverse_action(self, action: torch.Tensor) -> torch.Tensor:
        # action: b, t, 60
        keys = action[:, 0:N_KEYS]
        # pad with 7 zeros to make it 18
        keys = torch.nn.functional.pad(
            keys,
            (0, N_KEYS_WITH_REVERSE - N_KEYS),
            mode="constant",
            value=0,
        )

        move_keys = keys[:, :4]
        reverse_move_keys = torch.roll(move_keys, shifts=2, dims=-1)
        no_move_keys = keys[:, 4:]
        reverse_no_move_keys = mirror_actions(no_move_keys)
        reverse_keys = torch.cat([reverse_move_keys, reverse_no_move_keys], dim=-1)

        l_click = action[:, N_KEYS : N_KEYS + 1]
        r_click = action[:, N_KEYS + 1 : N_KEYS + N_CLICKS]

        # wherever l_click is 1, set reverse_l_click to 1, and l_click to 0
        reverse_l_click = torch.where(l_click == 1, torch.tensor(1), l_click)
        l_click = torch.where(l_click == 1, torch.tensor(0), l_click)
        # wherever r_click is 1, set reverse_r_click to 1, and r_click to 0
        reverse_r_click = torch.where(r_click == 1, torch.tensor(1), r_click)
        r_click = torch.where(r_click == 1, torch.tensor(0), r_click)

        mouse_x = action[:, N_KEYS + N_CLICKS : N_KEYS + N_CLICKS + N_MOUSE_X]
        reverse_mouse_x = mirror_actions(mouse_x)

        mouse_y = action[
            :,
            N_KEYS + N_CLICKS + N_MOUSE_X : N_KEYS + N_CLICKS + N_MOUSE_X + N_MOUSE_Y,
        ]
        reverse_mouse_y = mirror_actions(mouse_y)

        reverse_actions = torch.cat(
            [
                reverse_keys,
                l_click,
                r_click,
                reverse_l_click,
                reverse_r_click,
                reverse_mouse_x,
                reverse_mouse_y,
            ],
            dim=-1,
        )

        return reverse_actions

    def extend_actions(self, actions: torch.Tensor) -> torch.Tensor:
        # action: b, t, 60
        keys = actions[:, 0:N_KEYS]

        # pad with 7 zeros to make it 18
        keys = torch.nn.functional.pad(
            keys,
            (0, N_KEYS_WITH_REVERSE - N_KEYS),
            mode="constant",
            value=0,
        )

        clicks = actions[:, N_KEYS : N_KEYS + N_CLICKS]
        # pad with 2 zero to make it 4
        clicks = torch.nn.functional.pad(
            clicks,
            (0, 2),
            mode="constant",
            value=0,
        )

        return torch.cat([keys, clicks, actions[:, N_KEYS + N_CLICKS :]], dim=-1)

    def __getitem__(self, index: int) -> dict[str, Any]:
        data = super().__getitem__(index)

        observations = data["observations"]

        # duplicate the first half of the sequence in reverse order
        reversed_observations = observations[:-1].flip(dims=[0])
        data["observations"] = torch.cat([observations, reversed_observations], dim=0)
        data["extras"]["frames_hq"] = torch.cat(
            [data["extras"]["frames_hq"], data["extras"]["frames_hq"][:-1].flip(dims=[0])],
            dim=0,
        )

        actions = data["actions"]
        # now create the reverse half of the actions
        reversed_actions = self.get_reverse_action(actions[:-1].flip(dims=[0]))

        # pad the actions to the same length as the observations
        actions = self.extend_actions(actions)

        data["actions"] = torch.cat([actions[:-1], reversed_actions, actions[[-1]]], dim=0)

        return data