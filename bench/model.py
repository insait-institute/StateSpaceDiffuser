from __future__ import annotations

from typing import Any, Optional

import torch
from actions.action_base import BaseActionHandler
from torch import nn


class WorldModelBase(nn.Module):
    """A base model class inheriting from PyTorch's nn.Module.

    'forward(actions, observations)' must be defined by subclasses.

    """

    def __init__(
        self,
        action_handler: BaseActionHandler | None = None,
        enable_imagine: bool = False,
    ) -> None:
        super().__init__()
        self.action_handler = action_handler if action_handler is not None else BaseActionHandler()
        self.enable_imagine = enable_imagine

    def imagine(
        self,
        actions: torch.Tensor,
        observations: torch.Tensor,
        extras: dict | None = None,
    ) -> torch.Tensor:
        """Imagines the next observations given actions and current observations.

        actions:  shape [batch_size, ...] (for example)
        observations: shape [batch_size, ...]
        Return: new observations as a torch.Tensor.
        """
        msg = "WorldModelBase: imagine() not implemented."
        raise NotImplementedError(msg)

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        extras: dict | None = None,
        enable_imagine: bool = True,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of the model.

        actions:  shape [batch_size, ...] (for example)
        observations: shape [batch_size, ...]
        Return: new observations or model outputs as a torch.Tensor.
        """
        if enable_imagine:
            predictions = self.imagine(actions, observations, extras=extras, **kwargs)
            return {"predictions": predictions}

        msg = "WorldModelBase: forward() not implemented."
        raise NotImplementedError(msg)

    @property
    def device(self) -> torch.device:
        """Best-effort device lookup for wrappers without a dedicated attribute."""
        if hasattr(self, "_device"):
            return self._device
        try:
            return next(self.parameters()).device
        except StopIteration:
            for _, buf in self.named_buffers():
                return buf.device
            return torch.device("cpu")

    @device.setter
    def device(self, value: torch.device) -> None:
        self._device = value
