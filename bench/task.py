from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from metrics.fid_metric import FIDMetric
from metrics.lpips_metric import LPIPSMetric
from metrics.psnr_metric import PSNRMetric
from metrics.ssim_metric import SSIMMetric

if TYPE_CHECKING:
    import torch

    from bench.metric import MetricBase
    from bench.model import WorldModelBase


class TaskAbstract(ABC):
    @abstractmethod
    def execute(
        self,
        model: WorldModelBase,
        input_batch: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        """
        Implement the logic for one batch of data:
          1) Extract 'actions', 'observations', etc. from 'input_batch'
          2) Call model.forward(actions, observations)
          3) Possibly compute metrics, do post-processing, return results
        """

    @abstractmethod
    def get_results(self) -> dict[str, torch.Tensor]:
        """
        Return the accumulated results of the task.
        """

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the task.
        """


class TaskBase(TaskAbstract):
    def __init__(self, device="cpu", data_range: list[float] | None = None, **kwargs) -> None:
        super().__init__()
        if kwargs:
            unused_args = ", ".join(kwargs.keys())
            print(f"Warning: Unused arguments passed to TaskBase: {unused_args}")
        self.metrics: list[MetricBase] = [
            PSNRMetric(),
            FIDMetric(device=device),
            SSIMMetric(device=device),
            LPIPSMetric(device=device),
        ]
        self.data_range = [-1.0, 1.0] if data_range is None else data_range

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize the input tensor to the range [0, 1].

        Args:
            tensor (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        return ((tensor - self.data_range[0]) / (self.data_range[1] - self.data_range[0])).clamp(0, 1)

    def execute(
        self,
        model: WorldModelBase,
        input_batch: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        msg = "TaskBase: execute() not implemented."
        raise NotImplementedError(msg)

    def get_results(self) -> dict[str, torch.Tensor]:
        accumulated_results = {}
        for metric in self.metrics:
            accumulated_results[metric.name] = metric.compute().detach().cpu().numpy()

        return accumulated_results

    def reset(self) -> None:
        for metric in self.metrics:
            metric.reset()
