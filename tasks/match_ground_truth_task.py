from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import torch
import torchvision.utils as vutils
from dev_utils.logger import getLogger

from bench.task import TaskBase

log = getLogger(__name__)

if TYPE_CHECKING:
    from bench.metric import MetricBase
    from bench.model import WorldModelBase


class MatchGroundTruthTask(TaskBase):
    """Example of a custom task that evaluates a model using a given metric."""

    def __init__(
        self,
        n_steps: int,
        metrics: list[MetricBase] | None = None,
        data_range: list[float] | None = None,
        qualitative_dpath: str | None = None,
        device: str | torch.device = "cpu",
        **kwargs: dict[str, Any],
    ) -> None:
        super().__init__(data_range=data_range, device=device, **kwargs)

        self.n_steps = n_steps
        self.qualitative_dpath = qualitative_dpath
        self.step = 0

        if metrics is not None:
            self.metrics = metrics

        self.metrics_result = {metric.name: [] for metric in self.metrics}

    def _save_qualitative_results(
        self,
        prediction: torch.Tensor,
        expected_obs: torch.Tensor,
    ) -> None:
        """
        Save qualitative results comparing predicted frames with ground truth.

        Args:
            prediction: Model's predicted observations
            input_batch: Original input batch with ground truth observations
            n_steps_cond: Number of conditioning steps
        """
        # Concatenate the t dimension over the w dimension for prediction
        concatenated_prediction = torch.cat(torch.unbind(prediction, dim=1), dim=-1)

        # Denormalize and clamp prediction
        concatenated_prediction = (concatenated_prediction + 1) / 2
        concatenated_prediction = torch.clamp(concatenated_prediction, 0, 1)

        # Concatenate the t dimension over the w dimension for ground truth frames
        concatenated_frames_gt = torch.cat(
            torch.unbind(
                expected_obs,  # [:, n_steps_cond : n_steps_cond + prediction.shape[1]],
                dim=1,
            ),
            dim=-1,
        )

        # Denormalize and clamp ground truth frames
        concatenated_frames_gt = (concatenated_frames_gt + 1) / 2
        concatenated_image = torch.cat(
            [concatenated_frames_gt[0], concatenated_prediction[0]],
            dim=-2,
        )

        # Save the image
        os.makedirs(self.qualitative_dpath, exist_ok=True)
        filename = f"prediction_step_{self.step}.png"
        filepath = os.path.join(self.qualitative_dpath, filename)

        print(f"Saving qualitative results to {filepath}")
        vutils.save_image(concatenated_image, filepath)

    def execute(
        self,
        model: WorldModelBase,
        input_batch: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute the task on the given model and input batch.

        Args:
            model: (MiniGridWorldModelEnv): The world model to evaluate.
            input_batch: (Dict[str, Any]): A batch of data.

        Returns:
            Dict[str, Any]: Contains metric scores and possibly other evaluation results.
        """
        n_steps_cond = input_batch["observations"].shape[1] // 2 + 1
        # input_batch["observations"][:, 4:6] = 0
        # input_batch["extras"]["frames_hq"][:, 4:6] = 0
        output_batch = model(
            observations=input_batch["observations"],
            actions=input_batch["actions"],
            extras=input_batch["extras"],
            enable_imagine=model.enable_imagine,
            n_steps_cond=n_steps_cond,
        )

        # TODO: Change this when we add upsampler
        expected_obs = output_batch["observations"]  # [:, 1:]
        prediction = output_batch["prediction"]  # [:, n_steps_cond - 1 :]
        if self.qualitative_dpath is not None:
            self._save_qualitative_results(prediction, expected_obs)

        expected_obs = output_batch["observations"][:, n_steps_cond:]
        prediction = output_batch["prediction"][:, n_steps_cond:]

        expected_obs = self.normalize(expected_obs)
        prediction = self.normalize(prediction)
        score_dict = {}
        for metric in self.metrics:
            score_dict[metric.name] = metric.update(prediction, expected_obs)

        self.step += 1
        return score_dict
