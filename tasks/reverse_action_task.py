from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import torch
import torchvision.utils as vutils
from einops import repeat

from bench.task import TaskBase

if TYPE_CHECKING:
    from actions.action_base import BaseActionEnum
    from bench.metric import MetricBase
    from bench.model import WorldModelBase

from dev_utils.logger import getLogger

log = getLogger(__name__)

import lovely_tensors as lt


class ReverseActionTask(TaskBase):
    def __init__(
        self,
        action_enum: BaseActionEnum,
        n_steps: int,
        metrics: list[MetricBase] | None = None,
        data_range: list[float] = None,
        qualitative_dpath: str | None = None,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        super().__init__(data_range=data_range, device=device, **kwargs)

        if n_steps % 2 != 0:
            msg = "Number of steps must be even."
            raise ValueError(msg)

        self.action_enum = action_enum

        self.n_steps = n_steps

        if metrics is not None:
            self.metrics = metrics

        self.metrics_result = {metric.name: [] for metric in self.metrics}
        self.qualitative_dpath = qualitative_dpath
        self.step = 0
        self._warned_imagine_mismatch = False

    def execute(
        self,
        model: WorldModelBase,
        input_batch: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute the task on the given model and input batch.

        Args:
            model: (DiamondWorldModelEnv): The world model to evaluate.
            input_batch: (Dict[str, torch.Tensor]): A batch of data.

        Returns:
            Dict[str, Any]: Contains metric scores and possibly other evaluation results.

        """
        n_steps_cond = model.n_steps_cond

        if (
            not self._warned_imagine_mismatch
            and getattr(model, "enable_imagine", True) is False
        ):
            log.w(
                "ReverseActionTask forces enable_imagine=True; "
                "model.enable_imagine is False and will be ignored."
            )
            self._warned_imagine_mismatch = True

        forward_action = model.action_handler.encode_action(self.action_enum)
        reverse_action_enum = model.action_handler.get_reverse_action(self.action_enum)
        reverse_action = model.action_handler.encode_action(reverse_action_enum)

        b = input_batch["observations"].shape[0]
        actions_init = input_batch["actions"][:, :n_steps_cond]
        actions_pred_forward = repeat(
            forward_action.to(model.device),
            "d -> b t d",
            b=b,
            t=self.n_steps // 2,
        )
        actions_pred_backward = repeat(
            reverse_action.to(model.device),
            "d -> b t d",
            b=b,
            t=self.n_steps // 2,
        )
        # actions = torch.cat([actions_init, actions_pred_forward, actions_pred_backward], dim=1)
        # pad actions init to the same size as actions_pred_forward and actions_pred_backward
        actions_init = torch.nn.functional.pad(
            actions_init,
            (0, actions_pred_forward.shape[2] - actions_init.shape[2]),
            mode="constant",
            value=0,
        )

        actions = torch.cat([actions_init, actions_pred_forward, actions_pred_backward], dim=1)

        log.t(
            lt.lovely(actions),
        )

        output_batch = model(
            observations=input_batch["observations"],
            actions=actions,
            extras=input_batch["extras"],
            enable_imagine=True,
        )

        prediction = output_batch["prediction"]

        first_frame = output_batch["extras"]["frames_hq"][:, n_steps_cond - 1]
        final_frame = prediction[:, -1]

        final_frame = (final_frame + 1) / 2
        final_frame = torch.clamp(final_frame, 0, 1)

        first_frame = (first_frame + 1) / 2
        first_frame = torch.clamp(first_frame, 0, 1)

        # log.t(
        #     lt.lovely(first_frame),
        #     lt.lovely(final_frame),
        #     lt.lovely(prediction),
        # )

        if self.qualitative_dpath is not None:
            # Concatenate the t dimension over the w dimension for prediction
            concatenated_prediction = torch.cat(torch.unbind(prediction, dim=1), dim=-1)

            # Denormalize and clamp prediction
            concatenated_prediction = (concatenated_prediction + 1) / 2
            concatenated_prediction = torch.clamp(concatenated_prediction, 0, 1)

            # Concatenate the t dimension over the w dimension for frames_hq
            concatenated_frames_hq = torch.cat(
                torch.unbind(output_batch["extras"]["frames_hq"][:, : prediction.shape[1]], dim=1),
                dim=-1,
            )

            # Denormalize and clamp frames_hq
            concatenated_frames_hq = (concatenated_frames_hq + 1) / 2
            concatenated_frames_hq = torch.clamp(concatenated_frames_hq, 0, 1)

            # Concatenate prediction and frames_hq over the height dimension
            concatenated_image = torch.cat(
                [concatenated_frames_hq, concatenated_prediction],
                dim=-2,
            )[0]

            # Save the image
            os.makedirs(self.qualitative_dpath, exist_ok=True)
            # Generate a filename based on the current step or other criteria
            filename = f"prediction_step_{self.step}.png"
            self.step += 1
            filepath = os.path.join(self.qualitative_dpath, filename)

            print(f"Saving qualitative results to {filepath}")
            # Save the image
            vutils.save_image(concatenated_image, filepath)

        final_frame = self.normalize(final_frame)
        first_frame = self.normalize(first_frame)

        score_dict = {}
        for metric in self.metrics:
            score_dict[metric.name] = metric.update(final_frame, first_frame)

        return score_dict
