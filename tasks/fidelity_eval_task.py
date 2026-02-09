from __future__ import annotations

from typing import TYPE_CHECKING, Any

from bench.task import TaskBase
from metrics.psnr_metric import PSNRMetric

if TYPE_CHECKING:
    from bench.metric import MetricBase
    from bench.model import WorldModelBase


class FidelityEvalTask(TaskBase):
    """Example of a custom task that evaluates a model using a given metric."""

    def __init__(
        self,
        n_steps: int,
        metrics: list[MetricBase] | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        super().__init__(metrics, **kwargs)

        self.n_steps = n_steps

        if metrics is None:
            self.metrics = [PSNRMetric()]

        self.metrics_result = {metric.name: [] for metric in self.metrics}

    def execute(
        self,
        model: WorldModelBase,
        input_batch: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute the task on the given model and input batch.

        Args:
            model: (WorldModelEnv): The world model to evaluate.
            input_batch: (Dict[str, Any]): A batch of data.

        Returns:
            Dict[str, Any]: Contains metric scores and possibly other evaluation results.
        """
        n_steps_cond = model.num_steps_conditioning

        # Reset each metric
        for metric in self.metrics:
            metric.reset()

        actions = input_batch["actions"]

        # Model prediction: shape [B, (n_steps + 1), C, H, W]
        prediction = model(
            observations=input_batch["extras"]["frames_hq"][:, :n_steps_cond],
            actions=actions,
            frames_lq=input_batch["observations"][:, :n_steps_cond],
        )

        expected_obs = input_batch["observations"][:, n_steps_cond:]

        score_dict = {}
        for metric in self.metrics:
            score_dict[metric.name] = metric.update(prediction, expected_obs)

        return score_dict
