from abc import ABC, abstractmethod

import torch


class MetricAbstract(ABC):
    """Abstract metric interface."""

    @abstractmethod
    def get_score(self, predictions: torch.Tensor, ground_truths: torch.Tensor) -> torch.Tensor:
        """Returns a torch.Tensor representing the metric result."""


class MetricBase(MetricAbstract):
    """Base class for metrics. Users can subclass this for custom metrics."""

    def __init__(self, name: str) -> None:
        """Initializes the metric.

        Args:
            name (str): Name of the metric.

        """
        self.name = name
        self.reset()

    def reset(self) -> None:
        """Resets the metric."""
        self.n_steps = 0
        self.running_mean = None

    def update(self, predictions: torch.Tensor, ground_truths: torch.Tensor) -> torch.Tensor:
        """Updates the metric with new predictions and ground truths.

        Args:
            predictions (torch.Tensor): Predictions from the model.
            ground_truths (torch.Tensor): Ground truths.

        """
        predictions, ground_truths = self._reshape(predictions, ground_truths)
        score = self.get_score(predictions, ground_truths)

        if self.running_mean is None:
            self.running_mean = torch.mean(score, dim=0)
        else:
            self.running_mean = self.running_mean + (
                torch.mean(score, dim=0) - self.running_mean
            ) / (self.n_steps + 1)

        self.n_steps += 1

        return self.running_mean

    def compute(self) -> torch.Tensor:
        """Computes the final metric score.

        Returns:
            torch.Tensor: The final metric score.

        """
        if self.n_steps == 0:
            msg = "No data was provided to the metric."
            raise ValueError(msg)

        if torch.distributed.is_initialized():
            # Synchronize running_score and n_steps across all GPUs
            aggregated_score = self.running_mean.clone()
            torch.distributed.all_reduce(aggregated_score, op=torch.distributed.ReduceOp.SUM)
            aggregated_score /= torch.distributed.get_world_size()
        else:
            aggregated_score = self.running_mean

        return aggregated_score

    def _reshape(self, predictions: torch.Tensor, ground_truths: torch.Tensor) -> None:
        """Reshapes the predictions and ground truths to match.

        Args:
            predictions (torch.Tensor): Predictions from the model.
              Shape: (B, C, H, W) or (B, T, C, H, W).
            ground_truths (torch.Tensor): Ground truths. Shape: (B, C, H, W) or (B, T, C, H, W).

        """
        if predictions.shape != ground_truths.shape:
            msg = (
                f"Predictions and ground truths must have the same shape. "
                f"Got predictions: {predictions.shape}, ground truths: {ground_truths.shape}."
            )
            raise ValueError(msg)

        if predictions.ndim == 4:
            predictions = predictions.unsqueeze(1)
            ground_truths = ground_truths.unsqueeze(1)

        return predictions, ground_truths

    def get_score(self, predictions: torch.Tensor, ground_truths: torch.Tensor) -> torch.Tensor:
        msg = "MetricBase: get_score() not implemented."
        raise NotImplementedError(msg)
