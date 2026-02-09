from bench.metric import MetricBase
from bench.task import TaskBase


class CustomTask(TaskBase):
    """
    Example of a custom task that evaluates a model using a given metric.
    """

    def __init__(self, metric: MetricBase):
        """
        Args:
            metric (MetricBase): The metric to evaluate model performance.
        """
        super().__init__()
        self.metric = metric  # Assign a metric to the task

    def execute(self, model, input_batch):
        """
        Args:
            model (WorldModelBase): The world model to evaluate.
            input_batch (Dict[str, torch.Tensor]): A batch of data.

        Returns:
            Dict[str, Any]: Contains metric scores and possibly other evaluation results.
        """
        observations = input_batch["observations"]  # Shape: [batch_size, sequence_length, C, H, W]
        actions = input_batch["actions"]  # Shape: [batch_size, sequence_length]

        # Get predictions for the last time step
        predicted_next_state = model.forward(actions[:, -1], observations[:, -1])

        # Compute metric score
        score = self.metric.compute_score(predicted_next_state, observations[:, -1])

        return {"metric_score": score}
