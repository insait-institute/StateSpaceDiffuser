import torch
from torch import nn

from actions.action_base import BaseActionHandler
from bench.model import WorldModelBase


class CustomModel(WorldModelBase):
    """
    Custom world model that processes sequences of images (BxTxCxHxW).
    """

    def __init__(
        self,
        input_shape=(3, 64, 64),
        num_actions=4,
        sequence_length=10,
        num_steps_conditioning=4,
        action_handler: BaseActionHandler | None = None,
    ) -> None:
        """
        Args:
            input_shape (tuple): (C, H, W) shape of a single observation frame.
            num_actions (int): Number of possible actions.
            sequence_length (int): The expected sequence length T.
        """
        super().__init__(action_handler=action_handler)
        self.sequence_length = sequence_length
        C, H, W = input_shape

        # 2D CNN for feature extraction from image observations
        self.conv = nn.Sequential(
            nn.Conv2d(C, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Flattened size after convolution
        feature_size = 32 * H * W  # Assuming conv does not downscale spatial dimensions

        # GRU to process the temporal dimension T
        self.gru = nn.GRU(input_size=feature_size, hidden_size=256, batch_first=True)

        # Fully connected layer to output next state prediction
        self.fc = nn.Linear(256, C * H * W)  # Predict next frame as an image tensor

        self.num_steps_conditioning = num_steps_conditioning

    def forward(
        self,
        actions: torch.Tensor,
        observations: torch.Tensor,
        extra: dict[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            actions (torch.Tensor): Shape (B, T), representing a sequence of actions.
            observations (torch.Tensor): Shape (B, T, C, H, W), sequence of image observations.

        Returns:
            torch.Tensor: Shape (B, C, H, W), predicted next state (single frame).
        """
        B, T, C, H, W = observations.shape  # Ensure input has correct shape

        # Reshape: (B*T, C, H, W) for CNN processing
        x = observations.view(B * T, C, H, W)  # Collapse batch & sequence for CNN

        # Apply CNN
        x = self.conv(x)  # Shape: (B*T, 32, H, W)

        # Flatten spatial dimensions
        x = x.view(B, T, -1)  # Shape: (B, T, feature_size)

        # Process with GRU (temporal processing)
        _, hidden = self.gru(x)  # hidden: (1, B, 256) → final hidden state per batch

        # Predict next frame from final hidden state
        next_frame = self.fc(hidden.squeeze(0))  # Shape: (B, C*H*W)

        # Reshape back to (B, C, H, W)
        next_frame = next_frame.view(B, C, H, W)
        return next_frame

    def train(**kwargs):
        raise NotImplementedError("Training is not implemented for this model.")

    def evaluate(**kwargs):
        raise NotImplementedError("Evaluation is not implemented for this model.")

    def save(self, fpath: str):
        torch.save(self.state_dict(), fpath)

    def load(self, fpath: str):
        self.load_state_dict(torch.load(fpath))
