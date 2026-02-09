from abc import ABC, abstractmethod

import torch


class WorldDatasetAbstract(ABC):
    """
    Abstract definition of a dataset that returns dictionaries of Tensors,
    e.g. {"observations": ..., "actions": ..., "extras": ...}.
    """

    @abstractmethod
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


class WorldDatasetBase(WorldDatasetAbstract):
    """
    Base class for world datasets. Inherit from this to implement
    your dataset logic.
    """

    def __init__(self, n_actions: int = 4) -> None:
        """__init__ method for CustomDataset.

        Args:
            data_path (str): Path to the dataset.
            sequence_length (int): Length of observation sequences.

        """
        super().__init__()
        self.n_actions = n_actions  # Example number of actions
        self.data = self._load_data()  # Load your dataset here

    def _load_data(self) -> torch.Tensor:
        """A method to load the dataset."""
        return torch.randn(100, 10, 3, 64, 64)  # Example tensor (100 samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Returns a sample from the dataset.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - "observations": A sequence of images.
                - "actions": A sequence of actions.
                - "extras": Optional metadata.

        """
        output = {
            "observations": self.data[index],  # Shape: [sequence_length, 3, 64, 64]
            "actions": torch.randint(0, 4, (10,)),  # Example: 4 discrete actions
            "extras": {"meta": torch.tensor([index])},  # Extra info if needed
        }
        return output

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.data)
