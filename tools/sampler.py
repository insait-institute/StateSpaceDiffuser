import numpy as np
from torch.utils.data.sampler import Sampler


class DeterministicRandomSampler(Sampler):
    """
    A deterministic random sampler that selects a fixed number of samples
    in a reproducible manner using a given seed.
    """

    def __init__(self, dataset_size: int, sample_size: int = 0, seed: int = 42) -> None:
        """
        Args:
            dataset (Dataset): PyTorch dataset.
            sample_size (int, optional): Number of samples to draw. If None, use full dataset.
            seed (int): Seed for reproducibility.
        """
        self.dataset_size = dataset_size
        self.seed = seed
        self.sample_size = sample_size if sample_size != 0 else self.dataset_size

        # Ensure sample size is within dataset size
        if self.sample_size > self.dataset_size:
            msg = f"Sample size {self.sample_size} exceeds dataset size {self.dataset_size}."
            raise ValueError(
                msg,
            )

        self.indices = self._generate_deterministic_indices()

    def _generate_deterministic_indices(self) -> list[int]:
        """
        Generates a fixed random subset of indices using the given seed.
        """
        rng = np.random.default_rng(self.seed)  # NumPy's Generator for better reproducibility
        all_indices = rng.permutation(self.dataset_size)  # Shuffle full dataset indices
        return all_indices[: self.sample_size].tolist()  # Select only the desired number of samples

    def __iter__(self) -> iter:
        """
        Yields the shuffled dataset indices.
        """
        return iter(self.indices)

    def __len__(self) -> int:
        """
        Returns the total number of samples drawn.
        """
        return self.sample_size
