from __future__ import annotations

from typing import TYPE_CHECKING, Any

import hydra
import torch
from accelerate import Accelerator
from dev_utils.logger import getLogger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from tools.sampler import DeterministicRandomSampler

if TYPE_CHECKING:
    from bench.dataset import WorldDatasetBase
    from bench.model import WorldModelBase
    from bench.task import TaskBase

import os
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from torchvision.utils import make_grid
def save_image_sequence(tensor: torch.Tensor, save_dir: str, index, prefix: str = "frame", format: str = "png"):
    """
    Saves a sequence of images stored in a (b, c, h, w) GPU tensor to disk.

    Args:
        tensor (torch.Tensor): A 4D tensor of shape (b, c, h, w) on GPU.
        save_dir (str): Directory to save the images.
        prefix (str): Filename prefix for each image.
        format (str): Image format (e.g., 'png', 'jpg').
    """
    os.makedirs(save_dir, exist_ok=True)
    tensor_normalized = (tensor + 1.0) / 2.0
    tensor_normalized = tensor_normalized.detach().cpu()  # Move to CPU
    grid = make_grid(tensor_normalized, nrow=tensor_normalized.size(0)) 
    filename = os.path.join(save_dir, f"{prefix}_{index:05d}.{format}")
    save_image(grid, filename)

log = getLogger(__name__)


class Benchmark:
    def __init__(
        self,
        dataset: OmegaConf | WorldDatasetBase,
        model: OmegaConf | WorldModelBase,
        task: OmegaConf | TaskBase,
        accelerator: Accelerator,
        log_every: int = 100,
        sample_size: int = 1024,
        batch_size: int = 32,
        n_workers: int = 6,
    ) -> None:
        # If user provides a string, load from registry;
        # Otherwise, use the object directly.
        self.accelerator = accelerator

        if sample_size % batch_size != 0 and self.accelerator.is_main_process:
            log.w(
                "Sample size is not divisible by batch size. Consider adjusting this for \
                      maximum accuracy.",
            )

        self.dataset = (
            hydra.utils.instantiate(dataset) if isinstance(dataset, OmegaConf) else dataset
        )
        self.model = hydra.utils.instantiate(model) if isinstance(model, OmegaConf) else model
        self.task = hydra.utils.instantiate(task) if isinstance(task, OmegaConf) else task

        self.log_every = log_every
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.n_workers = n_workers

    def accumulate_metrics(self) -> dict[str, torch.Tensor]:
        accumulated_results = {}
        for metric in self.task.metrics:
            score = metric.compute()
            accumulated_results[metric.name] = torch.mean(score)

        return accumulated_results

    def log(self, msg: str) -> None:
        if self.accelerator.is_main_process:
            log.i(msg)

    @torch.no_grad()
    def run(self) -> dict[str, Any]:
        """
        1. Create a DataLoader from self.dataset
        2. Loop over each batch from the DataLoader
        3. For each batch, call self.task.execute(self.model, batch_dict)
        4. Aggregate results and return
        """

        seed = 1234
        test_sampler = DeterministicRandomSampler(
            len(self.dataset),
            sample_size=self.sample_size,
            seed=seed,
        )

        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_workers,
            sampler=test_sampler,
        )
        loader = self.accelerator.prepare(loader)

        self.log("Start of Evaluation")
        index_list = {5,32, 107, 183, 358, 408, 431, 560, 565, 648, 681, 685, 694, 800, 877, 930, 937, 953, 954, 958, 998, 1110, 1129, 1459, 1506, 1596, 1646, 1732, 1772, 2060, 2184, 2211, 2324}
        for i, batch_dict in tqdm(
            enumerate(loader),
            total=len(loader),
            disable=not self.accelerator.is_main_process,
        ):
            # TODO: Change this when we evaluate on the full dataset
            if not i in index_list:
                continue
            # batch_dict: Dict[str, torch.Tensor]
            # save_image_sequence(batch_dict['extras']['frames_hq'][0], "/work/deheng_zhang/data_filter", i)
            scores = self.task.execute(self.model, batch_dict)
            # Process metrics
            if i % self.log_every == 0:
                self.log(f"Scores: {scores}")

        all_scores = self.task.get_results()
        self.log("End of Evaluation")
        self.log(f"Final results: \n {all_scores}")
        # Return a summary or the raw list
        return {"results": all_scores}
