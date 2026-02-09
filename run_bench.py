import logging
import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import hydra
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf

from bench import Benchmark
from bench.model import WorldModelBase
from bench.task import TaskBase
from tools.build_tools import build_dataset, init_model, init_task

logging.basicConfig(level=logging.INFO)

OmegaConf.register_new_resolver("eval", eval)


@hydra.main(version_base="1.3", config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Resolve the full configuration
    # cfg = OmegaConf.to_container(cfg, resolve=True)
    # cfg = OmegaConf.create(cfg)

    # accelerator = Accelerator()  ##mixed_precision="bf16")

    accelerator = Accelerator(
        cpu=False,
        log_with="wandb",
        mixed_precision="bf16",
    )

    model: WorldModelBase = init_model(
        cfg,
        accelerator=accelerator,
    )

    # Build benchmark components
    test_dataset = build_dataset(cfg.data)
    task: TaskBase = init_task(cfg, accelerator=accelerator)

    benchmark = Benchmark(
        dataset=test_dataset,
        model=model,
        task=task,
        accelerator=accelerator,
        log_every=cfg.bench.log_every,
        sample_size=cfg.bench.sample_size,
        batch_size=cfg.bench.batch_size,
    )

    # Run the benchmark
    benchmark.run()


if __name__ == "__main__":
    main()
