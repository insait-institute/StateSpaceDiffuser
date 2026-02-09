from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from dev_utils.logger import getLogger

from bench.model import WorldModelBase
from models.components.s4 import S4Model
from tools.utils import resize_video
from wrappers.ssw.model import (
    MambaDynamics,
    Tokenizer,
)
from wrappers.ssw.model import MambaWorldModel as InnerMambaWorldModel

if TYPE_CHECKING:
    from collections.abc import Mapping

    from accelerate import Accelerator
    from omegaconf import DictConfig

    from actions.action_base import BaseActionHandler

log = getLogger(__name__)


class StateSpaceWorldModel(WorldModelBase):
    def __init__(
        self,
        cfg: DictConfig,
        accelerator: Accelerator,
        n_steps_cond: int,
        action_handler: BaseActionHandler | None = None,
    ) -> None:
        enable_imagine = cfg.enable_imagine
        super().__init__(action_handler=action_handler, enable_imagine=enable_imagine)
        self.cfg = cfg
        self.accelerator = accelerator
        self.inner_model = self.init_inner_model()
        self.n_steps_cond = n_steps_cond

    def init_inner_model(self) -> InnerMambaWorldModel:
        raise NotImplementedError("Must be implemented in subclasses")

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> None:
        self.inner_model.load_state_dict(state_dict, strict=strict, assign=assign)

    @torch.no_grad()
    def imagine(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        extras: dict | None = None,
        n_steps_cond: int | None = None,
    ) -> torch.Tensor:
        frames = extras["frames_hq"][:, :n_steps_cond]

        # Call the model's forward method
        prediction = self.inner_model(
            frames=frames,
            actions=actions,
            enable_encode=True,
            enable_decode=True,
            enable_loss=False,
            enable_imagine=True,
        )

        return prediction

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        extras: dict,
        enable_imagine: bool = False,
        n_steps_cond: int | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        n_steps_cond = n_steps_cond if n_steps_cond is not None else self.n_steps_cond

        frames = extras["frames_hq"]

        batch = {
            "observations": extras["frames_hq"],
            "actions": actions,
            "extras": extras,
        }

        if enable_imagine:
            prediction = self.imagine(observations, actions, extras, n_steps_cond)

        else:
            # Call the model's forward method
            prediction = self.inner_model(
                frames=frames,
                actions=actions,
                enable_encode=True,
                enable_decode=True,
                enable_loss=False,
                enable_imagine=False,
            )

        prediction = resize_video(
            prediction,
            target_size=frames.shape[-2:],
        )

        # get how many gray frames we need to add
        n_gray_frames = frames.shape[1] - prediction.shape[1]

        gray_frame = extras["frames_hq"][:, :n_gray_frames] * 0 + 0.5

        batch["prediction"] = torch.cat(
            [
                gray_frame,
                prediction,
            ],
            dim=1,
        )

        return batch


class MambaWorldModel(StateSpaceWorldModel):
    def __init__(
        self,
        cfg: DictConfig,
        accelerator: Accelerator,
        n_steps_cond: int,
        action_handler: BaseActionHandler | None = None,
    ) -> None:
        super().__init__(
            cfg=cfg,
            accelerator=accelerator,
            n_steps_cond=n_steps_cond,
            action_handler=action_handler,
        )
        log.i("Creating MambaWorldModel model")

    def init_inner_model(self) -> InnerMambaWorldModel:
        cfg = self.cfg
        n_action_classes = cfg.n_action_classes

        if self.accelerator.is_main_process:
            tokenizer = Tokenizer(
                model_name="Cosmos-0.1-Tokenizer-CI16x16",
                model_dpath="checkpoints/nvidia_pretrained",
            )

        self.accelerator.wait_for_everyone()
        if not self.accelerator.is_main_process:
            tokenizer = Tokenizer(
                model_name="Cosmos-0.1-Tokenizer-CI16x16",
                model_dpath="checkpoints/nvidia_pretrained",
            )

        dynamics = MambaDynamics(
            dim=cfg.dynamics.dim,
            action_dim=cfg.dynamics.action_dim,
            state_dim=cfg.dynamics.state_dim,
            expand=cfg.dynamics.expand,
            n_fuser_heads=cfg.dynamics.n_fuser_heads,
            n_layers=cfg.dynamics.n_layers,
            n_action_classes=n_action_classes,
            input_size=cfg.dynamics.input_size,
        )

        return InnerMambaWorldModel(tokenizer=tokenizer, dynamics=dynamics)


class S4Dynamics(MambaDynamics):
    def __init__(
        self,
        dim: int,
        action_dim: int,
        n_layers: int,
        n_action_classes: int,
        input_size: int,
        state_dim: int,
        dropout: float,
        prenorm: bool,
        lr: float,
    ) -> None:
        super().__init__(
            dim=dim,
            action_dim=action_dim,
            state_dim=state_dim,
            expand=False,
            n_layers=n_layers,
            n_action_classes=n_action_classes,
            n_fuser_heads=1,
            input_size=input_size,
            load_backbone=False,
        )

        self.backbone = S4Model(
            d_input=input_size + action_dim,
            d_output=input_size + action_dim,
            d_model=state_dim,
            n_layers=n_layers,
            dropout=dropout,
            prenorm=prenorm,
            lr=lr,
        )


class InnerS4WorldModel(InnerMambaWorldModel):
    def __init__(
        self,
        tokenizer: Tokenizer,
        dynamics: S4Dynamics,
    ) -> None:
        super().__init__(tokenizer, dynamics)


class S4WorldModel(StateSpaceWorldModel):
    def __init__(
        self,
        cfg: DictConfig,
        accelerator: Accelerator,
        n_steps_cond: int,
        action_handler: BaseActionHandler | None = None,
    ) -> None:
        super().__init__(
            cfg=cfg,
            accelerator=accelerator,
            n_steps_cond=n_steps_cond,
            action_handler=action_handler,
        )
        log.i("Creating S4WorldModel model")
        self.inner_model = self.init_inner_model()

    def init_inner_model(self) -> InnerS4WorldModel:
        cfg = self.cfg
        n_action_classes = cfg.n_action_classes

        if self.accelerator.is_main_process:
            tokenizer = Tokenizer(
                model_name="Cosmos-0.1-Tokenizer-CI16x16",
                model_dpath="checkpoints/nvidia_pretrained",
            )

        self.accelerator.wait_for_everyone()
        if not self.accelerator.is_main_process:
            tokenizer = Tokenizer(
                model_name="Cosmos-0.1-Tokenizer-CI16x16",
                model_dpath="checkpoints/nvidia_pretrained",
            )

        dynamics = S4Dynamics(
            dim=cfg.dynamics.dim,
            action_dim=cfg.dynamics.action_dim,
            n_layers=cfg.dynamics.n_layers,
            n_action_classes=n_action_classes,
            input_size=cfg.dynamics.input_size,
            state_dim=cfg.dynamics.state_dim,
            dropout=cfg.dynamics.dropout,
            prenorm=cfg.dynamics.prenorm,
            lr=cfg.dynamics.lr,
        )

        return InnerS4WorldModel(tokenizer=tokenizer, dynamics=dynamics)
