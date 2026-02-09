from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from accelerate import Accelerator
from dev_utils.logger import getLogger
from huggingface_hub import snapshot_download
from hydra.utils import instantiate
from omegaconf import OmegaConf

from bench.model import WorldModelBase
from tools.utils import resize_video
from wrappers.diamond.agent import Agent
from wrappers.diamond.envs import WorldModelEnv

log = getLogger(__name__)

if TYPE_CHECKING:
    from accelerate import Accelerator

    from actions.action_base import BaseActionHandler


def prepend_target(cfg: OmegaConf) -> None:
    if OmegaConf.is_dict(cfg):
        for key, value in cfg.items():
            if key == "_target_":
                cfg[key] = "wrappers.diamond." + value
            else:
                prepend_target(value)


def prepare_inference_mode(
    batch_size: int,
    device: torch.device,
    num_actions: int,
    ckpt_fpath: str | None = None,
    model_cfg: dict | None = None,
):
    path_hf = Path(snapshot_download(repo_id="eloialonso/diamond", allow_patterns="csgo/*"))
    spawn_dir = path_hf / "csgo/spawn"

    if ckpt_fpath is None:
        ckpt_fpath = path_hf / "csgo/model/csgo.pt"

        # Override config
        cfg_agent = OmegaConf.load(path_hf / "csgo/config/agent/csgo.yaml")
        cfg_env = OmegaConf.load(path_hf / "csgo/config/env/csgo.yaml")
        num_actions = cfg_env.num_actions
        cfg_wm = OmegaConf.create(
            {
                "_target_": "envs.WorldModelEnvConfig",
                "horizon": 1000,
                "num_batches_to_preload": 256,
                "diffusion_sampler_next_obs": {
                    "_target_": "models.diffusion.DiffusionSamplerConfig",
                    "num_steps_denoising": 3,
                    "sigma_min": 0.002,
                    "sigma_max": 5.0,
                    "rho": 7,
                    "order": 1,
                    "s_churn": 0.0,
                    "s_tmin": 0.0,
                    "s_tmax": "${eval:'float(\"inf\")'}",
                    "s_noise": 1.0,
                    "s_cond": 0.005,
                },
                "diffusion_sampler_upsampling": {
                    "_target_": "models.diffusion.DiffusionSamplerConfig",
                    "num_steps_denoising": 10,
                    "sigma_min": 1,
                    "sigma_max": 5.0,
                    "rho": 7,
                    "order": 2,
                    "s_churn": 10.0,
                    "s_tmin": 1,
                    "s_tmax": 5,
                    "s_noise": 0.9,
                    "s_cond": 0,
                },
            },
        )
    else:
        cfg_wm = model_cfg.world_model_env
        cfg_agent = model_cfg.agent
        cfg_env = model_cfg.env

    prepend_target(cfg_agent)
    prepend_target(cfg_env)
    prepend_target(cfg_wm)

    # cfg_agent["_target_"] = "wrappers.diamond.agent.AgentConfig"

    # Models
    agent = Agent(instantiate(cfg_agent, num_actions=num_actions)).to(device).eval()
    agent.load(ckpt_fpath)

    # World model environment
    sl = cfg_agent.denoiser.inner_model.num_steps_conditioning
    if agent.upsampler is not None:
        sl = max(sl, cfg_agent.upsampler.inner_model.num_steps_conditioning)

    wm_env_cfg = instantiate(cfg_wm, num_batches_to_preload=1)
    wm_env = WorldModelEnv(
        agent.denoiser,
        None,
        agent.rew_end_model,
        spawn_dir,
        batch_size,
        sl,
        wm_env_cfg,
        return_denoising_trajectory=False,
    )
    return agent, wm_env, cfg_agent


class DiamondWorldModel(WorldModelBase):
    """
    Custom world model that processes sequences of images (BxTxCxHxW).
    """

    def __init__(
        self,
        accelerator: Accelerator,
        num_actions: int,
        ckpt_fpath: str | None,
        model_cfg: OmegaConf,
        action_handler: BaseActionHandler | None = None,
        enable_imagine: bool = False,
    ) -> None:
        """
        Args:
            input_shape (tuple): (C, H, W) shape of a single observation frame.
            num_actions (int): Number of possible actions.
            sequence_length (int): The expected sequence length T.
        """
        enable_imagine = model_cfg.enable_imagine if model_cfg is not None else enable_imagine
        super().__init__(action_handler=action_handler, enable_imagine=enable_imagine)
        agent, wm_env, cfg_agent = prepare_inference_mode(
            1,
            accelerator.device,
            num_actions,
            ckpt_fpath,
            model_cfg,
        )
        agent = accelerator.prepare(agent)
        wm_env = accelerator.prepare(wm_env)
        self.wm_env: WorldModelEnv = accelerator.unwrap_model(wm_env)
        self.device = self.wm_env.device

        self.n_steps_cond = self.wm_env.num_steps_conditioning

    def imagine(
        self,
        actions: torch.Tensor,
        observations: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        actions = actions.to(self.device)
        initial_obs = {
            "frames": observations.to(self.device),
            "frames_hq": extras["frames_hq"].to(self.device),
            "actions": actions[:, : observations.size(1)],
        }
        first_frames, _ = self.wm_env.reset(obs=initial_obs)
        actions = actions[:, observations.size(1) :]
        all_frames = []
        for step in range(actions.size(1)):
            # don't use the ground truth observation
            next_frame, _, _, _, _ = self.wm_env.step(actions[:, step])
            next_frame = next_frame.unsqueeze(1)
            all_frames.append(next_frame.clone())
        return torch.cat(all_frames, dim=1)

    def imagine_gt(
        self,
        actions: torch.Tensor,
        observations: torch.Tensor,
        extras: dict[str, torch.Tensor],
        n_input: int = None,
    ) -> torch.Tensor:
        actions = actions.to(self.device)
        initial_obs = {
            "frames": observations.to(self.device)[:, :n_input],
            "frames_hq": extras["frames_hq"].to(self.device)[:, :n_input],
            "actions": actions[:, :n_input],
        }
        first_frames, _ = self.wm_env.reset(obs=initial_obs)
        actions = actions[:, n_input - 1 : -1]
        all_frames = []
        for step in range(actions.size(1)):
            next_frame, _, _, _, _ = self.wm_env.step(
                actions[:, step],
                obs=observations[:, step + n_input - 1],
            )
            next_frame = next_frame.unsqueeze(1)
            all_frames.append(next_frame.clone())
        return torch.cat(all_frames, dim=1)

    def imagine_half(
        self,
        actions: torch.Tensor,
        observations: torch.Tensor,
        extras: dict[str, torch.Tensor],
        n_frame_cut: int = 8,
        n_input: int = None,
    ) -> torch.Tensor:
        actions = actions.to(self.device)
        initial_obs = {
            "frames": observations.to(self.device)[:, :n_input],
            "frames_hq": extras["frames_hq"].to(self.device)[:, :n_input],
            "actions": actions[:, :n_input],
        }
        first_frames, _ = self.wm_env.reset(obs=initial_obs)
        actions = actions[:, n_input - 1 : -1]
        all_frames = []

        for step in range(n_frame_cut):
            next_frame, _, _, _, _ = self.wm_env.step(
                actions[:, step],
                obs=observations[:, step + n_input - 1],
                obs_hq=extras["frames_hq"].to(self.device)[:, step + n_input - 1],
            )
            all_frames.append(next_frame.unsqueeze(1).clone())

        for step in range(n_frame_cut, actions.size(1)):
            next_frame, _, _, _, _ = self.wm_env.step(
                actions[:, step],
            )
            all_frames.append(next_frame.unsqueeze(1).clone())
        return torch.cat(all_frames, dim=1)

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        extras: dict[str, torch.Tensor],
        enable_imagine: bool = True,
        n_steps_cond: int | None = None,
    ) -> dict[str, torch.Tensor]:
        if n_steps_cond is not None and self.n_steps_cond > n_steps_cond:
            msg = (
                "n_steps_cond is smaller than DiamondWorldModel.n_steps_cond "
                f"({n_steps_cond} < {self.n_steps_cond})."
            )
            raise ValueError(msg)

        # frames_hq = extras["frames_hq"]
        frames_hq = observations
        offset = 0
        if n_steps_cond is not None:
            offset = n_steps_cond - self.n_steps_cond

        if offset > 0:
            observations_in = observations[:, offset:]
            actions_in = actions[:, offset:]
            extras_in = {"frames_hq": frames_hq[:, offset:]}
        else:
            observations_in = observations
            actions_in = actions
            extras_in = extras

        batch = {
            "observations": frames_hq,
            "actions": actions,
            "extras": extras,
        }

        if enable_imagine:
            prediction = self.imagine(
                actions=actions_in,
                observations=observations_in[:, : self.n_steps_cond],
                extras={"frames_hq": extras_in["frames_hq"]},
            )
        else:
            prediction = self.imagine_gt(
                actions=actions_in,
                observations=observations_in,
                n_input=self.n_steps_cond,
                extras={"frames_hq": extras_in["frames_hq"]},
            )
        prediction = resize_video(
            prediction,
            target_size=frames_hq.shape[-2:],
        )
        gray_frames = prediction[:, : self.n_steps_cond] * 0 + 0.5
        prediction = torch.cat(
            [
                gray_frames,
                prediction,
            ],
            dim=1,
        )

        if offset > 0:
            gray_pad = frames_hq[:, :offset] * 0 + 0.5
            prediction = torch.cat([gray_pad, prediction], dim=1)

        batch["prediction"] = prediction
        return batch
