from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch
from dev_utils.logger import getLogger
from huggingface_hub import snapshot_download
from hydra.utils import instantiate
from omegaconf import OmegaConf

from bench.model import WorldModelBase
from tools.utils import resize_video
from wrappers.ssw.agent import AgentSSM
from wrappers.ssw.envs import WorldModelEnv

# from import DiffusionSampler

if TYPE_CHECKING:
    from accelerate import Accelerator

    from actions.action_base import BaseActionHandler
    from wrappers.ssw.game import PlayEnv


log = getLogger(__name__)


def prepend_target(cfg: OmegaConf) -> None:
    if OmegaConf.is_dict(cfg):
        for key, value in cfg.items():
            if key == "_target_":
                cfg[key] = "wrappers.ssw." + value
            else:
                prepend_target(value)


def prepare_inference_mode(
    batch_size: int,
    device: torch.device,
    num_actions: int,
    ckpt_fpath: str | None = None,
    model_cfg: dict | None = None,
) -> PlayEnv:
    path_hf = Path(snapshot_download(repo_id="eloialonso/diamond", allow_patterns="csgo/*"))
    spawn_dir = path_hf / "csgo/spawn"

    if ckpt_fpath is None:
        ckpt_fpath = path_hf / "csgo/model/csgo.pt"

        # Override config
        cfg_agent = OmegaConf.load(path_hf / "csgo/config/agent/csgo.yaml")
        cfg_env = OmegaConf.load(path_hf / "csgo/config/env/csgo.yaml")

        num_actions = cfg_env.num_actions

        cfg_agent.name = "diffusion"
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

    # Models
    agent = AgentSSM(instantiate(cfg_agent, num_actions=num_actions)).to(device).eval()
    agent.load(ckpt_fpath)

    # World model environment
    sl = cfg_agent.denoiser.inner_model.num_steps_conditioning
    if agent.upsampler is not None:
        sl = max(sl, cfg_agent.upsampler.inner_model.num_steps_conditioning)

    wm_env_cfg = instantiate(cfg_wm, num_batches_to_preload=1)
    # TODO: Change this when we add upsampler
    # breakpoint()
    wm_env = WorldModelEnv(
        agent.denoiser,
        # None,
        agent.upsampler,
        agent.rew_end_model,
        spawn_dir,
        batch_size,
        sl,
        wm_env_cfg,
        return_denoising_trajectory=False,
    )

    return agent, wm_env, cfg_agent


class SSWModel(WorldModelBase):
    """
    Custom world model that processes sequences of images (BxTxCxHxW).
    """

    def __init__(
        self,
        accelerator: Accelerator,
        num_actions: int,
        ckpt_fpath: str,
        model_cfg: OmegaConf,
        action_handler: BaseActionHandler | None = None,
        enable_imagine: bool = True,
    ) -> None:
        """
        Args:
            input_shape (tuple): (C, H, W) shape of a single observation frame.
            num_actions (int): Number of possible actions.
            sequence_length (int): The expected sequence length T.
        """
        super().__init__(action_handler=action_handler, enable_imagine=enable_imagine)
        log.i("Creating SSW model")
        agent, wm_env, cfg_agent = prepare_inference_mode(
            1,
            accelerator.device,
            num_actions,
            ckpt_fpath,
            model_cfg,
        )
        agent = accelerator.prepare(agent)
        self.agent = agent
        self.model_cfg = model_cfg
        wm_env = accelerator.prepare(wm_env)
        self.wm_env: WorldModelEnv = accelerator.unwrap_model(wm_env)
        self.device = self.wm_env.device

        self.n_steps_cond = self.wm_env.num_steps_conditioning

    def imagine(
        self,
        actions: torch.Tensor,
        observations: torch.Tensor,
        extras: Optional[dict] = None,
    ) -> torch.Tensor:
        log.t("imagination")

        actions = actions.to(self.device)
        initial_obs = {
            "frames": observations.to(self.device),
            "frames_hq": extras["frames_hq"].to(self.device),
            "actions": actions[:, : observations.size(1)],
        }
        # log.t(lt.lovely(initial_obs["frames"]))
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
        extras: Optional[dict] = None,
        n_input: int = None,
    ) -> torch.Tensor:
        log.t("using ground truth")
        if n_input is None:
            n_input = self.n_steps_cond

        actions = actions.to(self.device)
        initial_obs = {
            "frames": observations.to(self.device)[:, :n_input],
            "frames_hq": extras["frames_hq"].to(self.device)[:, :n_input],
            "actions": actions[:, :n_input],
        }
        # log.t(lt.lovely(initial_obs["frames"]))
        first_frames, _ = self.wm_env.reset(obs=initial_obs)
        actions = actions[:, n_input - 1 : -1]
        all_frames = []
        for step in range(actions.size(1)):
            next_frame, _, _, _, _ = self.wm_env.step(
                actions[:, step],
                obs=observations[:, step + n_input - 1],
                obs_hq=extras["frames_hq"].to(self.device)[:, step + n_input - 1],
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
        log.t("imagination half")
        if n_input is None:
            n_input = self.n_steps_cond

        actions = actions.to(self.device)
        initial_obs = {
            "frames": observations.to(self.device)[:, :n_input],
            "frames_hq": extras["frames_hq"].to(self.device)[:, :n_input],
            "actions": actions[:, :n_input],
        }
        # log.t(lt.lovely(initial_obs["frames"]))
        first_frames, _ = self.wm_env.reset(obs=initial_obs)
        actions = actions[:, n_input - 1 : -1]
        all_frames = []
        for step in range(n_frame_cut):
            next_frame, _, _, _, _ = self.wm_env.step(
                actions[:, step],
                obs=observations[:, step + n_input - 1],
                obs_hq=extras["frames_hq"].to(self.device)[:, step + n_input - 1],
            )
            next_frame = next_frame.unsqueeze(1)
            all_frames.append(next_frame.clone())
        # give more input when there is upsampler
        for step in range(n_frame_cut, actions.size(1)):
            next_frame, _, _, _, _ = self.wm_env.step(
                actions[:, step],
            )
            all_frames.append(next_frame.unsqueeze(1).clone())
        return torch.cat(all_frames, dim=1)

    # handle model 207
    def imagine_half_low_res(
        self,
        actions: torch.Tensor,
        observations: torch.Tensor,
        extras: dict[str, torch.Tensor],
        n_frame_cut: int = 8,
        n_input: int = None,
    ) -> torch.Tensor:
        log.t("imagination half low res")
        if n_input is None:
            n_input = self.n_steps_cond

        actions = actions.to(self.device)
        # breakpoint()
        original_shape = extras["frames_hq"].shape
        extras["frames_hq"] = torch.nn.functional.interpolate(
            extras["frames_hq"].view(-1, *original_shape[-3:]),
            size=(32, 64),
            mode="bicubic",
        ).view(*original_shape[:2], -1, 32, 64)

        initial_obs = {
            "frames": observations.to(self.device)[:, :n_input],
            "frames_hq": extras["frames_hq"].to(self.device)[:, :n_input],
            "actions": actions[:, :n_input],
        }
        # log.t(lt.lovely(initial_obs["frames"]))
        first_frames, _ = self.wm_env.reset(obs=initial_obs)
        actions = actions[:, n_input - 1 : -1]
        all_frames = []
        for step in range(min(n_frame_cut, actions.size(1))):
            next_frame, _, _, _, _ = self.wm_env.step(
                actions[:, step],
                obs=observations[:, step + n_input - 1],
                obs_hq=extras["frames_hq"].to(self.device)[:, step + n_input - 1],
            )
            next_frame = next_frame.unsqueeze(1)
            all_frames.append(next_frame.clone())

        next_frame = observations[:, 0]
        for step in range(n_frame_cut, actions.size(1)):
            next_frame = torch.nn.functional.interpolate(
                next_frame,
                size=(32, 64),
                mode="bicubic",
            )
            next_frame, _, _, _, _ = self.wm_env.step(
                actions[:, step],
                obs_hq=next_frame,
            )
            all_frames.append(next_frame.unsqueeze(1).clone())
        return torch.cat(all_frames, dim=1)

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        extras: dict[str, torch.Tensor],
        enable_imagine: bool = True,
        resize: bool = False,
        n_steps_cond=None,
    ) -> dict[str, torch.Tensor]:
        # if n_steps_cond is not None:
        #     self.n_steps_cond = n_steps_cond
        batch = {
            "observations": extras["frames_hq"],
            "actions": actions,
            "extras": extras,
        }
        # here we use imagine for both imagine and ground truth input
        if enable_imagine:
            frames_hq = extras["frames_hq"]
            # prediction = self.imagine(
            #     actions=actions,
            #     observations=observations[:, : self.n_steps_cond],
            #     extras={"frames_hq": frames_hq},
            # )
            # TODO: Change this when we use different ssw models (with high res) or change to full imagination
            prediction = self.imagine_half_low_res(
                actions=actions,
                observations=observations,
                n_input=self.n_steps_cond,
                n_frame_cut=observations.shape[1] // 2,
                extras={"frames_hq": frames_hq},
            )
        else:
            frames_hq = extras["frames_hq"]
            prediction = self.imagine_gt(
                actions=actions,
                observations=observations,
                n_input=self.n_steps_cond,
                extras={"frames_hq": frames_hq},
            )
        # TODO: Change this resize argument when we add upsampler
        if resize:
            prediction = resize_video(
                prediction,
                target_size=frames_hq.shape[-2:],
            )
        gray_frames = prediction[:, : self.n_steps_cond] * 0 + 0.5
        batch["prediction"] = torch.cat(
            [
                gray_frames,
                prediction,
            ],
            dim=1,
        )
        return batch
