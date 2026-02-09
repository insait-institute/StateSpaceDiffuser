from typing import Any

import torch
from dev_utils.import_hook import ImportHook

import_hook = ImportHook("ssw", "ssw/diamond/src")

# Enable custom imports
import_hook.enable_import_hook()

from .ssw.diamond.src.envs import WorldModelEnv as OldWorldModelEnv
from .ssw.diamond.src.envs import WorldModelEnvConfig  # noqa: F401

# Disable custom importer
import_hook.disable_import_hook()

ResetOutput = tuple[torch.FloatTensor, dict[str, Any]]


class WorldModelEnv(OldWorldModelEnv):
    def __init__(self, *args, **kwargs):
        if "num_envs" in kwargs:
            num_envs = kwargs["num_envs"]
            kwargs["num_envs"] = 1
        else:
            num_envs = args[4]
            args = args[:4] + (1,) + args[5:]

        super().__init__(*args, **kwargs)

        self.num_envs = num_envs
        self.num_steps_conditioning = (
            self.sampler_next_obs.denoiser.cfg.inner_model.num_steps_conditioning
        )

    @torch.no_grad()
    def reset(self, **kwargs) -> ResetOutput:
        if "obs" in kwargs:
            obs = kwargs["obs"]["frames"]
            obs_full_res = kwargs["obs"]["frames_hq"]
            act = kwargs["obs"]["actions"]
            next_act = [0]
            hx = 0
            cx = 0
            batch_size = obs.size(0)
        else:
            obs, obs_full_res, act, next_act, (hx, cx) = self.generator_init.send(self.num_envs)
            batch_size = self.num_envs

        self.sampler_next_obs.denoiser.cache_obs = []
        self.sampler_next_obs.denoiser.cache_acts = []
        if self.sampler_upsampling is not None:
            self.sampler_upsampling.denoiser.cache_obs = []
            self.sampler_upsampling.denoiser.cache_acts = []
        self.cache_acts = []
        self.obs_buffer = obs
        self.act_buffer = act
        self.next_act = next_act[0]
        self.obs_full_res_buffer = obs_full_res
        self.ep_len = torch.zeros(batch_size, dtype=torch.long, device=obs.device)
        self.hx_rew_end = hx
        self.cx_rew_end = cx

        # print(lt.lovely(self.obs_buffer))

        obs_to_return = (
            self.obs_buffer[:, -1]
            if self.sampler_upsampling is None
            else self.obs_full_res_buffer[:, -1]
        )
        return obs_to_return, {}

    def step(self, act: torch.LongTensor, obs: torch.Tensor = None, obs_hq: torch.Tensor = None):
        if obs is not None:
            self.obs_buffer[:, -1] = obs
            
        if obs_hq is not None:
            self.obs_full_res_buffer[:, -1] = obs_hq
        return super().step(act)
