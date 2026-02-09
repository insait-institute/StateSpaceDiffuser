from dev_utils.import_hook import ImportHook

import_hook = ImportHook("diamond", "diamond/src")
# Enable custom imports
import_hook.enable_import_hook()

from .diamond.src.agent import AgentConfig  # noqa: F401
from .diamond.src.agent import Agent as DiamondAgent

# Disable custom importer
import_hook.disable_import_hook()


import torch
import pickle
import types
from pathlib import Path

from collections import OrderedDict

# Create a fake pickle module with a patched Unpickler
patched_pickle = types.ModuleType("patched_pickle")

class DummyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if "evaluation" in module:
            return type('Dummy', (), {})  # Dummy class
        return super().find_class(module, name)

def custom_load(file, **kwargs):
    return DummyUnpickler(file).load()

# Patch the loads/load/dump methods (minimal set torch.load uses)
patched_pickle.Unpickler = DummyUnpickler
patched_pickle.load = custom_load


def extract_state_dict(state_dict: OrderedDict, module_name: str) -> OrderedDict:
    return OrderedDict(
        {
            k.split(".", 1)[1]: v
            for k, v in state_dict.items()
            if k.startswith(module_name)
        }
    )


class Agent(DiamondAgent):
    def __init__(self, cfg: AgentConfig):
        super().__init__(cfg)
        
    def load(
        self,
        path_to_ckpt: Path,
        load_denoiser: bool = True,
        load_upsampler: bool = True,
        load_rew_end_model: bool = True,
        load_actor_critic: bool = True,
    ) -> None:
        sd = torch.load(
            Path(path_to_ckpt), map_location=self.device, weights_only=False, pickle_module=patched_pickle
        )
        if "agent" in sd:
            sd = sd["agent"]
            
        if load_denoiser:
            self.denoiser.load_state_dict(extract_state_dict(sd, "denoiser"), strict=False)
        if load_upsampler:
            self.upsampler.load_state_dict(extract_state_dict(sd, "upsampler"), strict=False)
        if load_rew_end_model and self.rew_end_model is not None:
            self.rew_end_model.load_state_dict(extract_state_dict(sd, "rew_end_model"))
        if load_actor_critic and self.actor_critic is not None:
            self.actor_critic.load_state_dict(extract_state_dict(sd, "actor_critic"))

        return sd