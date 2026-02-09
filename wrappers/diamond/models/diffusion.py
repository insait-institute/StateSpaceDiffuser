from dev_utils.import_hook import ImportHook

import_hook = ImportHook("diamond", "../diamond/src")

# Enable custom imports
import_hook.enable_import_hook()

from ..diamond.src.models.diffusion import (  # noqa: E402
    DenoiserConfig,
    DiffusionSamplerConfig,
    InnerModelConfig,
)

# Access the imported configurations to avoid compile errors
_ = DenoiserConfig
_ = DiffusionSamplerConfig
_ = InnerModelConfig

import_hook.disable_import_hook()
