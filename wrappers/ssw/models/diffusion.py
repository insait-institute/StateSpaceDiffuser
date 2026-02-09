from dev_utils.import_hook import ImportHook

import_hook = ImportHook("ssw", "../ssw/diamond/src")

# Enable custom imports
import_hook.enable_import_hook()

from ..ssw.diamond.src.models.diffusion import (  # noqa: E402
    DenoiserConfig,
    DiffusionSamplerConfig,
    InnerModelConfig,
    DenoiserSSMConfig,
    DiffusionSampler
)

_ = DenoiserConfig
_ = DiffusionSamplerConfig
_ = InnerModelConfig
_ = DenoiserSSMConfig
_ = DiffusionSampler
import_hook.disable_import_hook()
