from dev_utils.import_hook import ImportHook

import_hook = ImportHook("ssw", "ssw")

# Enable custom imports
import_hook.enable_import_hook()

from .ssw.model.ssw import (  # noqa: E402
    MambaDynamics,
    MambaWorldModel,
    Tokenizer,
)

_ = MambaDynamics
_ = MambaWorldModel
_ = Tokenizer

import_hook.disable_import_hook()
