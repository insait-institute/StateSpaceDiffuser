from dev_utils.import_hook import ImportHook

import_hook = ImportHook("ssw", "ssw/diamond/src")

# Enable custom imports
import_hook.enable_import_hook()

from .ssw.diamond.src.game import PlayEnv  # noqa: F401

# Disable custom importer
import_hook.disable_import_hook()
