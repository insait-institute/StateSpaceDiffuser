from dev_utils.import_hook import ImportHook

import_hook = ImportHook("diamond", "diamond/src")
# Enable custom imports
import_hook.enable_import_hook()

from .diamond.src.game import PlayEnv  # noqa: F401

# Disable custom importer
import_hook.disable_import_hook()
