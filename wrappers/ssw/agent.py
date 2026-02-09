from dev_utils.import_hook import ImportHook

import_hook = ImportHook("ssw", "ssw/diamond/src")

import_hook.enable_import_hook()

from .ssw.diamond.src.agent import Agent, AgentConfig, AgentSSM  # noqa: F401

# Disable custom importer
import_hook.disable_import_hook()
