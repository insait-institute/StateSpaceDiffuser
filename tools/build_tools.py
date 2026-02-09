from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from dev_utils.logger import getLogger
from dev_utils.model_management import CheckpointDirManager
from omegaconf import OmegaConf

from actions.csgo_actions import CSGOActionEnum, CSGOActionHandler, ReverseCSGOActionHandler
from actions.minigrid_action import MiniGridActionEnum, MiniGridActionHandler
from data.dataset_csgo import CSGODataset, MirroredCSGODataset
from data.dataset_minigrid import MiniGridDataset
from models.diamond_model import DiamondWorldModel

if TYPE_CHECKING:
    from accelerate import Accelerator
    from action_base import BaseActionEnum, BaseActionHandler
    from omegaconf import DictConfig

    from bench.dataset import WorldDatasetBase
    from bench.model import WorldModelBase
    from bench.task import TaskBase

log = getLogger("init_model")


################# MODELS #################


def init_model(cfg: DictConfig, accelerator: Accelerator) -> WorldModelBase:
    """
    Initialize the model based on the provided configuration.

    Args:
        cfg (DictConfig): The configuration dictionary containing model parameters.
    """

    model = None

    if cfg.model.name == "mamba_wm":
        model = init_mamba_wm(deepcopy(cfg), accelerator)
    elif cfg.model.name == "s4_wm":
        model = init_s4_wm(deepcopy(cfg), accelerator)
    elif cfg.model.name == "diamond":
        model = init_diamond_wm(deepcopy(cfg), accelerator)
    elif cfg.model.name == "ssw":
        model = init_ssw(deepcopy(cfg), accelerator)
    if cfg.common.ckpt_load is not None and cfg.model.name not in {"diamond", "ssw"}:
        log.i(f"Loading checkpoint {cfg.common.ckpt_load} from {cfg.common.ckpt_dpath}")
        load_model(model=model, cfg=cfg)

    model = accelerator.prepare(model)
    model = accelerator.unwrap_model(model)
    model = model.to(accelerator.device)

    return model


def get_action_handler(cfg: DictConfig, accelerator: Accelerator) -> BaseActionHandler:
    """
    Get the action handler based on the provided name.

    Args:
        cfg (DictConfig): The configuration dictionary containing model parameters.
        accelerator (Accelerator): The accelerator instance.

    Returns:
        BaseActionHandler: The action handler instance.
    """
    name = cfg.name
    has_reverse_actions = cfg.get("has_reverse_actions", False)
    if name == "csgo":
        if has_reverse_actions:
            return ReverseCSGOActionHandler(
                device=accelerator.device,
            )
        return CSGOActionHandler(
            device=accelerator.device,
        )
    if name == "minigrid":
        return MiniGridActionHandler(
            device=accelerator.device,
        )
    return None


def get_action_enum(name: str, action: str) -> BaseActionEnum:
    """
    Get the action enum based on the provided name.

    Args:
        name (str): The name of the action enum to retrieve.
        action (str): The specific action to retrieve.

    Returns:
        BaseActionEnum: The action enum instance.
    """
    if name == "csgo":
        return CSGOActionEnum[action]
    if name == "minigrid":
        return MiniGridActionEnum[action]
    return None


def init_mamba_wm(cfg: DictConfig, accelerator: Accelerator):
    """
    Initialize the Mamba World Model based on the provided configuration.

    Args:
        cfg (DictConfig): The configuration dictionary containing model parameters.
    """

    from models.ssm_model import MambaWorldModel

    n_action_classes = cfg.data.n_actions
    action_handler = get_action_handler(cfg.data, accelerator)
    cfg = cfg.model

    OmegaConf.set_struct(cfg, False)
    cfg.n_action_classes = n_action_classes
    OmegaConf.set_struct(cfg, True)

    log.t(
        f"Currently on process id {accelerator.process_index} on device {accelerator.device}",
    )

    n_steps_cond = cfg.n_steps_cond

    model = MambaWorldModel(
        cfg=cfg,
        accelerator=accelerator,
        n_steps_cond=n_steps_cond,
        action_handler=action_handler,
    )

    return model


def init_s4_wm(cfg: DictConfig, accelerator: Accelerator):
    """
    Initialize the S4 World Model based on the provided configuration.

    Args:
        cfg (DictConfig): The configuration dictionary containing model parameters.
    """

    from models.ssm_model import S4WorldModel
    n_action_classes = cfg.data.n_actions
    action_handler = get_action_handler(cfg.data, accelerator)
    cfg = cfg.model

    OmegaConf.set_struct(cfg, False)
    cfg.n_action_classes = n_action_classes
    OmegaConf.set_struct(cfg, True)

    log.t(
        f"Currently on process id {accelerator.process_index} on device {accelerator.device}",
    )

    n_steps_cond = cfg.n_steps_cond

    model = S4WorldModel(
        cfg=cfg,
        accelerator=accelerator,
        n_steps_cond=n_steps_cond,
        action_handler=action_handler,
    )

    return model


def init_diamond_wm(cfg: DictConfig, accelerator: Accelerator) -> DiamondWorldModel:
    num_actions = cfg.data.n_actions
    ckpt_fpath = (
        load_model(cfg=cfg, return_ckpt_fpath_only=True) if cfg.common.load_enable else None
    )
    model_cfg = cfg.model
    action_handler = get_action_handler(cfg.data, accelerator)
    return DiamondWorldModel(
        accelerator=accelerator,
        num_actions=num_actions,
        ckpt_fpath=ckpt_fpath,
        model_cfg=model_cfg,
        action_handler=action_handler,
        enable_imagine=cfg.model.enable_imagine,
    )


def init_ssw(cfg: DictConfig, accelerator: Accelerator):
    from models.ssw_model import SSWModel
    
    num_actions = cfg.data.n_actions
    ckpt_fpath = (
        load_model(cfg=cfg, return_ckpt_fpath_only=True) if cfg.common.load_enable else None
    )
    model_cfg = cfg.model
    action_handler = get_action_handler(cfg.data, accelerator)
    return SSWModel(
        accelerator=accelerator,
        num_actions=num_actions,
        ckpt_fpath=ckpt_fpath,
        model_cfg=model_cfg,
        action_handler=action_handler,
        enable_imagine=cfg.model.enable_imagine,
    )


def load_model(
    model: WorldModelBase | None = None,
    cfg: DictConfig | None = None,
    return_ckpt_fpath_only: bool = False,
) -> str | None:
    assert cfg is not None, "cfg must be provided if model is None"

    assert model is not None or return_ckpt_fpath_only, (
        "model must be provided if return_ckpt_fpath_only is False"
    )

    if cfg.common.ckpt_enable_manager:
        if cfg.common.ckpt_dpath is None:
            msg = "Checkpoint directory is not set."
            raise ValueError(msg)

        try:
            ckpt_load = eval(cfg.common.ckpt_load)
        except Exception:
            ckpt_load = cfg.common.ckpt_load
        ckpt_load_fpath = None
        if ckpt_load is not None:
            # check if it is a tuple
            if isinstance(ckpt_load, tuple):
                cdm = CheckpointDirManager(cfg.common.ckpt_dpath)
                model_dpath = cdm.get_dpath_by_id(int(ckpt_load[0]))
                ckpt_load_fpath = model_dpath / f"model-{ckpt_load[1]}.pt"
            else:
                ckpt_load_fpath = Path(ckpt_load)
                ckpt_load = None

        ckpt_dpath = Path(cfg.common.ckpt_dpath)
        ckpt_dpath.mkdir(parents=True, exist_ok=True)

        if cfg.common.ckpt_central_dpath is not None:
            ckpt_central_dpath = Path(cfg.common.ckpt_central_dpath)
            ckpt_central_dpath.mkdir(parents=True, exist_ok=True)
        else:
            ckpt_central_dpath = ckpt_dpath
    else:
        ckpt_load_fpath = cfg.common.ckpt_load

    if return_ckpt_fpath_only:
        return ckpt_load_fpath

    if ckpt_load_fpath is not None:
        log.t(ckpt_load_fpath)
        data = torch.load(ckpt_load_fpath, map_location=torch.device("cpu"), weights_only=False)
        # model = accelerator.unwrap_model(model)
        model.load_state_dict(data["model"], strict=True)
    return None
    # model = accelerator.prepare(model)


################# DATASETS #################


def build_dataset(cfg: DictConfig) -> WorldDatasetBase:
    if cfg.name == "minigrid":
        return build_minigrid_dataset(cfg)
    if cfg.name == "csgo":
        return build_csgo_dataset(cfg)
    return None


def build_minigrid_dataset(cfg: DictConfig) -> MiniGridDataset:
    dataset_args = {
        key: value
        for key, value in cfg.items()
        if key in MiniGridDataset.__init__.__code__.co_varnames
        # or key in ExtendedMiniGridDataset.__init__.__code__.co_varnames
    }

    dataset = MiniGridDataset(
        # dataset = ExtendedMiniGridDataset(
        **dataset_args,
    )

    return dataset


def build_csgo_dataset(cfg: DictConfig) -> CSGODataset:
    dataset_args = {
        key: value for key, value in cfg.items() if key in CSGODataset.__init__.__code__.co_varnames
    }

    dataset = MirroredCSGODataset(
        **dataset_args,
    )

    return dataset


################# TASKS #################


def init_task(cfg: DictConfig, accelerator: Accelerator) -> TaskBase:
    n_steps = cfg.task.n_steps
    qualitative_dpath = cfg.task.qualitative_dpath
    data_range = cfg.model.data_range

    if cfg.task.name == "match_ground_truth":
        from tasks.match_ground_truth_task import MatchGroundTruthTask

        return MatchGroundTruthTask(
            n_steps,
            qualitative_dpath=qualitative_dpath,
            data_range=data_range,
            device=accelerator.device,
        )

    if cfg.task.name == "reverse_action":
        from tasks.reverse_action_task import ReverseActionTask

        action_enum = get_action_enum(cfg.data.name, cfg.task.action)

        return ReverseActionTask(
            action_enum=action_enum,
            n_steps=n_steps,
            data_range=data_range,
            device=accelerator.device,
            qualitative_dpath=qualitative_dpath,
        )

    return None
