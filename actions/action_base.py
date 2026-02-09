from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class BaseActionEnum(Enum):
    @staticmethod
    def get_reverse(action: BaseActionEnum) -> BaseActionEnum:
        raise NotImplementedError("This method should be implemented in subclasses.")

    @staticmethod
    def init_enum_class(cls) -> None:
        raise NotImplementedError("This method should be implemented in subclasses.")


class BaseActionHandler(ABC):
    @abstractmethod
    def __init__(self, action_size: int | None = None, device: str | torch.device = "cpu") -> None:
        self.action_size = len(BaseActionEnum) if action_size is None else action_size
        self.device = device

    @abstractmethod
    def encode_action(self, action: BaseActionEnum) -> torch.Tensor:
        pass

    @abstractmethod
    def decode_action(self, action: torch.Tensor) -> BaseActionEnum:
        pass

    @abstractmethod
    def get_reverse_action(self, action: BaseActionEnum) -> BaseActionEnum:
        pass
