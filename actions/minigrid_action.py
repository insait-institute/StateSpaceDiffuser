from __future__ import annotations

import torch

from .action_base import BaseActionEnum, BaseActionHandler


class MiniGridActionEnum(BaseActionEnum):
    MOVE_LEFT = "move_left"
    MOVE_RIGHT = "move_right"
    MOVE_UP = "move_up"
    MOVE_DOWN = "move_down"

    @staticmethod
    def get_reverse(action: BaseActionEnum) -> BaseActionEnum:
        reverse_map = {
            MiniGridActionEnum.MOVE_LEFT: MiniGridActionEnum.MOVE_RIGHT,
            MiniGridActionEnum.MOVE_RIGHT: MiniGridActionEnum.MOVE_LEFT,
            MiniGridActionEnum.MOVE_UP: MiniGridActionEnum.MOVE_DOWN,
            MiniGridActionEnum.MOVE_DOWN: MiniGridActionEnum.MOVE_UP,
        }
        return reverse_map.get(action)


class MiniGridActionHandler(BaseActionHandler):
    def __init__(self, action_size: int | None = None, device: str | torch.device = "cpu") -> None:
        super().__init__(action_size, device)
        self.action_size = len(MiniGridActionEnum) if action_size is None else action_size

    def encode_action(self, action: MiniGridActionEnum) -> torch.Tensor:
        action_tensor = torch.zeros(self.action_size, dtype=torch.float32)

        if action == MiniGridActionEnum.MOVE_RIGHT:
            action_tensor[0] = 1
        elif action == MiniGridActionEnum.MOVE_DOWN:
            action_tensor[1] = 1
        elif action == MiniGridActionEnum.MOVE_LEFT:
            action_tensor[2] = 1
        elif action == MiniGridActionEnum.MOVE_UP:
            action_tensor[3] = 1

        return action_tensor

    def decode_action(self, action: torch.Tensor) -> MiniGridActionEnum:
        if action[0] == 1:
            return MiniGridActionEnum.MOVE_RIGHT
        if action[1] == 1:
            return MiniGridActionEnum.MOVE_DOWN
        if action[2] == 1:
            return MiniGridActionEnum.MOVE_LEFT
        if action[3] == 1:
            return MiniGridActionEnum.MOVE_UP
        raise ValueError("Invalid action tensor.")

    def get_reverse_action(self, action: MiniGridActionEnum) -> MiniGridActionEnum:
        return MiniGridActionEnum.get_reverse(action)
