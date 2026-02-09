from __future__ import annotations

from typing import TYPE_CHECKING

from data.csgo.action_processing import (
    CSGOAction,
    ReverseCSGOAction,
    decode_csgo_action,
    decode_reverse_csgo_action,
    encode_csgo_action,
    encode_reverse_csgo_action,
)

from .action_base import BaseActionEnum, BaseActionHandler

if TYPE_CHECKING:
    import torch


class CSGOActionEnum(BaseActionEnum):
    """CSGO-specific action enum, including movement, mouse actions, and keys."""

    # Movement keys
    MOVE_FORWARD = "w"
    MOVE_BACKWARD = "s"
    MOVE_LEFT = "a"
    MOVE_RIGHT = "d"
    JUMP = "space"
    CROUCH = "left ctrl"
    WALK = "left shift"

    # Weapon actions
    WEAPON_1 = "1"
    WEAPON_2 = "2"
    WEAPON_3 = "3"
    RELOAD = "r"

    # Mouse actions
    LEFT_CLICK = "l_click"
    RIGHT_CLICK = "r_click"

    # Mouse X-axis movements
    TURN_LEFT_1000 = -1000
    TURN_LEFT_500 = -500
    TURN_LEFT_300 = -300
    TURN_LEFT_200 = -200
    TURN_LEFT_100 = -100
    TURN_LEFT_60 = -60
    TURN_LEFT_30 = -30
    TURN_LEFT_20 = -20
    TURN_LEFT_10 = -10
    TURN_LEFT_4 = -4
    TURN_LEFT_2 = -2
    MOUSE_X_ZERO = 0
    TURN_RIGHT_2 = 2
    TURN_RIGHT_4 = 4
    TURN_RIGHT_10 = 10
    TURN_RIGHT_20 = 20
    TURN_RIGHT_30 = 30
    TURN_RIGHT_60 = 60
    TURN_RIGHT_100 = 100
    TURN_RIGHT_200 = 200
    TURN_RIGHT_300 = 300
    TURN_RIGHT_500 = 500
    TURN_RIGHT_1000 = 1000

    # Mouse Y-axis movements
    LOOK_DOWN_200 = -200
    LOOK_DOWN_100 = -100
    LOOK_DOWN_50 = -50
    LOOK_DOWN_20 = -20
    LOOK_DOWN_10 = -10
    LOOK_DOWN_4 = -4
    LOOK_DOWN_2 = -2
    MOUSE_Y_ZERO = 0
    LOOK_UP_2 = 2
    LOOK_UP_4 = 4
    LOOK_UP_10 = 10
    LOOK_UP_20 = 20
    LOOK_UP_50 = 50
    LOOK_UP_100 = 100
    LOOK_UP_200 = 200

    @classmethod
    def init_enum_class(cls) -> None:
        pass

    @classmethod
    def KEYS(cls) -> list[CSGOActionEnum]:
        return [
            cls.MOVE_FORWARD,
            cls.MOVE_BACKWARD,
            cls.MOVE_LEFT,
            cls.MOVE_RIGHT,
            cls.JUMP,
            cls.CROUCH,
            cls.WALK,
            cls.WEAPON_1,
            cls.WEAPON_2,
            cls.WEAPON_3,
            cls.RELOAD,
        ]

    @classmethod
    def MOUSE_X(cls) -> list[CSGOActionEnum]:
        return [
            cls.TURN_LEFT_1000,
            cls.TURN_LEFT_500,
            cls.TURN_LEFT_300,
            cls.TURN_LEFT_200,
            cls.TURN_LEFT_100,
            cls.TURN_LEFT_60,
            cls.TURN_LEFT_30,
            cls.TURN_LEFT_20,
            cls.TURN_LEFT_10,
            cls.TURN_LEFT_4,
            cls.TURN_LEFT_2,
            cls.MOUSE_X_ZERO,
            cls.TURN_RIGHT_2,
            cls.TURN_RIGHT_4,
            cls.TURN_RIGHT_10,
            cls.TURN_RIGHT_20,
            cls.TURN_RIGHT_30,
            cls.TURN_RIGHT_60,
            cls.TURN_RIGHT_100,
            cls.TURN_RIGHT_200,
            cls.TURN_RIGHT_300,
            cls.TURN_RIGHT_500,
            cls.TURN_RIGHT_1000,
        ]

    @classmethod
    def MOUSE_Y(cls) -> list[CSGOActionEnum]:
        return [
            cls.LOOK_DOWN_200,
            cls.LOOK_DOWN_100,
            cls.LOOK_DOWN_50,
            cls.LOOK_DOWN_20,
            cls.LOOK_DOWN_10,
            cls.LOOK_DOWN_4,
            cls.LOOK_DOWN_2,
            cls.MOUSE_Y_ZERO,
            cls.LOOK_UP_2,
            cls.LOOK_UP_4,
            cls.LOOK_UP_10,
            cls.LOOK_UP_20,
            cls.LOOK_UP_50,
            cls.LOOK_UP_100,
            cls.LOOK_UP_200,
        ]

    @classmethod
    def MOUSE_CLICKS(cls) -> list[CSGOActionEnum]:
        return [cls.LEFT_CLICK, cls.RIGHT_CLICK]

    @staticmethod
    def get_reverse_action(action: CSGOActionEnum) -> CSGOActionEnum:
        """Return the reverse of an action if applicable."""
        reverse_map = {
            CSGOActionEnum.MOVE_FORWARD: CSGOActionEnum.MOVE_BACKWARD,
            CSGOActionEnum.MOVE_BACKWARD: CSGOActionEnum.MOVE_FORWARD,
            CSGOActionEnum.MOVE_LEFT: CSGOActionEnum.MOVE_RIGHT,
            CSGOActionEnum.MOVE_RIGHT: CSGOActionEnum.MOVE_LEFT,
            CSGOActionEnum.TURN_RIGHT_1000: CSGOActionEnum.TURN_LEFT_1000,
            CSGOActionEnum.TURN_RIGHT_500: CSGOActionEnum.TURN_LEFT_500,
            CSGOActionEnum.TURN_RIGHT_300: CSGOActionEnum.TURN_LEFT_300,
            CSGOActionEnum.TURN_RIGHT_200: CSGOActionEnum.TURN_LEFT_200,
            CSGOActionEnum.TURN_RIGHT_100: CSGOActionEnum.TURN_LEFT_100,
            CSGOActionEnum.TURN_RIGHT_60: CSGOActionEnum.TURN_LEFT_60,
            CSGOActionEnum.TURN_RIGHT_30: CSGOActionEnum.TURN_LEFT_30,
            CSGOActionEnum.TURN_RIGHT_20: CSGOActionEnum.TURN_LEFT_20,
            CSGOActionEnum.TURN_RIGHT_10: CSGOActionEnum.TURN_LEFT_10,
            CSGOActionEnum.TURN_RIGHT_4: CSGOActionEnum.TURN_LEFT_4,
            CSGOActionEnum.TURN_RIGHT_2: CSGOActionEnum.TURN_LEFT_2,
            CSGOActionEnum.LOOK_UP_200: CSGOActionEnum.LOOK_DOWN_200,
            CSGOActionEnum.LOOK_UP_100: CSGOActionEnum.LOOK_DOWN_100,
            CSGOActionEnum.LOOK_UP_50: CSGOActionEnum.LOOK_DOWN_50,
            CSGOActionEnum.LOOK_UP_20: CSGOActionEnum.LOOK_DOWN_20,
            CSGOActionEnum.LOOK_UP_10: CSGOActionEnum.LOOK_DOWN_10,
            CSGOActionEnum.LOOK_UP_4: CSGOActionEnum.LOOK_DOWN_4,
            CSGOActionEnum.LOOK_UP_2: CSGOActionEnum.LOOK_DOWN_2,
        }
        return reverse_map.get(action)


class CSGOActionHandler(BaseActionHandler):
    """
    A handler class for encoding, decoding, and reversing Counter-Strike: Global Offensive (CSGO) actions.
    Attributes:
        single_action (bool): A flag indicating whether a single action is expected.
    Methods:
        __init__(single_action=True):
            Initializes the CSGOActionHandler with the specified single_action flag.
        _creat_csgo_action(actions: list[CSGOActionEnum]) -> CSGOAction:
            Creates a CSGOAction object from a list of action enums.
        encode_action(action: CSGOActionEnum | list[CSGOActionEnum]) -> torch.Tensor:
            Encodes a CSGO action or list of actions into a tensor.
        decode_action(action: torch.Tensor) -> CSGOActionEnum:
            Decodes a CSGO action tensor into an action enum.
        get_reverse_action(action: CSGOActionEnum | list[CSGOActionEnum]) -> CSGOActionEnum:
            Gets the reverse of a given CSGO action or list of actions.
    """

    def __init__(
        self,
        action_size: int | None = None,
        single_action: bool = True,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(action_size, device)
        self.action_size = len(CSGOActionEnum) if action_size is None else action_size
        self.single_action = single_action
        CSGOActionEnum.init_enum_class()

    def _create_csgo_action(self, actions: list[CSGOActionEnum]) -> CSGOAction:
        """Create a CSGOAction object from an action enum."""

        keys = []
        mouse_x = 0
        mouse_y = 0
        l_click = False
        r_click = False

        for action in actions:
            if action in CSGOActionEnum.KEYS():
                keys.append(action.value)
            elif action in CSGOActionEnum.MOUSE_X():
                mouse_x = action.value
            elif action in CSGOActionEnum.MOUSE_Y():
                mouse_y = action.value
            elif action in CSGOActionEnum.MOUSE_CLICKS():
                if action == CSGOActionEnum.LEFT_CLICK:
                    l_click = True
                elif action == CSGOActionEnum.RIGHT_CLICK:
                    r_click = True

        return CSGOAction(
            keys=keys,
            mouse_x=mouse_x,
            mouse_y=mouse_y,
            l_click=l_click,
            r_click=r_click,
        )

    def encode_action(self, action: CSGOActionEnum | list[CSGOActionEnum]) -> torch.Tensor:
        """Encode a CSGO action into a tensor."""
        assert not (self.single_action and isinstance(action, list)), "Single action expected."
        action = [action] if not isinstance(action, list) else action
        csgo_action = self._create_csgo_action(action)
        return encode_csgo_action(csgo_action, self.device)

    def decode_action(self, action: torch.Tensor) -> CSGOActionEnum:
        """Decode a CSGO action tensor into an action enum."""
        csgo_action = decode_csgo_action(action)

        if self.single_action:
            num_actions = 0
            num_actions += len(csgo_action.keys)
            num_actions += int(csgo_action.l_click)
            num_actions += int(csgo_action.r_click)
            num_actions += csgo_action.mouse_x
            num_actions += csgo_action.mouse_y

            assert num_actions == 1, "Single action expected."

        actions = []
        for key in csgo_action.keys:
            actions.append(CSGOActionEnum(key))
        if csgo_action.l_click:
            actions.append(CSGOActionEnum.LEFT_CLICK)
        if csgo_action.r_click:
            actions.append(CSGOActionEnum.RIGHT_CLICK)
        if csgo_action.mouse_x != 0:
            actions.append(CSGOActionEnum(csgo_action.mouse_x))
        if csgo_action.mouse_y != 0:
            actions.append(CSGOActionEnum(csgo_action.mouse_y))

        if self.single_action:
            return actions[0]
        return actions

    def get_reverse_action(self, action: CSGOActionEnum | list[CSGOActionEnum]) -> CSGOActionEnum:
        assert not (self.single_action and isinstance(action, list)), "Single action expected."

        action = [action] if not isinstance(action, list) else action

        reverse_actions = []
        for act in action:
            reverse_actions.append(CSGOActionEnum.get_reverse_action(act))

        if self.single_action:
            return reverse_actions[0]
        return reverse_actions


class ReverseCSGOActionEnum(BaseActionEnum):
    """CSGO-specific action enum for reverse actions."""

    # Movement keys
    MOVE_FORWARD = "w"
    MOVE_BACKWARD = "s"
    MOVE_LEFT = "a"
    MOVE_RIGHT = "d"
    JUMP = "space"
    CROUCH = "left ctrl"
    WALK = "left shift"
    REVERSE_JUMP = "space"
    REVERSE_CROUCH = "left ctrl"
    REVERSE_WALK = "left shift"

    # Weapon actions
    WEAPON_1 = "1"
    WEAPON_2 = "2"
    WEAPON_3 = "3"
    RELOAD = "r"
    REVERSE_WEAPON_1 = "1"
    REVERSE_WEAPON_2 = "2"
    REVERSE_WEAPON_3 = "3"
    REVERSE_RELOAD = "r"

    # Mouse actions
    LEFT_CLICK = "l_click"
    RIGHT_CLICK = "r_click"
    REVERSE_LEFT_CLICK = "l_click"
    REVERSE_RIGHT_CLICK = "r_click"

    # Mouse X-axis movements
    TURN_LEFT_1000 = -1000
    TURN_LEFT_500 = -500
    TURN_LEFT_300 = -300
    TURN_LEFT_200 = -200
    TURN_LEFT_100 = -100
    TURN_LEFT_60 = -60
    TURN_LEFT_30 = -30
    TURN_LEFT_20 = -20
    TURN_LEFT_10 = -10
    TURN_LEFT_4 = -4
    TURN_LEFT_2 = -2
    MOUSE_X_ZERO = 0
    TURN_RIGHT_2 = 2
    TURN_RIGHT_4 = 4
    TURN_RIGHT_10 = 10
    TURN_RIGHT_20 = 20
    TURN_RIGHT_30 = 30
    TURN_RIGHT_60 = 60
    TURN_RIGHT_100 = 100
    TURN_RIGHT_200 = 200
    TURN_RIGHT_300 = 300
    TURN_RIGHT_500 = 500
    TURN_RIGHT_1000 = 1000

    # Mouse Y-axis movements
    LOOK_DOWN_200 = -200
    LOOK_DOWN_100 = -100
    LOOK_DOWN_50 = -50
    LOOK_DOWN_20 = -20
    LOOK_DOWN_10 = -10
    LOOK_DOWN_4 = -4
    LOOK_DOWN_2 = -2
    MOUSE_Y_ZERO = 0
    LOOK_UP_2 = 2
    LOOK_UP_4 = 4
    LOOK_UP_10 = 10
    LOOK_UP_20 = 20
    LOOK_UP_50 = 50
    LOOK_UP_100 = 100
    LOOK_UP_200 = 200

    @classmethod
    def init_enum_class(cls) -> None:
        members = CSGOActionEnum.__dict__.items()
        for name, member in members:
            if isinstance(member, CSGOActionEnum):
                setattr(cls, name, member)

    @classmethod
    def KEYS(cls) -> list[ReverseCSGOActionHandler]:
        return [
            cls.MOVE_FORWARD,
            cls.MOVE_BACKWARD,
            cls.MOVE_LEFT,
            cls.MOVE_RIGHT,
            cls.JUMP,
            cls.CROUCH,
            cls.WALK,
            cls.WEAPON_1,
            cls.WEAPON_2,
            cls.WEAPON_3,
            cls.RELOAD,
            cls.REVERSE_JUMP,
            cls.REVERSE_CROUCH,
            cls.REVERSE_WALK,
            cls.REVERSE_WEAPON_1,
            cls.REVERSE_WEAPON_2,
            cls.REVERSE_WEAPON_3,
            cls.REVERSE_RELOAD,
        ]

    @classmethod
    def MOUSE_X(cls) -> list[ReverseCSGOActionEnum]:
        return [
            cls.TURN_LEFT_1000,
            cls.TURN_LEFT_500,
            cls.TURN_LEFT_300,
            cls.TURN_LEFT_200,
            cls.TURN_LEFT_100,
            cls.TURN_LEFT_60,
            cls.TURN_LEFT_30,
            cls.TURN_LEFT_20,
            cls.TURN_LEFT_10,
            cls.TURN_LEFT_4,
            cls.TURN_LEFT_2,
            cls.MOUSE_X_ZERO,
            cls.TURN_RIGHT_2,
            cls.TURN_RIGHT_4,
            cls.TURN_RIGHT_10,
            cls.TURN_RIGHT_20,
            cls.TURN_RIGHT_30,
            cls.TURN_RIGHT_60,
            cls.TURN_RIGHT_100,
            cls.TURN_RIGHT_200,
            cls.TURN_RIGHT_300,
            cls.TURN_RIGHT_500,
            cls.TURN_RIGHT_1000,
        ]

    @classmethod
    def MOUSE_Y(cls) -> list[ReverseCSGOActionEnum]:
        return [
            cls.LOOK_DOWN_200,
            cls.LOOK_DOWN_100,
            cls.LOOK_DOWN_50,
            cls.LOOK_DOWN_20,
            cls.LOOK_DOWN_10,
            cls.LOOK_DOWN_4,
            cls.LOOK_DOWN_2,
            cls.MOUSE_Y_ZERO,
            cls.LOOK_UP_2,
            cls.LOOK_UP_4,
            cls.LOOK_UP_10,
            cls.LOOK_UP_20,
            cls.LOOK_UP_50,
            cls.LOOK_UP_100,
            cls.LOOK_UP_200,
        ]

    @classmethod
    def MOUSE_CLICKS(cls) -> list[ReverseCSGOActionEnum]:
        return [
            cls.LEFT_CLICK,
            cls.RIGHT_CLICK,
            cls.REVERSE_LEFT_CLICK,
            cls.REVERSE_RIGHT_CLICK,
        ]

    @staticmethod
    def get_reverse_action(action: ReverseCSGOActionEnum) -> ReverseCSGOActionEnum:
        """Return the reverse of an action if applicable."""

        reverse_action = CSGOActionEnum.get_reverse_action(action)
        if reverse_action:
            return reverse_action

        reverse_action_map = {
            ReverseCSGOActionEnum.MOVE_FORWARD: ReverseCSGOActionEnum.MOVE_BACKWARD,
            ReverseCSGOActionEnum.MOVE_BACKWARD: ReverseCSGOActionEnum.MOVE_FORWARD,
            ReverseCSGOActionEnum.MOVE_LEFT: ReverseCSGOActionEnum.MOVE_RIGHT,
            ReverseCSGOActionEnum.MOVE_RIGHT: ReverseCSGOActionEnum.MOVE_LEFT,
            ReverseCSGOActionEnum.TURN_RIGHT_1000: ReverseCSGOActionEnum.TURN_LEFT_1000,
            ReverseCSGOActionEnum.TURN_RIGHT_500: ReverseCSGOActionEnum.TURN_LEFT_500,
            ReverseCSGOActionEnum.TURN_RIGHT_300: ReverseCSGOActionEnum.TURN_LEFT_300,
            ReverseCSGOActionEnum.TURN_RIGHT_200: ReverseCSGOActionEnum.TURN_LEFT_200,
            ReverseCSGOActionEnum.TURN_RIGHT_100: ReverseCSGOActionEnum.TURN_LEFT_100,
            ReverseCSGOActionEnum.TURN_RIGHT_60: ReverseCSGOActionEnum.TURN_LEFT_60,
            ReverseCSGOActionEnum.TURN_RIGHT_30: ReverseCSGOActionEnum.TURN_LEFT_30,
            ReverseCSGOActionEnum.TURN_RIGHT_20: ReverseCSGOActionEnum.TURN_LEFT_20,
            ReverseCSGOActionEnum.TURN_RIGHT_10: ReverseCSGOActionEnum.TURN_LEFT_10,
            ReverseCSGOActionEnum.TURN_RIGHT_4: ReverseCSGOActionEnum.TURN_LEFT_4,
            ReverseCSGOActionEnum.TURN_RIGHT_2: ReverseCSGOActionEnum.TURN_LEFT_2,
            ReverseCSGOActionEnum.LOOK_UP_200: ReverseCSGOActionEnum.LOOK_DOWN_200,
            ReverseCSGOActionEnum.LOOK_UP_100: ReverseCSGOActionEnum.LOOK_DOWN_100,
            ReverseCSGOActionEnum.LOOK_UP_50: ReverseCSGOActionEnum.LOOK_DOWN_50,
            ReverseCSGOActionEnum.LOOK_UP_20: ReverseCSGOActionEnum.LOOK_DOWN_20,
            ReverseCSGOActionEnum.LOOK_UP_10: ReverseCSGOActionEnum.LOOK_DOWN_10,
            ReverseCSGOActionEnum.LOOK_UP_4: ReverseCSGOActionEnum.LOOK_DOWN_4,
            ReverseCSGOActionEnum.LOOK_UP_2: ReverseCSGOActionEnum.LOOK_DOWN_2,
            ReverseCSGOActionEnum.JUMP: ReverseCSGOActionEnum.REVERSE_JUMP,
            ReverseCSGOActionEnum.CROUCH: ReverseCSGOActionEnum.REVERSE_CROUCH,
            ReverseCSGOActionEnum.WALK: ReverseCSGOActionEnum.REVERSE_WALK,
            ReverseCSGOActionEnum.WEAPON_1: ReverseCSGOActionEnum.REVERSE_WEAPON_1,
            ReverseCSGOActionEnum.WEAPON_2: ReverseCSGOActionEnum.REVERSE_WEAPON_2,
            ReverseCSGOActionEnum.WEAPON_3: ReverseCSGOActionEnum.REVERSE_WEAPON_3,
            ReverseCSGOActionEnum.RELOAD: ReverseCSGOActionEnum.REVERSE_RELOAD,
            ReverseCSGOActionEnum.LEFT_CLICK: ReverseCSGOActionEnum.REVERSE_LEFT_CLICK,
            ReverseCSGOActionEnum.RIGHT_CLICK: ReverseCSGOActionEnum.REVERSE_RIGHT_CLICK,
        }

        reverse_action_map.update({value: key for key, value in reverse_action_map.items()})

        return reverse_action


class ReverseCSGOActionHandler(CSGOActionHandler):
    def __init__(
        self,
        action_size: int | None = None,
        single_action: bool = True,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(action_size, single_action, device)
        self.action_size = len(ReverseCSGOActionEnum)  # if action_size is None else action_size
        ReverseCSGOActionEnum.init_enum_class()

    def _create_csgo_action(self, actions: list[ReverseCSGOActionEnum]) -> ReverseCSGOAction:
        """Create a CSGOAction object from an action enum."""

        keys = []
        mouse_x = 0
        mouse_y = 0
        l_click = False
        r_click = False
        reverse_l_click = False
        reverse_r_click = False

        for action in actions:
            if action in ReverseCSGOActionEnum.KEYS():
                keys.append(action.value)
            elif action in ReverseCSGOActionEnum.MOUSE_X():
                mouse_x = action.value
            elif action in ReverseCSGOActionEnum.MOUSE_Y():
                mouse_y = action.value
            elif action in ReverseCSGOActionEnum.MOUSE_CLICKS():
                if action == ReverseCSGOActionEnum.LEFT_CLICK:
                    l_click = True
                elif action == ReverseCSGOActionEnum.RIGHT_CLICK:
                    r_click = True
                elif action == ReverseCSGOActionEnum.REVERSE_LEFT_CLICK:
                    reverse_l_click = True
                elif action == ReverseCSGOActionEnum.REVERSE_RIGHT_CLICK:
                    reverse_r_click = True

        return ReverseCSGOAction(
            keys=keys,
            mouse_x=mouse_x,
            mouse_y=mouse_y,
            l_click=l_click,
            r_click=r_click,
            reverse_l_click=reverse_l_click,
            reverse_r_click=reverse_r_click,
        )

    def encode_action(
        self,
        action: ReverseCSGOActionEnum | list[ReverseCSGOActionEnum],
    ) -> torch.Tensor:
        """Encode a CSGO action into a tensor."""
        assert not (self.single_action and isinstance(action, list)), "Single action expected."
        action = [action] if not isinstance(action, list) else action
        csgo_action = self._create_csgo_action(action)
        return encode_reverse_csgo_action(csgo_action, self.device)

    def decode_action(self, action: torch.Tensor) -> ReverseCSGOActionEnum:
        """Decode a CSGO action tensor into an action enum."""
        csgo_action = decode_reverse_csgo_action(action)

        if self.single_action:
            num_actions = 0
            num_actions += len(csgo_action.keys)
            num_actions += int(csgo_action.l_click)
            num_actions += int(csgo_action.r_click)
            num_actions += int(csgo_action.reverse_l_click)
            num_actions += int(csgo_action.reverse_r_click)
            num_actions += csgo_action.mouse_x
            num_actions += csgo_action.mouse_y

            assert num_actions == 1, "Single action expected."

        actions = []
        for key in csgo_action.keys:
            actions.append(ReverseCSGOActionEnum(key))
        if csgo_action.l_click:
            actions.append(ReverseCSGOActionEnum.LEFT_CLICK)
        if csgo_action.r_click:
            actions.append(ReverseCSGOActionEnum.RIGHT_CLICK)
        if csgo_action.reverse_l_click:
            actions.append(ReverseCSGOActionEnum.REVERSE_LEFT_CLICK)
        if csgo_action.reverse_r_click:
            actions.append(ReverseCSGOActionEnum.REVERSE_RIGHT_CLICK)
        if csgo_action.mouse_x != 0:
            actions.append(ReverseCSGOActionEnum(csgo_action.mouse_x))
        if csgo_action.mouse_y != 0:
            actions.append(ReverseCSGOActionEnum(csgo_action.mouse_y))

        if self.single_action:
            return actions[0]
        return actions

    def get_reverse_action(
        self,
        action: ReverseCSGOActionEnum | list[ReverseCSGOActionEnum],
    ) -> ReverseCSGOActionEnum:
        assert not (self.single_action and isinstance(action, list)), "Single action expected."

        action = [action] if not isinstance(action, list) else action

        reverse_actions = []
        for act in action:
            reverse_actions.append(ReverseCSGOActionEnum.get_reverse_action(act))

        if self.single_action:
            return reverse_actions[0]
        return reverse_actions
