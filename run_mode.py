from enum import Enum


class Mode(Enum):
    DEV, TEST, PROD = range(3)


mode = Mode.DEV
