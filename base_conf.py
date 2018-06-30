import os
import time

from enum import Enum

PROJECT_DIR = os.path.basename(__file__)


class Mode(Enum):
    DEV, TEST, PROD = range(3)


mode = Mode.DEV
wkdir = os.path.join(PROJECT_DIR, "tmp", int(time.time()))
