# coding=utf-8
import logging
import os
import time

from enum import Enum

PROJECT_DIR = os.path.dirname(__file__)


class Mode(Enum):
    DEV, TEST, PROD = range(3)


class LogLevel(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARN = logging.WARN
    ERROR = logging.ERROR


mode = Mode.DEV

logLevel = LogLevel.DEBUG
__LOG_FORMAT = '%(levelname)5s %(asctime)s [%(filename)s line:%(lineno)d] %(message)s'
logging.basicConfig(format=__LOG_FORMAT, level=logLevel)

wkdir = os.path.join(PROJECT_DIR, "tmp", int(time.time()))
