import os

from enum import Enum

from run_mode import mode

# base

PROJECT_DIR = os.path.basename(__file__)

# path
RESOURCE_DIR = os.path.join(PROJECT_DIR, "resource", mode.name)
CDR_DIR = os.path.join(RESOURCE_DIR, "cdr")
PROPERTY_DIR = os.path.join(RESOURCE_DIR, "property")


# Initial FeatureDict

class CdrDict(Enum):
    CALLING, \
    CALLED, \
    CHARGE_CALLING, \
    START_TIME, \
    CALL_TIME, \
    COST, \
    CDR_TYPE, \
    CALL_TYPE, \
    TALK_TYPE, \
    CALLING_AREA, \
    CALLED_AREA \
        = range(11)


class PropertyDict(Enum):
    CALLING, \
    USER_NAME, \
    PLAN_NAME, \
    CUST_CODE, \
    USER_TYPE, \
    OPEN_DATE, \
    SUB_STAT_TP, \
    IS_REAL_NAME, \
    SELL_PRODUCT, \
        = range(9)
