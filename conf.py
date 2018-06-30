# coding=utf-8
import os

from enum import Enum

from base_conf import mode, PROJECT_DIR

# path
# TODO(20180630) check whether paths exit
RESOURCE_DIR = os.path.join(PROJECT_DIR, "resource", mode.name)
CDR_DIR = os.path.join(RESOURCE_DIR, "cdr")
PROPERTY_DIR = os.path.join(RESOURCE_DIR, "property")


# Initial FeatureDict

class CdrDict(object):
    SEPERATOR = "|"
    COL_CNT = 11

    class Column(Enum):
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
            = range(CdrDict.COL_CNT)

    CDR_TYPE_DICT = {
        'cvoi': '0',
        'local': '1',
        'distance': '2'
    }


class PropertyDict(object):
    SEPERATOR = "\t"
    COL_CNT = 9

    class Column(Enum):
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

    USER_TYPE_DICT = {
        u'政企': 0,
        u'公众': 100
    }

    SUB_STAT_TP_DICT = {
        'BLANK': 0,
        u'活动': 40,
        u'停机': 80,
        u'拆机': 120,
        u'帐务停机': 160,
        u'割接': 200
    }

    SELL_PRODUCT_DICT = {
        u'CDMA预付费': 0,
        u'CDMA后付费': 30,
        u'CDMA准实时预付费': 60,
        u'C+W（E+W）预付费': 90,
        u'C+W（E+W）后付费': 120,
        u'C+W（E+W）准实时预付费': 150,
        u'其他': 180
    }
