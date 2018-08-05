# coding=utf-8
import logging
import os
import random
import time

from enum import Enum

PROJECT_DIR = os.path.dirname(__file__)


class LogLevel(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARN = logging.WARN
    ERROR = logging.ERROR


logLevel = LogLevel.DEBUG.value
__LOG_FORMAT = '%(levelname)5s %(asctime)s [%(filename)s line:%(lineno)d] %(message)s'
logging.basicConfig(format=__LOG_FORMAT, level=logLevel)


class TfLogLevel(Enum):
    DEBUG = "1"
    WARN = "2"
    ERROR = "3"


os.environ['TF_CPP_MIN_LOG_LEVEL'] = TfLogLevel.ERROR.value

wkdir = os.path.join(PROJECT_DIR, "tmp", str(int(time.time())))

# special file/dir suffix
DATA_FILE_SUFFIX = "data.txt"
FORMAT_FILE_SUFFIX = "format.json"
KEY_DIR_SUFFIX = "calling"
DIR_BY_DATE_SUFFIX = "date"
CACHE_FILE_SUFFIX = "cache.npy"

COL_SEPERATOR = "\t"
ROW_SEPERATOR = "\n"


# Initial FeatureDict
class CdrDict(object):
    SEPERATOR = "|"
    COL_CNT = 11

    class Column(Enum):
        (
            CALLING,  # 设备号
            CALLED,  # 呼叫号码
            CHARGE_CALLING,  # 计费号码（与主叫相同），但不同于于设备号
            START_TIME,  # 起呼时间 　 20161024105427
            CALL_TIME,  # 时长
            COST,  # 费用
            CDR_TYPE,  # 类型
            CALL_TYPE,  # 呼叫类别（C网）。只取"1"（主叫）
            TALK_TYPE,  # 通话类型（C网）
            # TODO(20180701) support other CALLING_AREA
            CALLING_AREA,  # 通话地点（C网）。只取"021"（上海区号）
            CALLED_AREA,  # 对方地点（C网，固话长途）。假设区号无脏数据
        ) = range(11)  # not support using CdrDict.COL_CNT here

    USEFUL_COLS = {
        Column.CALLING.value,
        Column.CALLED.value,
        Column.START_TIME.value,
        Column.CALL_TIME.value,
        Column.COST.value,
        Column.CDR_TYPE.value,
        Column.TALK_TYPE.value,
        Column.CALLED_AREA.value,
    }

    # TODO(20180701) tuning refer to stat in translating
    CALL_TIME_UNIT = 5
    COST_UNIT = 4

    CDR_TYPE_DICT = {
        "cvoi": 0,  # C网语音
        "local": 85,  # 本地固话
        "distance": 170,  # 外地固话
    }

    TALK_TYPE_DICT = {
        "1": 0,  # 本地
        "7": 25,  # 国际长途
        "6": 50,  # 国内长途
        "4": 75,  # 漫游国际
        "8": 100,  # 港澳台长途
        "2": 125,  # 漫游国内
        "3": 150,  # 漫游港澳台
        "9": 175,  # 多方通话本地
        "5": 200,  # 漫游本地通话
        "10": 225,  # 多方通话国内长途
    }


class PropertyDict(object):
    SEPERATOR = "\t"
    COL_CNT = 9

    class Column(Enum):
        (
            CALLING,  # 设备号
            USER_NAME,  # 用户名
            PLAN_NAME,  # 产品名。只取用户数量大于10的
            CUST_CODE,  # 客户标识
            USER_TYPE,  # 用户类别
            OPEN_DATE,  # 开户时间
            SUB_STAT_TP,  # 是否停机。只取"活动"
            IS_REAL_NAME,  # 是否实名制。只取"1"（是）
            SELL_PRODUCT,  # 销售产品
        ) = range(9)  # not support using PropertyDict.COL_CNT here

    USEFUL_COLS = {
        Column.CALLING.value,
        Column.PLAN_NAME.value,
        Column.USER_TYPE.value,
        Column.SELL_PRODUCT.value,
    }

    USER_TYPE_DICT = {
        "政企": 0,
        "公众": 127,
    }

    SELL_PRODUCT_DICT = {
        "CDMA预付费": 0,
        "CDMA后付费": 50,
        "CDMA准实时预付费": 100,
        "C+W（E+W）预付费": 150,
        "C+W（E+W）后付费": 200,
    }

    PLAN_NAME_DICT = {
        "201407乐享4G201407-副卡-后": 0,
        "上海安吉星信息服务有限公司套餐（onstar），20元/月": 7,
        "201407乐享4G201407 169元-主套餐": 14,
        "201407乐享4G201407 199元-主套餐": 21,
        "201407乐享4G201407 129元-主套餐": 28,
        "201407乐享4G201407 59元-主套餐": 35,
        "201407乐享4G201407 99元-主套餐": 42,
        "201605-乐享家201605 副卡": 49,
        "201407乐享4G201407 299元-主套餐": 56,
        "2014热力团购个性化套餐—加密通信，100元/月 （政企）": 63,
        "201505-飞Young 4G聊天版（非定制版）/19元": 70,
        "201609-乐享家201609 199元套餐": 77,
        "2014我的e家全家一起享e9套餐，1749元/年": 84,
        "201609-乐享家201609 129元套餐": 91,
        "201407乐享4G201407 79元-主套餐": 98,
        "2012我的e家e6自主本地版套餐39元/月，2年（原畅聊升级版39元）": 105,
        "201609-乐享家201609 169元套餐": 112,
        "2010固话天翼行20元/月套餐": 119,
        "2011我的e家畅聊升级版套餐29元/月（固话）": 126,
        "2012天翼飞young19元套餐（原2011天翼秋季校园19元套餐）": 133,
        "2014我的e家全家一起享手机加装包5元/月": 140,
        "201407飞Young4G 201407纯流量云卡49元-主套餐": 147,
        "2009宽带天翼行套餐10元/月，1M，市区ADSL（固话）": 154,
        "201407乐享4G201407 399元-主套餐": 161,
        "2010科教文卫行业套餐，25元/月": 168,
        "2009“科技助老卡”套餐，5元/月": 175,
        "2010天翼总机服务移动分机39本地套餐，39元/月（政企）": 182,
        "201607科教文卫套餐，30元/月（政企、后+预）": 189,
        "201609-乐享家201609 299元套餐": 196,
        "2012天翼乐享3G套餐聊天版0元/月（e9自主乐享版共用）（副卡，适用129元及以上档次）": 203,
        "201407乐享4G201407-副卡-预": 210,
        "201509-4G易通卡5元": 217,
        "2011我的e家e6-3G套餐56元/月": 224,
        "2013我的e家e6-3G套餐56元/月": 231,
    }


class AggregateCdrDict(object):
    AGGREGATE_FEATURES = [
        # key
        "CALLING",
        "START_TIME",
        # value
        "CALL_CNT",
        "CALLED_UCNT",  # prefix "u" means unique
        "CALLED_MODE",
        "CALLED_MCNT",  # prefix "m" means mode
        "CALL_TIME_UCNT",
        "CALL_TIME_MEAN",
        "CALL_TIME_MODE",
        "CALL_TIME_MCNT",
        "CALL_TIME_STD",
        "COST_UCNT",
        "COST_MEAN",
        "COST_MODE",
        "COST_MCNT",
        "COST_STD",
        "CDR_TYPE_UCNT",
        "CDR_TYPE_MODE",
        "CDR_TYPE_MCNT",
        "TALK_TYPE_UCNT",
        "TALK_TYPE_MODE",
        "TALK_TYPE_MCNT",
        "CALLED_AREA_UCNT",
        "CALLED_AREA_MODE",
        "CALLED_AREA_MCNT",
    ]
    AGGREGATE_FMT = None

    class AggregateTimeUnit(Enum):
        (
            HOUR_6,
            HOUR_1,
            MIN_10,
        ) = range(3)

    AGGREGATE_TIME_UNIT = AggregateTimeUnit.HOUR_1

    # TODO(20180703) tuning refer to stat in aggregating
    NORMAL_CALL_CNT_RATE = 20 / 3600.0

    @staticmethod
    def init():
        AggregateCdrDict.__initAggregateFmt()

    @staticmethod
    def __initAggregateFmt():
        if AggregateCdrDict.AGGREGATE_FMT is not None:
            return

        aggregateFeatures = AggregateCdrDict.AGGREGATE_FEATURES
        assert len(set(aggregateFeatures)) == len(aggregateFeatures)
        AggregateCdrDict.AGGREGATE_FMT = dict(zip(aggregateFeatures, range(len(aggregateFeatures))))


class FeatureFrame3dDict(object):
    # FIXME(20180705) cannot use 0 as nan-value for CDR_TYPE_MODE, TALK_TYPE_MODE, PLAN_NAME, USER_TYPE and SELL_PRODUCT
    FIRST_ROW_FEATURES = [
        # cdr value
        "CALL_CNT",
        "CALLED_UCNT",  # prefix "u" means unique
        "CALLED_MCNT",  # prefix "m" means mode
        "CALL_TIME_UCNT",
        "CALL_TIME_MEAN",
        "CALL_TIME_MODE",
        "CALL_TIME_MCNT",
        "CALL_TIME_STD",
        "COST_UCNT",
        "COST_MEAN",
        "COST_MODE",
        "COST_MCNT",
        "COST_STD",
        "CDR_TYPE_UCNT",
        "CDR_TYPE_MODE",
        "CDR_TYPE_MCNT",
        "TALK_TYPE_UCNT",
        "TALK_TYPE_MODE",
        "TALK_TYPE_MCNT",
        "CALLED_AREA_UCNT",
        "CALLED_AREA_MCNT",
        # property value
        "PLAN_NAME",
        "USER_TYPE",
        "SELL_PRODUCT",
    ]

    # TODO(20180703) accurate calculate COPY_CNT judging by p
    COPY_CNT = 10

    class DepthTimeUnit(Enum):
        (
            DAY_1,
        ) = range(1)

    DEPTH_TIME_UNIT = DepthTimeUnit.DAY_1
    assert (AggregateCdrDict.AGGREGATE_TIME_UNIT == AggregateCdrDict.AggregateTimeUnit.HOUR_1
            and DEPTH_TIME_UNIT == DepthTimeUnit.DAY_1)
    # TODO(20180721) calculate by AGGREGATE_TIME_UNIT and DEPTH_TIME_UNIT
    DEPTH_FF2D_CNT = 24

    FF3D_WIDTH = len(FIRST_ROW_FEATURES)
    FF3D_HEIGHT = COPY_CNT
    FF3D_DEPTH = DEPTH_FF2D_CNT
    FF3D_SHAPE = [FF3D_DEPTH, FF3D_HEIGHT, FF3D_WIDTH]

    FIRST_ROW_FMT = None
    SHUFFLE_FMT = None

    @staticmethod
    def init():
        FeatureFrame3dDict.__initFirstRowFmt()
        FeatureFrame3dDict.__initShuffleFmt()

    @staticmethod
    def __initFirstRowFmt():
        if FeatureFrame3dDict.FIRST_ROW_FMT is not None:
            return

        firstRowFeatures = FeatureFrame3dDict.FIRST_ROW_FEATURES
        assert len(set(firstRowFeatures)) == len(firstRowFeatures)
        # assert set(firstRowFeatures) < (set(aggCdrFmtDict) | set(tlPptFmtDict))
        FeatureFrame3dDict.FIRST_ROW_FMT = dict(zip(firstRowFeatures, range(len(firstRowFeatures))))

    @staticmethod
    def __initShuffleFmt():
        if FeatureFrame3dDict.SHUFFLE_FMT is not None:
            return

        shuffleFmt = range(FeatureFrame3dDict.COPY_CNT)
        firstRowOrder = range(len(FeatureFrame3dDict.FIRST_ROW_FMT))
        shuffleFmt[0] = firstRowOrder
        tmpRowOrder = list(firstRowOrder)
        for rowNo in range(1, FeatureFrame3dDict.COPY_CNT):
            random.shuffle(tmpRowOrder)
            shuffleFmt[rowNo] = list(tmpRowOrder)
        FeatureFrame3dDict.SHUFFLE_FMT = shuffleFmt


class FeatureVectorDict(object):
    FEATURE_VECTOR_LEN = 64


class TrainerDict(object):
    LEARN_DAY_CNT = 7
