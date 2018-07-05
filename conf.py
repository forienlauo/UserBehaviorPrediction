# coding=utf-8
import os

from enum import Enum

from base_conf import wkdir as WKDIR, FORMAT_FILE_SUFFIX

# middle-state data path
# cleaner
CLEAN_CDR_DIR = os.path.join(WKDIR, "clean", "cdr")
CLEAN_PPT_DIR = os.path.join(WKDIR, "clean", "property")
DIRTY_CDR_DIR = os.path.join(WKDIR, "dirty", "cdr")
DIRTY_PPT_DIR = os.path.join(WKDIR, "dirty", "property")
# translator
TRANSLATE_CDR_DIR = os.path.join(WKDIR, "translate", "cdr")
TRANSLATE_PPT_DIR = os.path.join(WKDIR, "translate", "property")
# aggregator
AGGREGATE_CDR_DIR = os.path.join(WKDIR, "aggregate")
# feature frame 3d constructor
FEATURE_FRAME_3D_DIR = os.path.join(WKDIR, "featureFrame3d")
# target behavior constructor
TARGET_BEHAVIOR_DIR = os.path.join(WKDIR, "targetBehavior")

# special file name
CLEAN_CDR_FORMAT_FILE = "__cdr.%s" % FORMAT_FILE_SUFFIX
CLEAN_PPT_FORMAT_FILE = "__property.%s" % FORMAT_FILE_SUFFIX
TL_CDR_FORMAT_FILE = "__cdr.%s" % FORMAT_FILE_SUFFIX
TL_PPT_FORMAT_FILE = "__property.%s" % FORMAT_FILE_SUFFIX
AGG_CDR_FORMAT_FILE = "__cdr.%s" % FORMAT_FILE_SUFFIX
SHUFFLE_FORMAT_FILE = "__shuffle.%s" % FORMAT_FILE_SUFFIX
FF_FIRST_ROW_FORMAT_FILE = "__ffFirstRow.%s" % FORMAT_FILE_SUFFIX

COL_SEPERATOR = "\t"
ROW_SEPERATOR = "\n"


class AggregateTimeUnit(Enum):
    (
        HOUR_6,
        HOUR_1,
        MIN_10,
    ) = range(3)


# TODO(20180703) tuning refer to stat in aggregating
NORMAL_CALL_RATE = 20 / 3600.0


class FeatureFrame3dDict(object):
    # TODO(20180703) accurate calculate COPY_CNT judging by p
    COPY_CNT = 10
