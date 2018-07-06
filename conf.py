# coding=utf-8
import os

from base_conf import wkdir as WKDIR, FORMAT_FILE_SUFFIX

# REFACTOR(20180706) extract conf about wkdir
# REFACTOR(20180706) remove conf, and rename bconf to conf
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

# TODO(20180703) tuning refer to stat in aggregating
NORMAL_CALL_RATE = 20 / 3600.0
