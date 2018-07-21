# coding=utf-8
import glob
import logging
import os

import conf
from src.common.util import loadFormatDict
from src.moduler.moduler import Moduler


# TODO(20180630) monitor performance
class TargetBehaviorConstructor(Moduler):
    def __init__(
            self,
            featureFrame3dDir=None, ffFirstRowFmtFileName=None,
            targetBehaviorDir=None,
    ):
        super(TargetBehaviorConstructor, self).__init__()
        self.featureFrame3dDir = featureFrame3dDir
        self.ffFirstRowFmtFilePath = os.path.join(self.featureFrame3dDir, ffFirstRowFmtFileName)
        self.targetBehaviorDir = targetBehaviorDir

        self.__ffFirstRowFmtDict = None

    def run(self):
        os.mkdir(self.targetBehaviorDir)

        logging.info("start to construct target behavior: %s" % self.targetBehaviorDir)
        self.__intRun()
        logging.info("output aggregate cdr: %s" % self.targetBehaviorDir)

    # OPT(20180701) parallel
    def __intRun(self):
        logging.debug("load first row format of FeatureFrame: %s" % self.ffFirstRowFmtFilePath)
        self.__ffFirstRowFmtDict = loadFormatDict(self.ffFirstRowFmtFilePath)

        ff3dDirsBy_calling = glob.glob(os.path.join(self.featureFrame3dDir, "*.%s" % conf.KEY_DIR_SUFFIX))
        for ff3dDirBy_calling in ff3dDirsBy_calling:
            # FIXME(20180721) replace rstrip with split as used in parallel_process.py
            calling = os.path.basename(ff3dDirBy_calling).rstrip(".%s" % conf.KEY_DIR_SUFFIX)
            tgtBhvDirBy_calling = os.path.join(self.targetBehaviorDir, "%s.%s" % (calling, conf.KEY_DIR_SUFFIX))
            os.mkdir(tgtBhvDirBy_calling)
            ff3dFilePathsBy_date = glob.glob(os.path.join(ff3dDirBy_calling, "*.%s" % conf.DATA_FILE_SUFFIX))
            for ff3dFilePathBy_date in ff3dFilePathsBy_date:
                date = os.path.basename(ff3dFilePathBy_date).rstrip("*.%s" % conf.DATA_FILE_SUFFIX)
                ffFirstRows = range(24)
                with open(ff3dFilePathBy_date, "r") as rFf3dFile:
                    hour = 0
                    for ffLine in rFf3dFile:
                        ffAsRow = ffLine.strip().split(conf.COL_SEPERATOR)
                        ffFirstRow = ffAsRow[:len(self.__ffFirstRowFmtDict)]
                        ffFirstRows[hour] = ffFirstRow
                        hour += 1
                    assert hour == 24
                # OPT(20180705) compress files cnt if needed
                tgtBhvFilePathBy_date = os.path.join(tgtBhvDirBy_calling, "%s.%s" % (date, conf.DATA_FILE_SUFFIX))
                with open(tgtBhvFilePathBy_date, "w") as wFile:
                    wFile.write(self.__resolveTargetHehavior(ffFirstRows))

    def __resolveTargetHehavior(self, ffFirstRows):
        # TODO(20180705) support other target behavior
        # attention: 由于套餐的业务逻辑，cost通常为0，预测cost无法得到有意义的结果
        sumCallTime = 0.0
        for ffFirstRow in ffFirstRows:
            costBy_hour = float(ffFirstRow[self.__ffFirstRowFmtDict["CALL_CNT"]]) \
                          * float(ffFirstRow[self.__ffFirstRowFmtDict["CALL_TIME_MEAN"]])
            sumCallTime += float(costBy_hour)
        return str(sumCallTime)
