# coding=utf-8
import glob
import json
import logging
import os
import random
import shutil

import conf
from src.common.usr_exceptions import NotSupportedError
from src.common.util import loadFormatDict, dumpFormatDict
from src.moduler.moduler import Moduler, Stat


# TODO(20180630) monitor performance
class FeatureFrame3dConstructor(Moduler):
    def __init__(
            self,
            aggregateCdrDir=None, aggCdrFmtFileName=None,
            aggregateTimeUnit=None,
            translatePropertyDir=None, tlPptFmtFileName=None,
            featureFrame3dDir=None, ffFmtFileName=None, ffFirstRowFmtFileName=None,
    ):
        super(FeatureFrame3dConstructor, self).__init__()
        self.aggregateCdrDir = aggregateCdrDir
        self.aggCdrFmtFilePath = os.path.join(self.aggregateCdrDir, aggCdrFmtFileName)
        self.aggregateTimeUnit = aggregateTimeUnit
        self.translatePropertyDir = translatePropertyDir
        self.tlPptFmtFilePath = os.path.join(self.translatePropertyDir, tlPptFmtFileName)
        self.featureFrame3dDir = featureFrame3dDir
        self.ff3TmpDir = os.path.join(self.featureFrame3dDir, ".tmp")
        self.ffFmtFilePath = os.path.join(self.featureFrame3dDir, ffFmtFileName)
        self.ffFirstRowFmtFilePath = os.path.join(self.featureFrame3dDir, ffFirstRowFmtFileName)
        self.stat = None
        self.__aggCdrFmtDict = None
        self.__tlPptFmtDict = None
        self.__ffFirstRowFmtDict = None
        self.__ffFmtDict = None
        self.__shuffleFmt = None
        self.__sharedZeroFfAsLine = None

        if self.aggregateTimeUnit != conf.AggregateTimeUnit.HOUR_1:
            raise NotSupportedError("Not supported aggregateTimeUnit: %s" % self.aggregateTimeUnit.name)

    def run(self):
        self.__init()

        logging.info("start to construct FeatureFrame3D with aggregate cdr: %s, translate property: %s"
                     % (self.aggregateCdrDir, self.translatePropertyDir,))
        self.__intRun()
        logging.info("output FeatureFrame3D: %s" % self.featureFrame3dDir)

        self.__clean()

    def __init(self):
        os.makedirs(self.featureFrame3dDir)
        if os.path.isdir(self.ff3TmpDir):
            shutil.rmtree(self.ff3TmpDir)
        os.makedirs(self.ff3TmpDir)

    def __clean(self):
        shutil.rmtree(self.ff3TmpDir)

    # OPT(20180701) parallel
    def __intRun(self):
        logging.debug("load translate property format: %s" % self.tlPptFmtFilePath)
        self.__tlPptFmtDict = loadFormatDict(self.tlPptFmtFilePath)
        logging.debug("load aggregate cdr format: %s" % self.aggCdrFmtFilePath)
        self.__aggCdrFmtDict = loadFormatDict(self.aggCdrFmtFilePath)

        self.stat = _FeatureFrame3dStat()

        joinDir = os.path.join(self.ff3TmpDir, "join")
        os.mkdir(joinDir)
        self.__joinAggCdrAndPpt(self.aggregateCdrDir, self.aggregateTimeUnit, self.translatePropertyDir, joinDir)
        logging.debug("dump FeatureFrame first row format: %s" % self.ffFirstRowFmtFilePath)
        dumpFormatDict(self.__ffFirstRowFmtDict, self.ffFirstRowFmtFilePath)

        # TODO(20180703) copy && shuffle
        featureFrameDir = os.path.join(self.ff3TmpDir, "featureFrame")
        os.mkdir(featureFrameDir)
        self.__copyAndShuffle(joinDir, featureFrameDir)
        logging.debug("dump FeatureFrame format: %s" % self.ffFmtFilePath)
        dumpFormatDict(self.__ffFmtDict, self.ffFmtFilePath)

        # TODO(20180703) construct
        constructDir = os.path.join(self.ff3TmpDir, "construct")
        os.mkdir(constructDir)
        self.__constructFeatureFrame3d(featureFrameDir, constructDir)

        logging.debug("move construct FeatureFrame3D dir: %s" % constructDir)
        constructFf3DirsBy_uid = glob.glob(os.path.join(constructDir, "*"))
        for constructFf3DirBy_uid in constructFf3DirsBy_uid:
            os.rename(constructFf3DirBy_uid,
                      os.path.join(self.featureFrame3dDir, os.path.basename(constructFf3DirBy_uid)))

        logging.debug("%s stat: %s" % (self.name, self.stat,))

    def __joinAggCdrAndPpt(self, aggregateCdrDir, aggregateTimeUnit, translatePropertyDir, joinDir):
        aggCdrFmtDict = self.__aggCdrFmtDict
        tlPptFmtDict = self.__tlPptFmtDict
        # FIXME(20180705) cannot use 0 as nan value for CDR_TYPE_MODE, TALK_TYPE_MODE, PLAN_NAME, USER_TYPE and SELL_PRODUCT
        ffFirstRowFeatures = [
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
        assert len(set(ffFirstRowFeatures)) == len(ffFirstRowFeatures)
        assert set(ffFirstRowFeatures) < (set(aggCdrFmtDict) | set(tlPptFmtDict))
        ffFirstRowFmtDict = self.__ffFirstRowFmtDict = dict(zip(ffFirstRowFeatures, range(len(ffFirstRowFeatures))))

        tlPptFilePaths = glob.glob(os.path.join(translatePropertyDir, "*.%s" % conf.DATA_FILE_SUFFIX))
        for tlPptFilePath in tlPptFilePaths:
            with open(tlPptFilePath, "r") as rTlPptFile:
                for tlPptLine in rTlPptFile:
                    tlPptCols = tlPptLine.strip().split(conf.COL_SEPERATOR)
                    calling = tlPptCols[tlPptFmtDict["CALLING"]]
                    joinDictBy_date = dict()
                    aggCdrFilePath = os.path.join(
                        aggregateCdrDir, aggregateTimeUnit.name, "%s.%s" % (calling, conf.DATA_FILE_SUFFIX))
                    if not os.path.isfile(aggCdrFilePath):
                        self.stat.noCdrCallingCnt.append(calling)
                        continue
                    with open(aggCdrFilePath, "r") as rAggCdrFile:
                        for aggCdrLine in rAggCdrFile:
                            aggCdrCols = aggCdrLine.strip().split(conf.COL_SEPERATOR)
                            startTime = aggCdrCols[aggCdrFmtDict["START_TIME"]]
                            assert self.aggregateTimeUnit == conf.AggregateTimeUnit.HOUR_1 and len(startTime) == 10
                            date = startTime[:8]
                            hour = startTime[8:10]
                            joinCols = range(len(ffFirstRowFeatures))
                            for ffFirstRowFeature in ffFirstRowFeatures:
                                joinCols[ffFirstRowFmtDict[ffFirstRowFeature]] = \
                                    self.__getFeatureValue(ffFirstRowFeature, aggCdrCols, tlPptCols)
                            if date not in joinDictBy_date:
                                joinDictBy_date[date] = dict()
                                # joinLinesBy_hour = joinDictBy_date[date]
                            assert hour not in joinDictBy_date[date]
                            joinDictBy_date[date][hour] = conf.COL_SEPERATOR.join(joinCols)
                    joinDirBy_calling = os.path.join(joinDir, calling)
                    os.mkdir(joinDirBy_calling)
                    for date in joinDictBy_date:
                        joinFilePathBy_date = os.path.join(
                            joinDirBy_calling, "%s.%s" % (date, conf.DATA_FILE_SUFFIX))
                        joinLinesBy_hour = joinDictBy_date[date]
                        with open(joinFilePathBy_date, "w") as wJoinFile:
                            sortedHours = sorted(joinLinesBy_hour.keys())
                            # file header
                            wJoinFile.write(json.dumps(sortedHours))
                            wJoinFile.write("\n")
                            # file body(sorted by hour)
                            wJoinFile.write(conf.ROW_SEPERATOR.join([joinLinesBy_hour[hour] for hour in sortedHours]))

    def __getFeatureValue(self, featureName, aggCdrCols, tlPptCols):
        if featureName in self.__tlPptFmtDict:
            return tlPptCols[self.__tlPptFmtDict[featureName]]
        else:
            assert featureName in self.__aggCdrFmtDict
            return aggCdrCols[self.__aggCdrFmtDict[featureName]]

    def __copyAndShuffle(self, joinDir, featureFrameDir):
        self.__initShuffleFmt()
        shuffleFmt = self.__shuffleFmt

        joinDirsBy_calling = glob.glob(os.path.join(joinDir, "*"))
        for joinDirBy_calling in joinDirsBy_calling:
            calling = os.path.basename(joinDirBy_calling)
            ffDirBy_calling = os.path.join(featureFrameDir, calling)
            os.mkdir(ffDirBy_calling)
            joinFilePathsBy_date = glob.glob(os.path.join(joinDirBy_calling, "*.%s" % conf.DATA_FILE_SUFFIX))
            for joinFilePathBy_date in joinFilePathsBy_date:
                date = os.path.basename(joinFilePathBy_date).rstrip("*.%s" % conf.DATA_FILE_SUFFIX)
                featureFramesAsLines = list()
                with open(joinFilePathBy_date, "r") as rJoinFile:
                    sortedHoursAsLine = rJoinFile.readline()
                    for joinLine in rJoinFile:
                        firstRow = joinLine.strip().split(conf.COL_SEPERATOR)
                        featureFrameAsRow = list()
                        for shuffleRowFmt in shuffleFmt:
                            for colNoInFirstRow in shuffleRowFmt:
                                featureFrameAsRow.append(firstRow[colNoInFirstRow])
                        featureFramesAsLines.append(conf.COL_SEPERATOR.join(featureFrameAsRow))
                ffFilePathBy_date = os.path.join(ffDirBy_calling, "%s.%s" % (date, conf.DATA_FILE_SUFFIX))
                with open(ffFilePathBy_date, "w") as wFile:
                    # file header
                    wFile.write(sortedHoursAsLine)
                    # file body(sorted by hour)
                    wFile.write(conf.ROW_SEPERATOR.join(featureFramesAsLines))

    def __initShuffleFmt(self):
        if self.__shuffleFmt is not None:
            return
        self.__shuffleFmt = range(conf.FeatureFrame3dDict.COPY_CNT)
        firstRowOrder = range(len(self.__ffFirstRowFmtDict))
        self.__shuffleFmt[0] = firstRowOrder
        tmpRowOrder = list(firstRowOrder)
        for rowNo in range(1, conf.FeatureFrame3dDict.COPY_CNT):
            random.shuffle(tmpRowOrder)
            self.__shuffleFmt[rowNo] = list(tmpRowOrder)

    def __constructFeatureFrame3d(self, featureFrameDir, constructDir):
        self.__initSharedZeroFfAsLine()

        ffDirsBy_calling = glob.glob(os.path.join(featureFrameDir, "*"))
        for ffDirBy_calling in ffDirsBy_calling:
            calling = os.path.basename(ffDirBy_calling)
            csDirBy_calling = os.path.join(constructDir, calling)
            os.mkdir(csDirBy_calling)
            ffFilePathsBy_date = glob.glob(os.path.join(ffDirBy_calling, "*.%s" % conf.DATA_FILE_SUFFIX))
            for ffFilePathBy_date in ffFilePathsBy_date:
                date = os.path.basename(ffFilePathBy_date).rstrip("*.%s" % conf.DATA_FILE_SUFFIX)
                # REFACTOR(20180705) extract depth time unit 24
                # OPT(20180705) shrink to 12 hours in day
                csFf3dAsMLinesBy_hour = map(lambda _: self.__sharedZeroFfAsLine, range(24))  # m means multi
                with open(ffFilePathBy_date, "r") as rFfFile:
                    sortedHours = map(int, json.loads(rFfFile.readline().strip()))
                    hourNo = 0
                    for ffLine in rFfFile:
                        csFf3dAsMLinesBy_hour[sortedHours[hourNo]] = ffLine.strip()
                        hourNo += 1
                    self.stat.zeroFfCnt += 24 - len(sortedHours)
                csFilePathBy_date = os.path.join(csDirBy_calling, "%s.%s" % (date, conf.DATA_FILE_SUFFIX))
                with open(csFilePathBy_date, "w") as wFile:
                    wFile.write(conf.ROW_SEPERATOR.join(csFf3dAsMLinesBy_hour))
                self.stat.FfCnt += 1

    def __initSharedZeroFfAsLine(self):
        if self.__sharedZeroFfAsLine is not None:
            return
        self.__sharedZeroFfAsLine = conf.COL_SEPERATOR.join(
            map(lambda _: "0", range(len(self.__ffFirstRowFmtDict) * conf.FeatureFrame3dDict.COPY_CNT)))


class _FeatureFrame3dStat(Stat):
    def __init__(self):
        super(_FeatureFrame3dStat, self).__init__()
        self.noCdrCallingCnt = list()
        self.FfCnt = 0
        self.zeroFfCnt = 0
