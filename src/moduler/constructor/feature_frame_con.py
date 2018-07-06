# coding=utf-8
import glob
import json
import logging
import os
import random
import shutil

import base_conf as bconf
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
            featureFrame3dDir=None, ffFirstRowFmtFileName=None, shuffleFmtFileName=None,
            ffFirstRowFeatures=None, ffFirstRowFmt=None, shuffleFmt=None,
    ):
        super(FeatureFrame3dConstructor, self).__init__()
        self.aggregateCdrDir = aggregateCdrDir
        self.aggregateTimeUnit = aggregateTimeUnit
        self.aggCdrFmtFilePath = os.path.join(self.aggregateCdrDir, self.aggregateTimeUnit.name, aggCdrFmtFileName)
        self.translatePropertyDir = translatePropertyDir
        self.tlPptFmtFilePath = os.path.join(self.translatePropertyDir, tlPptFmtFileName)
        self.featureFrame3dDir = featureFrame3dDir
        self.ff3TmpDir = os.path.join(self.featureFrame3dDir, ".tmp")
        self.shuffleFmtFilePath = os.path.join(self.featureFrame3dDir, shuffleFmtFileName)
        self.ffFirstRowFmtFilePath = os.path.join(self.featureFrame3dDir, ffFirstRowFmtFileName)

        self.ffFirstRowFeatures = ffFirstRowFeatures
        self.ffFirstRowFmt = ffFirstRowFmt
        self.shuffleFmt = shuffleFmt

        self.stat = None
        self.__aggCdrFmtDict = None
        self.__tlPptFmtDict = None
        self.__sharedZeroFfAsLine = None

        if self.aggregateTimeUnit != bconf.AggregateTimeUnit.HOUR_1:
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
        logging.debug("dump first row format of FeatureFrame: %s" % self.ffFirstRowFmtFilePath)
        dumpFormatDict(self.ffFirstRowFmt, self.ffFirstRowFmtFilePath)

        # TODO(20180703) copy && shuffle
        featureFrameDir = os.path.join(self.ff3TmpDir, "featureFrame")
        os.mkdir(featureFrameDir)
        self.__copyAndShuffle(joinDir, featureFrameDir)
        logging.debug("dump shuffle format of FeatureFrame: %s" % self.shuffleFmtFilePath)
        dumpFormatDict(self.shuffleFmt, self.shuffleFmtFilePath)

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
        ffFirstRowFeatures = self.ffFirstRowFeatures
        ffFirstRowFmt = self.ffFirstRowFmt

        tlPptFilePaths = glob.glob(os.path.join(translatePropertyDir, "*.%s" % bconf.DATA_FILE_SUFFIX))
        for tlPptFilePath in tlPptFilePaths:
            with open(tlPptFilePath, "r") as rTlPptFile:
                for tlPptLine in rTlPptFile:
                    tlPptCols = tlPptLine.strip().split(bconf.COL_SEPERATOR)
                    calling = tlPptCols[tlPptFmtDict["CALLING"]]
                    joinDictBy_date = dict()
                    aggCdrFilePath = os.path.join(
                        aggregateCdrDir, aggregateTimeUnit.name, "%s.%s" % (calling, bconf.DATA_FILE_SUFFIX))
                    if not os.path.isfile(aggCdrFilePath):
                        self.stat.noCdrCallingCnt.append(calling)
                        continue
                    with open(aggCdrFilePath, "r") as rAggCdrFile:
                        for aggCdrLine in rAggCdrFile:
                            aggCdrCols = aggCdrLine.strip().split(bconf.COL_SEPERATOR)
                            startTime = aggCdrCols[aggCdrFmtDict["START_TIME"]]
                            assert self.aggregateTimeUnit == bconf.AggregateTimeUnit.HOUR_1 and len(startTime) == 10
                            date = startTime[:8]
                            hour = startTime[8:10]
                            joinCols = range(len(ffFirstRowFeatures))
                            for ffFirstRowFeature in ffFirstRowFeatures:
                                joinCols[ffFirstRowFmt[ffFirstRowFeature]] = \
                                    self.__getFeatureValue(ffFirstRowFeature, aggCdrCols, tlPptCols)
                            if date not in joinDictBy_date:
                                joinDictBy_date[date] = dict()
                                # joinLinesBy_hour = joinDictBy_date[date]
                            assert hour not in joinDictBy_date[date]
                            joinDictBy_date[date][hour] = bconf.COL_SEPERATOR.join(joinCols)
                    joinDirBy_calling = os.path.join(joinDir, "%s.%s" % (calling, bconf.KEY_DIR_SUFFIX))
                    os.mkdir(joinDirBy_calling)
                    for date in joinDictBy_date:
                        joinFilePathBy_date = os.path.join(joinDirBy_calling, "%s.%s" % (date,
                                                                                         bconf.DATA_FILE_SUFFIX))
                        joinLinesBy_hour = joinDictBy_date[date]
                        with open(joinFilePathBy_date, "w") as wJoinFile:
                            sortedHours = sorted(joinLinesBy_hour.keys())
                            # file header
                            wJoinFile.write(json.dumps(sortedHours))
                            wJoinFile.write("\n")
                            # file body(sorted by hour)
                            wJoinFile.write(
                                bconf.ROW_SEPERATOR.join([joinLinesBy_hour[hour] for hour in sortedHours]))

    def __getFeatureValue(self, featureName, aggCdrCols, tlPptCols):
        if featureName in self.__tlPptFmtDict:
            return tlPptCols[self.__tlPptFmtDict[featureName]]
        else:
            assert featureName in self.__aggCdrFmtDict
            return aggCdrCols[self.__aggCdrFmtDict[featureName]]

    def __copyAndShuffle(self, joinDir, featureFrameDir):
        shuffleFmt = self.shuffleFmt

        joinDirsBy_calling = glob.glob(os.path.join(joinDir, "*.%s" % bconf.KEY_DIR_SUFFIX))
        for joinDirBy_calling in joinDirsBy_calling:
            calling = os.path.basename(joinDirBy_calling).rstrip(".%s" % bconf.KEY_DIR_SUFFIX)
            ffDirBy_calling = os.path.join(featureFrameDir, "%s.%s" % (calling, bconf.KEY_DIR_SUFFIX))
            os.mkdir(ffDirBy_calling)
            joinFilePathsBy_date = glob.glob(os.path.join(joinDirBy_calling, "*.%s" % bconf.DATA_FILE_SUFFIX))
            for joinFilePathBy_date in joinFilePathsBy_date:
                date = os.path.basename(joinFilePathBy_date).rstrip("*.%s" % bconf.DATA_FILE_SUFFIX)
                featureFramesAsLines = list()
                with open(joinFilePathBy_date, "r") as rJoinFile:
                    sortedHoursAsLine = rJoinFile.readline()
                    for joinLine in rJoinFile:
                        firstRow = joinLine.strip().split(bconf.COL_SEPERATOR)
                        featureFrameAsRow = list()
                        for shuffleRowFmt in shuffleFmt:
                            for colNoInFirstRow in shuffleRowFmt:
                                featureFrameAsRow.append(firstRow[colNoInFirstRow])
                        featureFramesAsLines.append(bconf.COL_SEPERATOR.join(featureFrameAsRow))
                ffFilePathBy_date = os.path.join(ffDirBy_calling, "%s.%s" % (date, bconf.DATA_FILE_SUFFIX))
                with open(ffFilePathBy_date, "w") as wFile:
                    # file header
                    wFile.write(sortedHoursAsLine)
                    # file body(sorted by hour)
                    wFile.write(bconf.ROW_SEPERATOR.join(featureFramesAsLines))

    def __constructFeatureFrame3d(self, featureFrameDir, constructDir):
        self.__initSharedZeroFfAsLine()

        ffDirsBy_calling = glob.glob(os.path.join(featureFrameDir, "*.%s" % bconf.KEY_DIR_SUFFIX))
        for ffDirBy_calling in ffDirsBy_calling:
            calling = os.path.basename(ffDirBy_calling).rstrip(".%s" % bconf.KEY_DIR_SUFFIX)
            csDirBy_calling = os.path.join(constructDir, "%s.%s" % (calling, bconf.KEY_DIR_SUFFIX))
            os.mkdir(csDirBy_calling)
            ffFilePathsBy_date = glob.glob(os.path.join(ffDirBy_calling, "*.%s" % bconf.DATA_FILE_SUFFIX))
            for ffFilePathBy_date in ffFilePathsBy_date:
                date = os.path.basename(ffFilePathBy_date).rstrip("*.%s" % bconf.DATA_FILE_SUFFIX)
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
                # OPT(20180705) compress files cnt if needed
                csFilePathBy_date = os.path.join(csDirBy_calling, "%s.%s" % (date, bconf.DATA_FILE_SUFFIX))
                with open(csFilePathBy_date, "w") as wFile:
                    wFile.write(bconf.ROW_SEPERATOR.join(csFf3dAsMLinesBy_hour))
                self.stat.FfCnt += 1

    def __initSharedZeroFfAsLine(self):
        if self.__sharedZeroFfAsLine is not None:
            return
        self.__sharedZeroFfAsLine = bconf.COL_SEPERATOR.join(
            map(lambda _: "0", range(len(self.ffFirstRowFmt) * bconf.FeatureFrame3dDict.COPY_CNT)))


class _FeatureFrame3dStat(Stat):
    def __init__(self):
        super(_FeatureFrame3dStat, self).__init__()
        self.noCdrCallingCnt = list()
        self.FfCnt = 0
        self.zeroFfCnt = 0
