# coding=utf-8
import glob
import logging
import os
import shutil

import numpy as np
import pandas as pd

import conf
from src.common.usr_exceptions import NotSupportedError
from src.common.util import loadFormatDict, dumpFormatDict
from src.moduler.moduler import Moduler, Stat


# TODO(20180630) monitor performance
class CdrAggregator(Moduler):
    def __init__(
            self,
            translateCdrDir=None, tlCdrFmtFileName=None,
            aggregateCdrDir=None, aggregateTimeUnit=None, aggCdrFmtFileName=None,
            aggregateFeatures=None, aggregateFmt=None,
    ):
        super(CdrAggregator, self).__init__()
        self.translateCdrDir = translateCdrDir
        self.tlCdrFmtFilePath = os.path.join(self.translateCdrDir, tlCdrFmtFileName)
        self.aggregateCdrDir = aggregateCdrDir
        self.aggregateTimeUnit = aggregateTimeUnit
        self.aggCdrFmtFileName = aggCdrFmtFileName
        self.aggCdrTmpDir = os.path.join(self.aggregateCdrDir, ".tmp")

        self.aggregateFeatures = aggregateFeatures
        self.aggregateFmt = aggregateFmt

        self.stat = None
        self.__tlCdrFmtDict = None

    def run(self):
        self.__init()

        logging.info("start to aggregate cdr: %s" % self.translateCdrDir)
        self.__intRun()
        logging.info("output aggregate cdr: %s" % self.aggregateCdrDir)

        self.__clean()

    def __init(self):
        os.mkdir(self.aggregateCdrDir)
        if os.path.isdir(self.aggCdrTmpDir):
            shutil.rmtree(self.aggCdrTmpDir)
        os.makedirs(self.aggCdrTmpDir)

    def __clean(self):
        shutil.rmtree(self.aggCdrTmpDir)

    # OPT(20180701) parallel
    def __intRun(self):
        logging.debug("load translate format: %s" % self.tlCdrFmtFilePath)
        self.__tlCdrFmtDict = loadFormatDict(self.tlCdrFmtFilePath)

        self.stat = _CdrStat()

        noAggMapDirBy_calling = os.path.join(self.aggCdrTmpDir, "noAggregateBy_calling")
        os.mkdir(noAggMapDirBy_calling)
        self.__mapBy_calling(self.translateCdrDir, noAggMapDirBy_calling)

        noAggMapDirBy_calling_startTime = os.path.join(self.aggCdrTmpDir, "noAggMapBy_calling_startTime")
        os.mkdir(noAggMapDirBy_calling_startTime)
        self.__mapBy_calling_startTime(noAggMapDirBy_calling, noAggMapDirBy_calling_startTime)

        aggRedDirBy_calling_startTime = os.path.join(self.aggCdrTmpDir, self.aggregateTimeUnit.name)
        aggCdrFmtFilePath = os.path.join(aggRedDirBy_calling_startTime, self.aggCdrFmtFileName)
        os.mkdir(aggRedDirBy_calling_startTime)
        self.__reduceBy_calling_startTime(noAggMapDirBy_calling_startTime, aggRedDirBy_calling_startTime)
        logging.debug("dump aggregate format: %s" % aggCdrFmtFilePath)
        dumpFormatDict(self.aggregateFmt, aggCdrFmtFilePath)

        logging.debug("move aggregate reduce dir: %s" % aggRedDirBy_calling_startTime)
        os.rename(aggRedDirBy_calling_startTime,
                  os.path.join(self.aggregateCdrDir, os.path.basename(aggRedDirBy_calling_startTime)))

        logging.debug("%s stat: %s" % (self.name, self.stat,))

    def __mapBy_calling(self, translateCdrDir, noAggMapDir_calling):
        tlFmtDict = self.__tlCdrFmtDict

        tlCdrFilePaths = glob.glob(os.path.join(translateCdrDir, "*.%s" % conf.DATA_FILE_SUFFIX))
        for tlCdrFilePath in tlCdrFilePaths:
            mappedLinesBy_calling = dict()
            with open(tlCdrFilePath, "r") as rFile:
                for tlCdrLine in rFile:
                    cols = map(lambda col: col.strip(), tlCdrLine.strip().split(conf.COL_SEPERATOR))
                    calling = cols[tlFmtDict["CALLING"]]
                    if calling not in mappedLinesBy_calling:
                        mappedLinesBy_calling[calling] = list()
                    mappedLinesBy_calling[calling].append(tlCdrLine)
            for calling in mappedLinesBy_calling:
                mappedLines = mappedLinesBy_calling[calling]
                # OPT(20180702) decre io cost
                mappedFilePath = os.path.join(noAggMapDir_calling, "%s.%s" % (calling, conf.DATA_FILE_SUFFIX,))
                with open(mappedFilePath, "a") as afile:
                    afile.write("".join(mappedLines))

    def __mapBy_calling_startTime(self, noAggMapDirBy_calling, noAggMapDirBy_calling_startTime):
        tlFmtDict = self.__tlCdrFmtDict

        noAggMapFilePathsBy_calling = glob.glob(os.path.join(noAggMapDirBy_calling, "*.%s" % conf.DATA_FILE_SUFFIX))
        for noAggMapFilePathBy_calling in noAggMapFilePathsBy_calling:
            mappedLinesByKey = dict()
            calling = None
            with open(noAggMapFilePathBy_calling, "r") as rfile:
                for line in rfile:
                    cols = map(lambda col: col.strip(), line.strip().split(conf.COL_SEPERATOR))
                    if calling is None:
                        calling = cols[tlFmtDict["CALLING"]]
                    assert calling == cols[tlFmtDict["CALLING"]]
                    startTime = self.__simplifyStartTime(cols[tlFmtDict["START_TIME"]])
                    if startTime not in mappedLinesByKey:
                        mappedLinesByKey[startTime] = list()
                    mappedLinesByKey[startTime].append(line)
            mappedDir = os.path.join(noAggMapDirBy_calling_startTime, "%s.%s" % (calling, conf.KEY_DIR_SUFFIX))
            os.mkdir(mappedDir)
            for startTime in mappedLinesByKey:
                mappedLines = mappedLinesByKey[startTime]
                # OPT(20180702) decre io cost
                mappedFilePath = os.path.join(mappedDir, "%s.%s" % (startTime, conf.DATA_FILE_SUFFIX,))
                with open(mappedFilePath, "w") as wfile:
                    wfile.write("".join(mappedLines))
            self.stat.callingUCnt += 1
            self.stat.calling_startTimeUCnt += len(mappedLinesByKey)

    def __reduceBy_calling_startTime(self, noAggMapDirBy_calling_startTime, aggRedDirBy_calling_startTime):
        mappedDirsBy_Uid = glob.glob(os.path.join(noAggMapDirBy_calling_startTime, "*.%s" % conf.KEY_DIR_SUFFIX))
        for mappedDirBy_Uid in mappedDirsBy_Uid:
            calling = os.path.basename(mappedDirBy_Uid).rstrip(".%s" % conf.KEY_DIR_SUFFIX)
            inputFilePaths = glob.glob(os.path.join(mappedDirBy_Uid, "*.%s" % conf.DATA_FILE_SUFFIX))
            aggLines = list()
            for inputFilePath in inputFilePaths:
                with open(inputFilePath, "r") as rfile:
                    startTime = os.path.basename(inputFilePath).rstrip(".%s" % conf.DATA_FILE_SUFFIX)
                    noAggRows = map(lambda _: _.split(conf.COL_SEPERATOR),
                                    rfile.read(-1).strip().split(conf.ROW_SEPERATOR))
                    aggRow = self.__aggregate(calling, startTime, noAggRows)
                    self.stat.updateCallCnt(float(aggRow[self.aggregateFmt["CALL_CNT"]]))
                    aggLines.append(conf.COL_SEPERATOR.join(aggRow))
            reducedFilePath = os.path.join(aggRedDirBy_calling_startTime, "%s.%s" % (calling,
                                                                                     conf.DATA_FILE_SUFFIX,))
            with open(reducedFilePath, "w") as wfile:
                wfile.write(conf.ROW_SEPERATOR.join(aggLines))

    def __aggregate(self, calling, startTime, noAggRows):
        tlFmtDict = self.__tlCdrFmtDict
        aggregateFeatures = self.aggregateFeatures
        aggregateFmt = self.aggregateFmt
        aggregateRow = map(lambda _: "0", range(len(aggregateFeatures)))

        noAggDf = pd.DataFrame(noAggRows)
        # OPT(20180703) random drop
        noAggDf.drop_duplicates([tlFmtDict["START_TIME"]], "first", True)
        noAggDf[tlFmtDict["CALL_TIME"]] = map(int, noAggDf[tlFmtDict["CALL_TIME"]])
        noAggDf[tlFmtDict["COST"]] = map(int, noAggDf[tlFmtDict["COST"]])

        # key
        aggregateRow[aggregateFmt["CALLING"]] = calling
        aggregateRow[aggregateFmt["START_TIME"]] = startTime

        # value
        # TODO(20180704) compare with int value
        aggregateRow[aggregateFmt["CALL_CNT"]] = round(
            self.__translateXCnt(noAggDf[tlFmtDict["START_TIME"]].count()), 2)

        aggregateRow[aggregateFmt["CALLED_UCNT"]] = round(
            self.__translateXCnt(len(noAggDf[tlFmtDict["CALLED"]].unique())), 2)
        if not noAggDf[tlFmtDict["CALLED"]].is_unique:
            aggregateRow[aggregateFmt["CALLED_MODE"]] = np.random.choice(noAggDf[tlFmtDict["CALLED"]].mode(), 1)[0]
        else:
            aggregateRow[aggregateFmt["CALLED_MODE"]] = np.random.choice(noAggDf[tlFmtDict["CALLED"]], 1)[0]
        aggregateRow[aggregateFmt["CALLED_MCNT"]] = round(self.__translateXCnt(
            noAggDf[tlFmtDict["CALLED"]][
                noAggDf[tlFmtDict["CALLED"]] == aggregateRow[aggregateFmt["CALLED_MODE"]]].count()),
            2)

        aggregateRow[aggregateFmt["CALL_TIME_UCNT"]] = round(
            self.__translateXCnt(len(noAggDf[tlFmtDict["CALL_TIME"]].unique())), 2)
        aggregateRow[aggregateFmt["CALL_TIME_MEAN"]] = round(noAggDf[tlFmtDict["CALL_TIME"]].mean(), 2)
        if not noAggDf[tlFmtDict["CALL_TIME"]].is_unique:
            aggregateRow[aggregateFmt["CALL_TIME_MODE"]] = round(np.mean(noAggDf[tlFmtDict["CALL_TIME"]].mode()), 2)
        else:
            aggregateRow[aggregateFmt["CALL_TIME_MODE"]] = round(np.mean(noAggDf[tlFmtDict["CALL_TIME"]]), 2)
        aggregateRow[aggregateFmt["CALL_TIME_MCNT"]] = round(self.__translateXCnt(
            noAggDf[tlFmtDict["CALL_TIME"]][
                noAggDf[tlFmtDict["CALL_TIME"]] == aggregateRow[aggregateFmt["CALL_TIME_MODE"]]].count()), 2)
        if noAggDf[tlFmtDict["START_TIME"]].count() > 1:
            aggregateRow[aggregateFmt["CALL_TIME_STD"]] = round(noAggDf[tlFmtDict["CALL_TIME"]].std(), 2)
        else:
            aggregateRow[aggregateFmt["CALL_TIME_STD"]] = 0

        aggregateRow[aggregateFmt["COST_UCNT"]] = round(self.__translateXCnt(len(noAggDf[tlFmtDict["COST"]].unique())),
                                                        2)
        aggregateRow[aggregateFmt["COST_MEAN"]] = round(noAggDf[tlFmtDict["COST"]].mean(), 2)
        if not noAggDf[tlFmtDict["START_TIME"]].is_unique:
            aggregateRow[aggregateFmt["COST_MODE"]] = round(np.mean(noAggDf[tlFmtDict["COST"]].mode()), 2)
        else:
            aggregateRow[aggregateFmt["COST_MODE"]] = round(np.mean(noAggDf[tlFmtDict["COST"]]), 2)
        aggregateRow[aggregateFmt["COST_MCNT"]] = round(self.__translateXCnt(
            noAggDf[tlFmtDict["COST"]][noAggDf[tlFmtDict["COST"]] == aggregateRow[aggregateFmt["COST_MODE"]]].count()),
            2)
        if noAggDf[tlFmtDict["START_TIME"]].count() > 1:
            aggregateRow[aggregateFmt["COST_STD"]] = round(noAggDf[tlFmtDict["COST"]].std(), 2)
        else:
            aggregateRow[aggregateFmt["COST_STD"]] = 0

        aggregateRow[aggregateFmt["CDR_TYPE_UCNT"]] = round(
            self.__translateXCnt(len(noAggDf[tlFmtDict["CDR_TYPE"]].unique())), 2)
        if not noAggDf[tlFmtDict["START_TIME"]].is_unique:
            aggregateRow[aggregateFmt["CDR_TYPE_MODE"]] = np.random.choice(noAggDf[tlFmtDict["CDR_TYPE"]].mode(), 1)[0]
        else:
            aggregateRow[aggregateFmt["CDR_TYPE_MODE"]] = np.random.choice(noAggDf[tlFmtDict["CDR_TYPE"]], 1)[0]
        aggregateRow[aggregateFmt["CDR_TYPE_MCNT"]] = round(self.__translateXCnt(
            noAggDf[tlFmtDict["CDR_TYPE"]][
                noAggDf[tlFmtDict["CDR_TYPE"]] == aggregateRow[aggregateFmt["CDR_TYPE_MODE"]]].count()), 2)

        aggregateRow[aggregateFmt["TALK_TYPE_UCNT"]] = round(
            self.__translateXCnt(len(noAggDf[tlFmtDict["TALK_TYPE"]].unique())), 2)
        if not noAggDf[tlFmtDict["START_TIME"]].is_unique:
            aggregateRow[aggregateFmt["TALK_TYPE_MODE"]] = np.random.choice(noAggDf[tlFmtDict["TALK_TYPE"]].mode(), 1)[
                0]
        else:
            aggregateRow[aggregateFmt["TALK_TYPE_MODE"]] = np.random.choice(noAggDf[tlFmtDict["TALK_TYPE"]], 1)[0]
        aggregateRow[aggregateFmt["TALK_TYPE_MCNT"]] = round(self.__translateXCnt(
            noAggDf[tlFmtDict["TALK_TYPE"]][
                noAggDf[tlFmtDict["TALK_TYPE"]] == aggregateRow[aggregateFmt["TALK_TYPE_MODE"]]].count()), 2)

        aggregateRow[aggregateFmt["CALLED_AREA_UCNT"]] = round(
            self.__translateXCnt(len(noAggDf[tlFmtDict["CALLED_AREA"]].unique())), 2)
        if not noAggDf[tlFmtDict["START_TIME"]].is_unique:
            aggregateRow[aggregateFmt["CALLED_AREA_MODE"]] = \
            np.random.choice(noAggDf[tlFmtDict["CALLED_AREA"]].mode(), 1)[0]
        else:
            aggregateRow[aggregateFmt["CALLED_AREA_MODE"]] = np.random.choice(noAggDf[tlFmtDict["CALLED_AREA"]], 1)[0]
        aggregateRow[aggregateFmt["CALLED_AREA_MCNT"]] = round(self.__translateXCnt(
            noAggDf[tlFmtDict["CALLED_AREA"]][
                noAggDf[tlFmtDict["CALLED_AREA"]] == aggregateRow[aggregateFmt["CALLED_AREA_MODE"]]].count()), 2)

        return map(str, aggregateRow)

    def __simplifyStartTime(self, startTime):
        if self.aggregateTimeUnit == conf.AggregateCdrDict.AggregateTimeUnit.HOUR_6:
            return startTime[:8] + str(int(startTime[8:10]) / 6)
        if self.aggregateTimeUnit == conf.AggregateCdrDict.AggregateTimeUnit.HOUR_1:
            return startTime[:10]
        elif self.aggregateTimeUnit == conf.AggregateCdrDict.AggregateTimeUnit.MIN_10:
            return startTime[:10] + str(int(startTime[10:12]) / 10)
        elif self.aggregateTimeUnit == conf.AggregateCdrDict.AggregateTimeUnit.MIN_1:
            return startTime[:12]
        return

    def __translateXCnt(self, xCnt):
        translateXCnt = None
        if self.aggregateTimeUnit == conf.AggregateCdrDict.AggregateTimeUnit.HOUR_6:
            translateXCnt = 255.0 * xCnt / (6 * 60 * 60)
        elif self.aggregateTimeUnit == conf.AggregateCdrDict.AggregateTimeUnit.HOUR_1:
            translateXCnt = 255.0 * xCnt / (1 * 60 * 60)
        elif self.aggregateTimeUnit == conf.AggregateCdrDict.AggregateTimeUnit.MIN_10:
            translateXCnt = 255.0 * xCnt / (10 * 60)
        else:
            raise NotSupportedError("Not support current AggregateTimeUnit: %s" % self.aggregateTimeUnit.name)
        assert translateXCnt is not None

        translateXCnt = min(translateXCnt, 255.0 * conf.AggregateCdrDict.NORMAL_CALL_CNT_RATE)
        translateXCnt /= conf.AggregateCdrDict.NORMAL_CALL_CNT_RATE
        return translateXCnt


class _CdrStat(Stat):
    def __init__(self):
        super(_CdrStat, self).__init__()
        self.callingUCnt = 0
        self.calling_startTimeUCnt = 0

        self.__callCntThresh = 255.0
        self.callCntEqThresh = 0
        self.callCntGtHalfThresh = 0
        self.callCntLeHalfThresh = 0

    def updateCallCnt(self, callCnt):
        if callCnt == self.__callCntThresh:
            self.callCntEqThresh += 1
        elif callCnt > self.__callCntThresh / 2:
            self.callCntGtHalfThresh += 1
        else:
            self.callCntLeHalfThresh += 1
