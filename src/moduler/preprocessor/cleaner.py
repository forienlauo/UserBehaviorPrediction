# coding=utf-8
import datetime
import glob
import json
import logging
import os

import base_conf as bconf
import conf
from src.moduler.moduler import Moduler, Stat


# TODO(20180630) monitor performance
class Cleaner(Moduler):
    def __init__(
            self,
            cdrDir=None, propertyDir=None,
            cleanCdrDir=None, cleanPropertyDir=None, dirtyCdrDir=None, dirtyPropertyDir=None,
            cleanPptFmtFileName=None, cleanCdrFmtFileName=None,
    ):
        super(Cleaner, self).__init__()
        self.cdrDir = cdrDir
        self.propertyDir = propertyDir
        self.cleanCdrDir = cleanCdrDir
        self.cleanPropertyDir = cleanPropertyDir
        self.dirtyCdrDir = dirtyCdrDir
        self.dirtyPropertyDir = dirtyPropertyDir
        self.cleanPptFmtFileName = cleanPptFmtFileName
        self.cleanCdrFmtFileName = cleanCdrFmtFileName
        self.__cdrCleaner = None
        self.__propertyCleaner = None

    def run(self):
        self.__init()

        logging.info("start to clean property: %s" % self.propertyDir)
        self.__propertyCleaner = PropertyCleaner(
            inputDir=self.propertyDir,
            cleanDir=self.cleanPropertyDir, dirtyDir=self.dirtyPropertyDir,
            cleanFmtFileName=self.cleanPptFmtFileName,
        )
        self.__propertyCleaner.run()
        cleanCallings = self.__propertyCleaner.getCleanCalling()
        logging.info("output clean property: %s" % self.cleanPropertyDir)
        logging.info("output dirty property: %s" % self.dirtyPropertyDir)

        logging.info("start to clean cdr: %s" % self.cdrDir)
        self.__cdrCleaner = CdrCleaner(
            inputDir=self.cdrDir, cleanCallings=cleanCallings,
            cleanDir=self.cleanCdrDir, dirtyDir=self.dirtyCdrDir,
            cleanFmtFileName=self.cleanCdrFmtFileName,
        )
        self.__cdrCleaner.run()
        logging.info("output clean cdr: %s" % self.cleanCdrDir)
        logging.info("output dirty cdr: %s" % self.dirtyCdrDir)

    def __init(self):
        os.makedirs(self.cleanCdrDir)
        os.makedirs(self.dirtyCdrDir)
        os.makedirs(self.cleanPropertyDir)
        os.makedirs(self.dirtyPropertyDir)

    def checkExistCleanData(self):
        assert os.path.isdir(self.cleanCdrDir) and os.path.isdir(self.dirtyCdrDir) \
               and os.path.isdir(self.cleanPropertyDir) and os.path.isdir(self.dirtyPropertyDir)
        assert self.__cdrCleaner is not None and self.__propertyCleaner is not None
        return self.__cdrCleaner.stat.cleanCdrCnt > 0 and self.__propertyCleaner.stat.cleanPropertyCnt


class PropertyCleaner(Moduler):
    def __init__(
            self,
            inputDir=None,
            cleanDir=None, dirtyDir=None,
            cleanFmtFileName=None,
    ):
        super(PropertyCleaner, self).__init__()
        self.inputDir = inputDir
        self.cleanDir = cleanDir
        self.dirtyDir = dirtyDir
        self.cleanCallings = set()
        self.cleanFmtFilePath = os.path.join(self.cleanDir, cleanFmtFileName)
        self.stat = None

    # OPT(20180701) parallel
    def run(self):
        logging.debug("dump format: %s" % self.cleanFmtFilePath)
        self.__dumpFormat()
        self.stat = _PropertyStat()

        inputFilePaths = glob.glob(os.path.join(self.inputDir, "*.%s" % bconf.DATA_FILE_SUFFIX))
        for inputFilePath in inputFilePaths:
            cleanLines = list()
            dirtyLines = list()
            with open(inputFilePath, "r") as rfile:
                for line in rfile:
                    cols = map(lambda col: col.strip(), line.strip().split(bconf.PropertyDict.SEPERATOR))
                    isClean = self.__check(cols)
                    if isClean:
                        self.cleanCallings.add(cols[bconf.PropertyDict.Column.CALLING.value])
                        usefulCols = list()
                        for colNo in range(len(cols)):
                            if colNo in bconf.PropertyDict.USEFUL_COLS:
                                usefulCols.append(cols[colNo])
                        cleanLines.append(conf.COL_SEPERATOR.join(usefulCols))
                        self.stat.cleanPropertyCnt += 1
                    else:
                        dirtyLines.append(conf.COL_SEPERATOR.join(cols))
                        self.stat.dirtyPropertyCnt += 1
            # TODO(20180701) re-filter by cleanLineCntThresh
            if len(cleanLines) > 0:
                cleanFilePath = os.path.join(self.cleanDir, os.path.basename(inputFilePath))
                with open(cleanFilePath, "w") as wfile:
                    wfile.write(conf.ROW_SEPERATOR.join(cleanLines))
            if len(dirtyLines) > 0:
                dirtyFilePath = os.path.join(self.dirtyDir, os.path.basename(inputFilePath))
                with open(dirtyFilePath, "w") as wfile:
                    wfile.write(conf.ROW_SEPERATOR.join(dirtyLines))

        logging.debug("%s stat: %s" % (self.name, self.stat,))

    def getCleanCalling(self):
        return set(self.cleanCallings)

    def __dumpFormat(self):
        _dumpFormat(bconf.PropertyDict, self.cleanFmtFilePath)

    def __check(self, cols):
        if len(cols) < bconf.PropertyDict.COL_CNT:
            return False
        C = bconf.PropertyDict.Column
        if not (30 <= len(cols[C.CALLING.value]) <= 32 and cols[C.CALLING.value] not in self.cleanCallings):
            return False
        if not (cols[C.PLAN_NAME.value] in bconf.PropertyDict.PLAN_NAME_DICT):
            return False
        if not (cols[C.USER_TYPE.value] in bconf.PropertyDict.USER_TYPE_DICT):
            return False
        if not (self.__checkOpenDate(cols[C.OPEN_DATE.value])):
            return False
        if not (cols[C.SUB_STAT_TP.value] == "活动"):
            return False
        if not (cols[C.IS_REAL_NAME.value] == "1"):
            return False
        if not (cols[C.SELL_PRODUCT.value] in bconf.PropertyDict.SELL_PRODUCT_DICT):
            return False
        return True

    def __checkOpenDate(self, openDateStr):
        try:
            datetime.datetime.strptime(openDateStr, "%Y-%m-%d %H:%M:%S")
            return True
        except ValueError:
            return False


class CdrCleaner(Moduler):
    def __init__(
            self,
            inputDir=None, cleanCallings=None,
            cleanDir=None, dirtyDir=None,
            cleanFmtFileName=None,
    ):
        super(CdrCleaner, self).__init__()
        self.inputDir = inputDir
        self.cleanCallings = cleanCallings
        self.cleanDir = cleanDir
        self.dirtyDir = dirtyDir
        self.cleanFmtFilePath = os.path.join(self.cleanDir, cleanFmtFileName)
        self.stat = None

    # OPT(20180701) parallel
    def run(self):
        logging.debug("dump format: %s" % self.cleanFmtFilePath)
        self.__dumpFormat()
        self.stat = _CdrStat()

        inputFilePaths = glob.glob(os.path.join(self.inputDir, "*.%s" % bconf.DATA_FILE_SUFFIX))
        for inputFilePath in inputFilePaths:
            cleanLines = []
            dirtyLines = []
            with open(inputFilePath, "r") as rfile:
                for line in rfile:
                    cols = map(lambda col: col.strip(), line.strip().split(bconf.CdrDict.SEPERATOR))
                    isClean = self.__check(cols)
                    if isClean:
                        usefulCols = []
                        for colNo in range(len(cols)):
                            if colNo in bconf.CdrDict.USEFUL_COLS:
                                usefulCols.append(cols[colNo])
                        cleanLines.append(conf.COL_SEPERATOR.join(usefulCols))
                        self.stat.cleanCdrCnt += 1
                        date = cols[bconf.CdrDict.Column.START_TIME.value][:8]
                        if date not in self.stat.cleanCdrCntByDate:
                            self.stat.cleanCdrCntByDate[date] = 0
                        self.stat.cleanCdrCntByDate[date] += 1
                    else:
                        dirtyLines.append(conf.COL_SEPERATOR.join(cols))
                        self.stat.dirtyCdrCnt += 1
            # TODO(20180701) re-filter by cleanLineCntThresh
            if len(cleanLines) > 0:
                cleanFilePath = os.path.join(self.cleanDir, os.path.basename(inputFilePath))
                with open(cleanFilePath, "w") as wfile:
                    wfile.write(conf.ROW_SEPERATOR.join(cleanLines))
            if len(dirtyLines) > 0:
                dirtyFilePath = os.path.join(self.dirtyDir, os.path.basename(inputFilePath))
                with open(dirtyFilePath, "w") as wfile:
                    wfile.write(conf.ROW_SEPERATOR.join(dirtyLines))

        logging.debug("%s stat: %s" % (self.name, self.stat,))

    def __dumpFormat(self):
        _dumpFormat(bconf.CdrDict, self.cleanFmtFilePath)

    # TODO(20180701) check duplicate cdr
    def __check(self, cols):
        if len(cols) < bconf.CdrDict.COL_CNT:
            return False
        # REFACTOR(20180701) load format
        C = bconf.CdrDict.Column
        if not (30 <= len(cols[C.CALLING.value]) <= 32 and cols[C.CALLING.value] in self.cleanCallings):
            return False
        if not (30 <= len(cols[C.CALLED.value]) <= 32):
            return False
        if not (30 <= len(cols[C.CHARGE_CALLING.value]) <= 32):
            return False
        if not (self.__checkStartTime(cols[C.START_TIME.value])):
            return False
        if not (self.__isNonNegativeInt(cols[C.CALL_TIME.value])):
            return False
        if not (self.__isNonNegativeInt(cols[C.COST.value])):
            return False
        if not (cols[C.CDR_TYPE.value] in bconf.CdrDict.CDR_TYPE_DICT):
            return False
        if not (cols[C.CALL_TYPE.value] == "1"):
            return False
        if not (cols[C.TALK_TYPE.value] in bconf.CdrDict.TALK_TYPE_DICT):
            return False
        if not (cols[C.CALLING_AREA.value] == "021"):
            return False
        # TODO(20180701) check area number
        # http://www.knowsky.com/tools/toolsdianhuaquhaoduizhaobiao.asp
        if not (3 <= len(cols[C.CALLED_AREA.value]) <= 4 and cols[C.CALLED_AREA.value].isdigit()):
            return False
        return True

    def __checkStartTime(self, startTimeStr):
        try:
            datetime.datetime.strptime(startTimeStr, "%Y%m%d%H%M%S")
            return True
        except ValueError:
            return False

    def __isNonNegativeInt(self, s):
        return len(s) > 0 and s.isdigit() \
               and (s == "0" or s[:1] != "0")


class _CdrStat(Stat):
    def __init__(self):
        super(_CdrStat, self).__init__()
        self.dirtyCdrCnt = 0
        self.cleanCdrCnt = 0
        self.cleanCdrCntByDate = dict()


class _PropertyStat(Stat):
    def __init__(self):
        super(_PropertyStat, self).__init__()
        self.dirtyPropertyCnt = 0
        self.cleanPropertyCnt = 0


def _dumpFormat(InitialFeatureDict, cleanFormatFilePath):
    formatDict = dict()
    for colEnum in InitialFeatureDict.Column:
        colName = colEnum.name
        colNo = colEnum.value
        if colNo in InitialFeatureDict.USEFUL_COLS:
            formatDict[colName] = len(formatDict)
    with open(cleanFormatFilePath, "w") as wfile:
        wfile.write(json.dumps(formatDict))
