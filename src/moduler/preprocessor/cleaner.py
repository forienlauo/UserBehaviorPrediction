# coding=utf-8
import datetime
import glob
import logging
import os

import conf
from src.moduler.moduler import Moduler, Stat


# TODO(20180630) monitor performance
class Cleaner(Moduler):
    def __init__(
            self,
            cdrDir=None, propertyDir=None,
            cleanCdrDir=None, cleanPropertyDir=None, dirtyCdrDir=None, dirtyPropertyDir=None,
    ):
        super(Cleaner, self).__init__()
        self.cdrDir = cdrDir
        self.propertyDir = propertyDir
        self.cleanCdrDir = cleanCdrDir
        self.cleanPropertyDir = cleanPropertyDir
        self.dirtyCdrDir = dirtyCdrDir
        self.dirtyPropertyDir = dirtyPropertyDir

    def run(self):
        self.__init()

        logging.info("start to clean property: %s" % self.propertyDir)
        pptCleaner = PropertyCleaner(
            inputDir=self.propertyDir,
            cleanDir=self.cleanPropertyDir, dirtyDir=self.dirtyPropertyDir,
        )
        pptCleaner.run()
        cleanCallings = pptCleaner.getCleanCalling()
        logging.info("Output clean property: %s" % self.cleanPropertyDir)
        logging.info("Output dirty property: %s" % self.dirtyPropertyDir)

        logging.info("start to clean cdr: %s" % self.cdrDir)
        CdrCleaner(
            inputDir=self.cdrDir, cleanCallings=cleanCallings,
            cleanDir=self.cleanCdrDir, dirtyDir=self.dirtyCdrDir,
        ).run()
        logging.info("Output clean cdr: %s" % self.cleanCdrDir)
        logging.info("Output dirty cdr: %s" % self.dirtyCdrDir)

    def __init(self):
        os.makedirs(self.cleanCdrDir)
        os.makedirs(self.dirtyCdrDir)
        os.makedirs(self.cleanPropertyDir)
        os.makedirs(self.dirtyPropertyDir)


class PropertyCleaner(Moduler):
    def __init__(
            self,
            inputDir=None,
            cleanDir=None, dirtyDir=None,
    ):
        super(PropertyCleaner, self).__init__()
        self.inputDir = inputDir
        self.cleanDir = cleanDir
        self.dirtyDir = dirtyDir
        self.cleanCallings = set()
        self.stat = None

    # OPT(20180701) parallel
    def run(self):
        self.stat = _PropertyStat()

        inputFilePaths = glob.glob(os.path.join(self.inputDir, "*"))
        for inputFilePath in inputFilePaths:
            cleanLines = list()
            dirtyLines = list()
            with open(inputFilePath, "r") as rfile:
                for line in rfile:
                    cols = map(lambda col: col.strip(), line.strip().split(conf.PropertyDict.SEPERATOR))
                    isClean = self.__check(cols)
                    if isClean:
                        self.cleanCallings.add(cols[conf.PropertyDict.Column.CALLING.value])
                        usefulCols = list()
                        for colNo in range(len(cols)):
                            if colNo in conf.PropertyDict.USEFUL_COLS:
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

        logging.debug("property stat: %s" % self.stat)

    def getCleanCalling(self):
        return set(self.cleanCallings)

    def __check(self, cols):
        if len(cols) < conf.PropertyDict.COL_CNT:
            return False
        C = conf.PropertyDict.Column
        if not (30 <= len(cols[C.CALLING.value]) <= 32 and cols[C.CALLING.value] not in self.cleanCallings):
            return False
        if not (cols[C.PLAN_NAME.value] in conf.PropertyDict.PLAN_NAME_DICT):
            return False
        if not (cols[C.USER_TYPE.value] in conf.PropertyDict.USER_TYPE_DICT):
            return False
        if not (self.__checkOpenDate(cols[C.OPEN_DATE.value])):
            return False
        if not (cols[C.SUB_STAT_TP.value] == "活动"):
            return False
        if not (cols[C.IS_REAL_NAME.value] == "1"):
            return False
        if not (cols[C.SELL_PRODUCT.value] in conf.PropertyDict.SELL_PRODUCT_DICT):
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
    ):
        super(CdrCleaner, self).__init__()
        self.inputDir = inputDir
        self.cleanCallings = cleanCallings
        self.cleanDir = cleanDir
        self.dirtyDir = dirtyDir
        self.stat = None

    # OPT(20180701) parallel
    def run(self):
        self.stat = _CdrStat()

        inputFilePaths = glob.glob(os.path.join(self.inputDir, "*"))
        for inputFilePath in inputFilePaths:
            cleanLines = []
            dirtyLines = []
            with open(inputFilePath, "r") as rfile:
                for line in rfile:
                    cols = map(lambda col: col.strip(), line.strip().split(conf.CdrDict.SEPERATOR))
                    isClean = self.__check(cols)
                    if isClean:
                        usefulCols = []
                        for colNo in range(len(cols)):
                            if colNo in conf.CdrDict.USEFUL_COLS:
                                usefulCols.append(cols[colNo])
                        cleanLines.append(conf.COL_SEPERATOR.join(usefulCols))
                        self.stat.cleanCdrCnt += 1
                        date = cols[conf.CdrDict.Column.START_TIME.value][:8]
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

        logging.debug("cdr stat: %s" % self.stat)

    # TODO(20180701) check duplicate cdr
    def __check(self, cols):
        if len(cols) < conf.CdrDict.COL_CNT:
            return False
        C = conf.CdrDict.Column
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
        if not (cols[C.CDR_TYPE.value] in conf.CdrDict.CDR_TYPE_DICT):
            return False
        if not (cols[C.CALL_TYPE.value] == "1"):
            return False
        if not (cols[C.TALK_TYPE.value] in conf.CdrDict.TALK_TYPE_DICT):
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
