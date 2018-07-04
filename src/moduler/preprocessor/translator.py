# coding=utf-8
import datetime
import glob
import json
import logging
import os
import shutil

import conf
from src.moduler.moduler import Moduler, Stat


# TODO(20180701) monitor performance
class Translator(Moduler):
    def __init__(
            self,
            cleanCdrDir=None, cleanPropertyDir=None,
            cleanPptFmtFileName=None, cleanCdrFmtFileName=None,
            translateCdrDir=None, translatePropertyDir=None,
            tlPptFmtFileName=None, tlCdrFmtFileName=None,
    ):
        super(Translator, self).__init__()
        self.cleanCdrDir = cleanCdrDir
        self.cleanPropertyDir = cleanPropertyDir
        self.cleanPptFmtFileName = cleanPptFmtFileName
        self.cleanCdrFmtFileName = cleanCdrFmtFileName
        self.translateCdrDir = translateCdrDir
        self.tlPptDir = translatePropertyDir
        self.tlPptFmtFileName = tlPptFmtFileName
        self.tlCdrFmtFileName = tlCdrFmtFileName

    def run(self):
        self.__init()

        logging.info("start to translate property: %s" % self.cleanPropertyDir)
        PropertyTranslator(
            cleanDir=self.cleanPropertyDir, cleanFmtFileName=self.cleanPptFmtFileName,
            translateDir=self.tlPptDir, tlFmtFileName=self.tlPptFmtFileName,
        ).run()
        logging.info("output translate property: %s" % self.tlPptDir)

        logging.info("start to translate cdr: %s" % self.cleanCdrDir)
        CdrTranslator(
            cleanDir=self.cleanCdrDir, cleanFmtFileName=self.cleanCdrFmtFileName,
            translateDir=self.translateCdrDir, tlFmtFileName=self.tlCdrFmtFileName,
        ).run()
        logging.info("output translate cdr: %s" % self.translateCdrDir)

    def __init(self):
        os.makedirs(self.translateCdrDir)
        os.makedirs(self.tlPptDir)


class _AbstractTranslator(Moduler):
    def __init__(
            self,
            cleanDir=None, cleanFmtFileName=None,
            translateDir=None, tlFmtFileName=None,
    ):
        super(_AbstractTranslator, self).__init__()
        self.cleanDir = cleanDir
        self.cleanFmtFilePath = os.path.join(self.cleanDir, cleanFmtFileName)
        self.translateDir = translateDir
        self.tlFmtFilePath = os.path.join(self.translateDir, tlFmtFileName)
        self._formatDict = None
        self.stat = None

    # OPT(20180701) parallel
    def run(self):
        logging.debug("load clean format: %s" % self.cleanFmtFilePath)
        self._formatDict = self._loadFormat()
        logging.debug("dump translate format: %s" % self.tlFmtFilePath)
        self._copyFormat()

        self.stat = self._newStat()

        cleanFilePaths = glob.glob(os.path.join(self.cleanDir, "*.%s" % conf.DATA_FILE_SUFFIX))
        for cleanFilePath in cleanFilePaths:
            translateLines = list()
            with open(cleanFilePath, "r") as rfile:
                for line in rfile:
                    cols = map(lambda col: col.strip(), line.strip().split(conf.COL_SEPERATOR))
                    translateCols = self._translate(cols)
                    self._updateStat(cols, translateCols)
                    translateLines.append(conf.COL_SEPERATOR.join(translateCols))
            translateFilePath = os.path.join(self.translateDir, os.path.basename(cleanFilePath))
            with open(translateFilePath, "w") as wfile:
                wfile.write(conf.ROW_SEPERATOR.join(translateLines))

        self._printStat()

    def _translate(self, cols):
        raise NotImplementedError("abstract method")

    def _newStat(self):
        pass

    def _updateStat(self, cols, translateCols):
        pass

    def _printStat(self):
        pass

    def _loadFormat(self, ):
        with open(self.cleanFmtFilePath, "r") as rfile:
            return json.loads(rfile.read(-1))

    def _copyFormat(self, ):
        shutil.copy(self.cleanFmtFilePath, self.tlFmtFilePath)


class PropertyTranslator(_AbstractTranslator):
    def __init__(self, cleanDir=None, cleanFmtFileName=None, translateDir=None, tlFmtFileName=None):
        super(PropertyTranslator, self).__init__(cleanDir, cleanFmtFileName, translateDir, tlFmtFileName)

    def _translate(self, cols):
        translateCols = list(cols)
        fmtDict = self._formatDict
        translateCols[fmtDict["PLAN_NAME"]] = str(conf.PropertyDict.PLAN_NAME_DICT[cols[fmtDict["PLAN_NAME"]]])
        translateCols[fmtDict["USER_TYPE"]] = str(conf.PropertyDict.USER_TYPE_DICT[cols[fmtDict["USER_TYPE"]]])
        translateCols[fmtDict["SELL_PRODUCT"]] = str(
            conf.PropertyDict.SELL_PRODUCT_DICT[cols[fmtDict["SELL_PRODUCT"]]])
        return translateCols


class CdrTranslator(_AbstractTranslator):
    def __init__(self, cleanDir=None, cleanFmtFileName=None, translateDir=None, tlFmtFileName=None):
        super(CdrTranslator, self).__init__(cleanDir, cleanFmtFileName, translateDir, tlFmtFileName)

    def _translate(self, cols):
        translateCols = list(cols)
        fmtDict = self._formatDict
        translateCols[fmtDict["CALL_TIME"]] = str(min(
            int(cols[fmtDict["CALL_TIME"]]) / conf.CdrDict.CALL_TIME_UNIT, 255))
        translateCols[fmtDict["COST"]] = str(min(int(cols[fmtDict["COST"]]) / conf.CdrDict.COST_UNIT, 255))
        translateCols[fmtDict["CDR_TYPE"]] = str(conf.CdrDict.CDR_TYPE_DICT[cols[fmtDict["CDR_TYPE"]]])
        translateCols[fmtDict["TALK_TYPE"]] = str(conf.CdrDict.TALK_TYPE_DICT[cols[fmtDict["TALK_TYPE"]]])
        return translateCols

    def _newStat(self):
        return _CdrStat()

    def _updateStat(self, cols, translateCols):
        callTime = cols[self._formatDict["CALL_TIME"]]
        if callTime not in self.stat.cntByCallTime:
            self.stat.cntByCallTime[callTime] = 0
        self.stat.cntByCallTime[callTime] += 1

        callTimeUnit = translateCols[self._formatDict["CALL_TIME"]]
        if callTimeUnit not in self.stat.cntByCallTimeUnit:
            self.stat.cntByCallTimeUnit[callTimeUnit] = 0
        self.stat.cntByCallTimeUnit[callTimeUnit] += 1

        cost = cols[self._formatDict["COST"]]
        if cost not in self.stat.cntByCost:
            self.stat.cntByCost[cost] = 0
        self.stat.cntByCost[cost] += 1

        costUnit = translateCols[self._formatDict["COST"]]
        if costUnit not in self.stat.cntByCostUnit:
            self.stat.cntByCostUnit[costUnit] = 0
        self.stat.cntByCostUnit[costUnit] += 1

    def _printStat(self):
        logging.debug("%s stat print closed for performance" % self.stat)
        # logging.debug("%s stat: %s" % (self.name, self.stat,))


class _CdrStat(Stat):
    def __init__(self):
        super(_CdrStat, self).__init__()
        self.cntByCallTime = dict()
        self.cntByCallTimeUnit = dict()
        self.cntByCost = dict()
        self.cntByCostUnit = dict()

    def addCost(self, cost, costUnit):
        if cost not in self.cntByCost:
            self.cntByCost[cost] = 0
        self.cntByCost[cost] += 1
        if costUnit not in self.cntByCostUnit:
            self.cntByCostUnit[costUnit] = 0
        self.cntByCostUnit[costUnit] += 1
