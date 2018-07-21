# coding=utf-8
import datetime
import glob
import logging
import os
from collections import OrderedDict

import numpy as np

import conf
from src.moduler.moduler import Moduler, Stat


class CmDataAdapter(Moduler):
    def __init__(
            self,
            featureFrame3dDir=None, targetBehaviorDir=None,
            cacheDir=None,
    ):
        super(CmDataAdapter, self).__init__()
        self.featureFrame3dDir = featureFrame3dDir
        self.targetBehaviorDir = targetBehaviorDir
        self.cacheDir = cacheDir
        if self.cacheDir is not None:
            self.callingsCFilePath = os.path.join(cacheDir, "callings.%s" % conf.CACHE_FILE_SUFFIX)  # m means cache
            self.predictDatesCFilePath = os.path.join(cacheDir, "predictDates.%s" % conf.CACHE_FILE_SUFFIX)
            self.learnMFf3dsCFilePath = os.path.join(cacheDir, "learnMFf3ds.%s" % conf.CACHE_FILE_SUFFIX)
            self.predictTbsCFilePath = os.path.join(cacheDir, "predictTbs.%s" % conf.CACHE_FILE_SUFFIX)
        self.__sharedZeroFf3d = None
        self.__sharedZeroTb = None

    def run(self):
        if self.cacheDir is not None and os.path.isdir(self.cacheDir):
            logging.info("loading from cacheDir: %s" % self.cacheDir)
            callings = np.load(self.callingsCFilePath)
            predictDates = np.load(self.predictDatesCFilePath)
            learnMFf3ds = np.load(self.learnMFf3dsCFilePath)
            predictTbs = np.load(self.predictTbsCFilePath)
        else:
            self.__init()
            if self.cacheDir is None:
                logging.info("cache closed")
            else:
                logging.info("cacheDir not exist: %s" % self.cacheDir)

            logging.info("loading from featureFrame3dDir(%s) and targetBehaviorDir(%s)" %
                         (self.featureFrame3dDir, self.targetBehaviorDir))
            callings, predictDates, learnMFf3ds, predictTbs = map(np.array, self.__intRun())

            if self.cacheDir is not None:
                os.mkdir(self.cacheDir)
                np.save(self.callingsCFilePath, callings)
                np.save(self.predictDatesCFilePath, predictDates)
                np.save(self.learnMFf3dsCFilePath, learnMFf3ds)
                np.save(self.predictTbsCFilePath, predictTbs)
        return CMData(callings, predictDates, learnMFf3ds, predictTbs)

    def __init(self):
        if self.__sharedZeroFf3d is not None:
            return
        self.__sharedZeroFf3d = np.zeros(conf.FeatureFrame3dDict.FF3D_SHAPE)
        if self.__sharedZeroTb is not None:
            return
        self.__sharedZeroTb = np.zeros([1])

    # OPT(20180701) parallel
    def __intRun(self):
        callings = list()
        predictDates = list()
        learnMFf3ds = list()  # m means multi
        predictTbs = list()

        # OPT(20180721) decre memory cost
        # OPT(20180721) decre time cost
        ff3dDict = self.__loadFf3dAsDict()
        tbDict = self.__loadTbAsDict()
        # TODO(20180721) check whether ff3dDict and tbDict matches

        self.stat = _CmDataStat()

        for calling in ff3dDict:
            learnDayCnt = 0
            learnMFf3d = None

            dates = ff3dDict[calling].keys()
            minDate = dates[0]
            maxDate = dates[-1]
            if (maxDate - minDate).days < conf.TrainerDict.LEARN_DAY_CNT:
                # TODO(20180721) stat users not having enough cdr record
                continue
            dayOne = datetime.timedelta(days=1)
            date = minDate
            while (maxDate - date).days > 0:
                # TODO(20180721) stat sharedZeroFf3d's used times
                ff3d = ff3dDict[calling].get(date, self.__sharedZeroFf3d)
                if learnDayCnt == conf.TrainerDict.LEARN_DAY_CNT:
                    # make current learning example
                    predictDate = date
                    predictTb = tbDict[calling].get(date, self.__sharedZeroTb)
                    callings.append(calling)
                    predictDates.append(predictDate.strftime("%Y%m%d"))
                    learnMFf3ds.append(list(learnMFf3d))
                    predictTbs.append(predictTb)
                    if date not in tbDict[calling]:
                        self.stat.usefulZeroFf3dCnt += 1
                    self.stat.usefulExampleCnt += 1
                    # prepare for next learning example
                    learnMFf3d.pop(0)
                    learnMFf3d.append(ff3d)
                elif 0 < learnDayCnt < conf.TrainerDict.LEARN_DAY_CNT:
                    learnMFf3d[learnDayCnt] = ff3d
                    learnDayCnt += 1
                else:
                    assert learnMFf3d is None
                    learnMFf3d = [None for _ in range(conf.TrainerDict.LEARN_DAY_CNT)]
                    learnMFf3d[learnDayCnt] = ff3d
                    learnDayCnt += 1
                date += dayOne

        logging.debug("%s stat: %s" % (self.name, self.stat,))

        return callings, predictDates, learnMFf3ds, predictTbs

    def __loadFf3dAsDict(self):
        """
        :return: {calling: {date: ff3d}
        """
        ff3dDict = OrderedDict()
        ff3dDirsBy_calling = glob.glob(os.path.join(self.featureFrame3dDir, "*.%s" % conf.KEY_DIR_SUFFIX))
        for ff3dDirBy_calling in ff3dDirsBy_calling:
            calling = os.path.basename(ff3dDirBy_calling)[:-len(".%s" % conf.KEY_DIR_SUFFIX)]
            ff3dsBy_date = OrderedDict()
            ff3dFilePathsBy_date = glob.glob(os.path.join(ff3dDirBy_calling, "*.%s" % conf.DATA_FILE_SUFFIX))
            for ff3dFilePathBy_date in ff3dFilePathsBy_date:
                date = os.path.basename(ff3dFilePathBy_date)[:-len(".%s" % conf.DATA_FILE_SUFFIX)]
                assert len(date) == 8
                date = datetime.datetime(year=int(date[0:4]), month=int(date[4:6]), day=int(date[6:8]))
                ff3dsBy_date[date] = np.loadtxt(ff3dFilePathBy_date, delimiter=conf.COL_SEPERATOR) \
                    .reshape(conf.FeatureFrame3dDict.FF3D_SHAPE)
            ff3dDict[calling] = ff3dsBy_date
        return ff3dDict  #

    def __loadTbAsDict(self):
        """
        :return: {calling: {date: tb}
        """
        tbDict = OrderedDict()
        tbDirsBy_calling = glob.glob(os.path.join(self.targetBehaviorDir, "*.%s" % conf.KEY_DIR_SUFFIX))
        for tbDirBy_calling in tbDirsBy_calling:
            calling = os.path.basename(tbDirBy_calling)[:-len(".%s" % conf.KEY_DIR_SUFFIX)]
            tbsBy_date = OrderedDict()
            tbFilePathsBy_date = glob.glob(os.path.join(tbDirBy_calling, "*.%s" % conf.DATA_FILE_SUFFIX))
            for tbFilePathBy_date in tbFilePathsBy_date:
                date = os.path.basename(tbFilePathBy_date)[:-len(".%s" % conf.DATA_FILE_SUFFIX)]
                assert len(date) == 8
                date = datetime.datetime(year=int(date[0:4]), month=int(date[4:6]), day=int(date[6:8]))
                tbsBy_date[date] = np.loadtxt(tbFilePathBy_date, delimiter=conf.COL_SEPERATOR)
            tbDict[calling] = tbsBy_date
        return tbDict


class CMData(object):
    def __init__(
            self,
            callings=None, predictDates=None,
            learnFf3ds=None, predictTbs=None,
    ):
        super(CMData, self).__init__()
        self.callings = callings
        self.predictDates = predictDates
        self.learnFf3ds = learnFf3ds
        self.predictTbs = predictTbs


class _CmDataStat(Stat):
    def __init__(self):
        super(_CmDataStat, self).__init__()
        self.usefulExampleCnt = 0
        self.usefulZeroFf3dCnt = 0
