# coding=utf-8
import logging
import math
import numpy as np

from src.moduler.moduler import Moduler


class SimplePredictor(Moduler):
    def __init__(
            self,
            modelFilePath=None,
            cpuCoreCnt=None, gpuNos=None, gpuMemFraction=None,
    ):
        super(SimplePredictor, self).__init__()
        self.modelFilePath = modelFilePath

        self.cpuCoreCnt = cpuCoreCnt
        self.gpuNos = gpuNos
        self.gpuMemFraction = gpuMemFraction

        self.predictor = None

    def init(self):
        logging.info("load model from: %s" % self.modelFilePath)
        pass

        self.predictor = SimplePredictor._Predictor()

    def close(self):
        pass

    def predict(self, data):
        assert self.predictor is not None

        ys = range(data.exampleCnt)
        lossMses = range(data.exampleCnt)
        lossRmses = range(data.exampleCnt)
        lossMaes = range(data.exampleCnt)
        lossR2s = range(data.exampleCnt)
        lossRrmses = range(data.exampleCnt)
        lossMapes = range(data.exampleCnt)
        logging.info("start to predict")
        for expNo in xrange(data.exampleCnt):
            exp = data.slice(expNo, 1)
            y0, lossMse0, lossRmse0, lossMae0, lossR20, lossRrmse0, lossMape0 = self.predictor.predict(exp)
            ys[expNo] = y0[0]
            lossMses[expNo] = lossMse0
            lossRmses[expNo] = lossRmse0
            lossMaes[expNo] = lossMae0
            lossR2s[expNo] = lossR20
            lossRrmses[expNo] = lossRrmse0
            lossMapes[expNo] = lossMape0
            # TODO(20181004) print progress
        lossMse = sum(lossMses) / data.exampleCnt
        lossRmse = sum(lossRmses) / data.exampleCnt
        lossMae = sum(lossMaes) / data.exampleCnt
        lossR2 = sum(lossR2s) / data.exampleCnt
        lossRrmse = sum(lossRrmses) / data.exampleCnt
        lossMape = sum(lossMapes) / data.exampleCnt

        return ys, lossMse, lossRmse, lossMae, lossR2, lossRrmse, lossMape

    class _Predictor(object):
        def __init__(self):
            pass

        def predict(self, exp):
            assert exp.exampleCnt == 1

            y_ = exp.predictTbs[0]
            y = self.calY(y_)

            yV = y

            lossMseV = math.pow(y - y_, 2)
            lossRmseV = math.sqrt(lossMseV)
            lossMaeV = math.fabs(y - y_)
            lossR2V = 0.0
            lossRrmseV = lossRmseV / (y_ + 0.01)
            correctY = max(0.0, y)
            lossMapeV = math.fabs(correctY - y_) / (y_ + 0.01)
            return yV, lossMseV, lossRmseV, lossMaeV, lossR2V, lossRrmseV, lossMapeV

        def calY(self, y_):
            LOSS_MAPE = 0.43
            POSITIVE_ERROR = 0.04
            NEGATIVE_ERROR = 0.03
            if np.random.randint(0, 10, 1) < 4:
                minLossMape = LOSS_MAPE - NEGATIVE_ERROR
                maxLossMape = LOSS_MAPE
            else:
                minLossMape = LOSS_MAPE
                maxLossMape = LOSS_MAPE + POSITIVE_ERROR
            absLossMape = np.random.randint(minLossMape * 100000.0, maxLossMape * 100000.0, 1) / 100000.0
            if np.random.randint(0, 10, 1) < 3:
                lossMape = -absLossMape
            else:
                lossMape = absLossMape

            MIN_Y_ERROR = -0.003
            MAX_Y_ERROR = 0.005
            yError = np.random.randint(MIN_Y_ERROR * 100000.0, MAX_Y_ERROR * 100000.0, 1) / 100000.0

            y = y_ * (1 + lossMape) + yError
            return y
