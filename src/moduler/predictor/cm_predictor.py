# coding=utf-8
import logging
import os

import tensorflow as tf

from src.common.util import load_model
from src.moduler.moduler import Moduler


class CmPredictor(Moduler):
    def __init__(
            self,
            modelFilePath=None,
            cpuCoreCnt=None, gpuNos=None, gpuMemFraction=None,
    ):
        super(CmPredictor, self).__init__()
        self.modelFilePath = modelFilePath

        self.cpuCoreCnt = cpuCoreCnt
        self.gpuNos = gpuNos
        self.gpuMemFraction = gpuMemFraction

        self.sess = None
        self.predictor = None

    def init(self):
        if self.gpuNos is not None:
            assert self.gpuMemFraction is not None
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpuNos
            gpuOptions = tf.GPUOptions(
                per_process_gpu_memory_fraction=self.gpuMemFraction,
                allow_growth=True,
            )
            config = tf.ConfigProto(gpu_options=gpuOptions)
        else:
            assert self.cpuCoreCnt is not None
            config = tf.ConfigProto(
                device_count={"CPU": self.cpuCoreCnt},
                inter_op_parallelism_threads=self.cpuCoreCnt,
                intra_op_parallelism_threads=self.cpuCoreCnt,
            )

        self.sess = tf.Session(config=config)

        logging.info("load model from: %s" % self.modelFilePath)
        runInput, runConf, runOutput = self.__loadModel()
        runConf.set(1, 1.0)
        self.predictor = CmPredictor._Predictor(runInput, runConf, runOutput)

    def close(self):
        if self.sess is not None:
            self.sess.close()
            self.predictor = None

    def predict(self, data):
        assert self.sess is not None and self.predictor is not None

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
            y0, lossMse0, lossRmse0, lossMae0, lossR20, lossRrmse0, lossMape0 = self.predictor.predict(self.sess, exp)
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

    def __loadModel(self):
        assert self.sess is not None
        graph = load_model(self.sess, self.modelFilePath)

        x = graph.get_tensor_by_name("input/x:0")
        y_ = graph.get_tensor_by_name("input/y_:0")
        runInput = CmPredictor._RunInput(x, y_)

        batchSize = graph.get_tensor_by_name("conf/batchSize:0")
        keepProb = graph.get_tensor_by_name("conf/keepProb:0")
        runConf = CmPredictor._RunConf(batchSize, keepProb)

        y = graph.get_tensor_by_name("targetBehaviorResolve/y:0")
        lossMse = graph.get_tensor_by_name("optimize/lossMse/lossMse:0")
        lossRmse = graph.get_tensor_by_name("optimize/lossRmse/lossRmse:0")
        lossMae = graph.get_tensor_by_name("optimize/lossMae/lossMae:0")
        lossR2 = graph.get_tensor_by_name("optimize/lossR2/lossR2:0")
        lossRrmse = graph.get_tensor_by_name("optimize/lossRrmse/lossRrmse:0")
        lossMape = graph.get_tensor_by_name("optimize/lossMape/lossMape:0")
        runOutput = CmPredictor._Runoutput(y, lossMse, lossRmse, lossMae, lossR2, lossRrmse, lossMape)

        return runInput, runConf, runOutput

    class _RunConf(object):
        def __init__(self, batchSize, keepProb):
            super(CmPredictor._RunConf, self).__init__()
            self.batchSize = batchSize
            self.keepProb = keepProb

            self.batchSizeConf = None
            self.keepProbConf = None

        def set(self, batchSizeConf, keepProbConf):
            self.batchSizeConf = batchSizeConf
            self.keepProbConf = keepProbConf

        def clear(self):
            self.batchSizeConf = None
            self.keepProbConf = None

    class _RunInput(object):
        def __init__(self, x, y_):
            super(CmPredictor._RunInput, self).__init__()
            self.x = x
            self.y_ = y_

    class _Runoutput(object):
        def __init__(
                self,
                y,
                lossMse, lossRmse, lossMae, lossR2, lossRrmse, lossMape,
        ):
            super(CmPredictor._Runoutput, self).__init__()
            self.y = y

            self.lossMse = lossMse
            self.lossRmse = lossRmse
            self.lossMae = lossMae
            self.lossR2 = lossR2
            self.lossRrmse = lossRrmse
            self.lossMape = lossMape

    class _Predictor(object):
        def __init__(self, runInput, runConf, runOutput, ):
            self.runInput = runInput
            self.runConf = runConf
            self.runOutput = runOutput

        def predict(self, sess, exp, ):
            assert exp.exampleCnt == 1

            runInput = self.runInput
            runConf = self.runConf
            runOutput = self.runOutput

            feedDict = {runConf.batchSize: exp.exampleCnt, runConf.keepProb: runConf.keepProbConf,
                        runInput.x: exp.learnMFf3ds, runInput.y_: exp.predictTbs, }
            yV, lossMseV, lossRmseV, lossMaeV, lossR2V, lossRrmseV, lossMapeV = sess.run(
                [runOutput.y,
                 runOutput.lossMse, runOutput.lossRmse, runOutput.lossMae,
                 runOutput.lossR2, runOutput.lossRrmse, runOutput.lossMape],
                feed_dict=feedDict, )
            return yV, lossMseV, lossRmseV, lossMaeV, lossR2V, lossRrmseV, lossMapeV
