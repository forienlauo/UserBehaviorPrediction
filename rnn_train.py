# coding=utf-8
import argparse
import logging
import os
import shutil
import sys

import conf
from src.moduler.adapter.rnn_data_adapter import \
    RnnDataAdapter as DataAdapter, \
    RnnData as Data
from src.moduler.trainer.rnn_trainer import \
    RnnTrainer as Trainer


def __parseArgs():
    parser = argparse.ArgumentParser(description='Train combined model')

    parser.add_argument("-f", "--featureFrame3dDir", required=True, help=u"featureFrame3d dir")
    parser.add_argument("-t", "--targetBehaviorDir", required=True, help=u"targetBehavior dir")
    parser.add_argument("-p", "--trainExampleP", type=float, required=False, default=0.8,
                        help=u"train example splitting probability")
    parser.add_argument("-c", "--cacheDir", required=False, default="./.cache", help=u"cache dir")

    parser.add_argument("-w", "--wkdir", required=False, default="./train_wkdir",
                        help=u"wkdir, must exist and be empty")

    parser.add_argument("-l", "--lstmSize", type=int, required=False, default=20, help=u"LSTMCell size")

    parser.add_argument("-b", "--batchSizeConf", type=int, required=False, default=20, help=u"batch size configured")
    parser.add_argument("-k", "--keepProbConf", type=float, required=False, default=0.5, help=u"keepProb configured")

    parser.add_argument("-r", "--learnRate", type=float, required=False, default=0.001, help=u"learn rate")
    parser.add_argument("-d1", "--diff1", type=float, required=False, default=0, help=u"diff1 in evaluating")
    parser.add_argument("-d2", "--diff2", type=float, required=False, default=0, help=u"diff2 in evaluating")

    parser.add_argument("-C", "--cpuCoreCnt", type=int, required=False, default=1, help=u"cpuCore cnt")
    parser.add_argument("-G", "--gpuNos", required=False, default=None, help=u"gpu no s, separated by comma")
    parser.add_argument("-M", "--gpuMemFraction", type=float, required=False, default=1.0,
                        help=u"per process gpu memory fraction")
    parser.add_argument("-I", "--iteration", type=int, required=False, default=100, help=u"train step iteration")
    parser.add_argument("-S", "--printProgressPerStepCnt", type=int, required=False, default=10,
                        help=u"print progress per step cnt")

    parser.add_argument("-L", "--logLevel", required=False, default="info", help=u"log level")

    parser.add_argument("-F", "--force", action="store_true", required=False, default=False,
                        help=u"general force option, for example force removing wkdir is already exist")

    g_options = parser.parse_args()
    return g_options


def main():
    logging.info("argv: %s" % " ".join(sys.argv))
    options = __parseArgs()

    featureFrame3dDir = options.featureFrame3dDir
    targetBehaviorDir = options.targetBehaviorDir
    trainExampleP = options.trainExampleP
    cacheDir = options.cacheDir
    wkdir = options.wkdir

    lstmSize = options.lstmSize

    batchSizeConf = options.batchSizeConf
    keepProbConf = options.keepProbConf

    learnRate = options.learnRate
    diff1 = options.diff1
    diff2 = options.diff2

    cpuCoreCnt = options.cpuCoreCnt
    gpuNos = options.gpuNos
    gpuMemFraction = options.gpuMemFraction
    iteration = options.iteration
    printProgressPerStepCnt = options.printProgressPerStepCnt

    logLevel = options.logLevel

    force = options.force

    logLevel = logLevel.upper()

    # TODO(20180630) check args

    logging.info("initing")
    rc = __init(logLevel, wkdir, force)
    if rc != 0:
        return rc

    logging.info("loading data")
    if cacheDir is not None and os.path.isdir(cacheDir):
        logging.info("loading from cacheDir: %s" % cacheDir)
        data = Data.loadFromCacheDir(cacheDir)
    else:
        if cacheDir is None:
            logging.info("cache closed")
        else:
            logging.info("cacheDir not exist: %s" % cacheDir)
        data = DataAdapter(
            featureFrame3dDir=featureFrame3dDir, targetBehaviorDir=targetBehaviorDir,
        ).run()
        if cacheDir is not None:
            logging.info("dumping to cacheDir: %s" % cacheDir)
            os.mkdir(cacheDir)
            Data.dumpToCacheDir(data, cacheDir)

    logging.info("splitting data for train(%f) and test(%f)" % (trainExampleP, 1.0 - trainExampleP))
    trainData, testData = data.splitTrainTest(trainExampleP)
    del data

    logging.info("training combined model")
    Trainer(
        trainData=trainData, testData=testData,
        wkdir=wkdir,
        lstmSize=lstmSize,
        batchSizeConf=batchSizeConf, keepProbConf=keepProbConf,
        learnRate=learnRate, diff1=diff1, diff2=diff2,
        cpuCoreCnt=cpuCoreCnt, gpuNos=gpuNos, gpuMemFraction=gpuMemFraction,
        iteration=iteration, printProgressPerStepCnt=printProgressPerStepCnt,
    ).run()

    logging.info("cleaning")
    __clean()

    return 0


def __init(logLevel, wkdir, force):
    # useless
    conf.logLevel = conf.LogLevel[logLevel]
    conf.FeatureFrame3dDict.init()

    if os.path.isdir(wkdir):
        logging.warn("wkdir already exist: %s" % wkdir)
        if not force:
            return 1
        logging.warn("delete and create")
        shutil.rmtree(wkdir)
    os.makedirs(wkdir)
    return 0


def __clean():
    pass


if __name__ == "__main__":
    sys.exit(main())
