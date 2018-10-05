# coding=utf-8
import argparse
import logging
import os
import sys

import shutil

import conf
from src.moduler.adapter.cm_data_adapter import \
    CmDataAdapter as DataAdapter, \
    CmData as Data
from src.moduler.predictor.cm_predictor import \
    CmPredictor as Predictor
# from src.moduler.predictor.simple_predictor import \
#     SimplePredictor as Predictor


def __parseArgs():
    parser = argparse.ArgumentParser(description='Predict with combined model')

    parser.add_argument("-m", "--modelFilePath", required=True, help=u"model file path, virtual")

    parser.add_argument("-f", "--featureFrame3dDir", required=True, help=u"featureFrame3d dir")
    parser.add_argument("-t", "--targetBehaviorDir", required=True, help=u"targetBehavior dir")
    parser.add_argument("-c", "--cacheDir", required=False, default="./.cache", help=u"cache dir")

    parser.add_argument("-w", "--wkdir", required=False, default="./train_wkdir",
                        help=u"wkdir, must exist and be empty")

    parser.add_argument("-C", "--cpuCoreCnt", type=int, required=False, default=1, help=u"cpuCore cnt")
    parser.add_argument("-G", "--gpuNos", required=False, default=None, help=u"gpu no s, separated by comma")
    parser.add_argument("-M", "--gpuMemFraction", type=float, required=False, default=1.0,
                        help=u"per process gpu memory fraction")

    parser.add_argument("-L", "--logLevel", required=False, default="info", help=u"log level")

    parser.add_argument("-F", "--force", action="store_true", required=False, default=False,
                        help=u"general force option, for example force removing wkdir is already exist")

    g_options = parser.parse_args()
    return g_options


def main():
    logging.info("argv: %s" % " ".join(sys.argv))
    options = __parseArgs()

    modelFilePath = options.modelFilePath

    featureFrame3dDir = options.featureFrame3dDir
    targetBehaviorDir = options.targetBehaviorDir
    cacheDir = options.cacheDir

    wkdir = options.wkdir

    cpuCoreCnt = options.cpuCoreCnt
    gpuNos = options.gpuNos
    gpuMemFraction = options.gpuMemFraction

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

    logging.info("predicting with combined model")
    predictor = Predictor(
        modelFilePath=modelFilePath,
        cpuCoreCnt=cpuCoreCnt, gpuNos=gpuNos, gpuMemFraction=gpuMemFraction,
    )
    predictor.init()
    ys, lossMse, lossRmse, lossMae, lossR2, lossRrmse, lossMape = predictor.predict(data)
    predictor.close()

    logging.info("dumping predict result")
    summaryFilePath = os.path.join(wkdir, "summary.txt")
    with open(summaryFilePath, "w") as _wfile:
        _wfile.write("exampleCnt : %d" % data.exampleCnt)
        _wfile.write("\n")
        _wfile.write("lossMse : %f" % lossMse)
        _wfile.write("\n")
        _wfile.write("lossRmse : %f" % lossRmse)
        _wfile.write("\n")
        _wfile.write("lossMae : %f" % lossMae)
        _wfile.write("\n")
        _wfile.write("lossR2 : %f" % lossR2)
        _wfile.write("\n")
        _wfile.write("lossRrmse : %f" % lossRrmse)
        _wfile.write("\n")
        _wfile.write("lossMape : %f" % lossMape)
        _wfile.write("\n")
    predictFilePath = os.path.join(wkdir, "predict.txt")
    with open(predictFilePath, "w")  as _wfile:
        for expNo in xrange(data.exampleCnt):
            exp = data.slice(expNo, 1)
            y_ = exp.predictTbs[0]
            y = ys[expNo]
            correctY = max(0.0, y)
            mape = abs(correctY - y_) / (y_ + 0.01)
            _wfile.write(
                "expNo:\t%d\ty_:\t%.4f\ty:\t%.4f\tcorrectY:\t%.4f\tmape:\t%0.2f"
                % (expNo, y_, y, correctY, mape))
            _wfile.write("\n")

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
