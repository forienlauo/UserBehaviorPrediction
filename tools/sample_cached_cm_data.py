# coding=utf-8
import argparse
import logging
import os
import shutil
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import conf
from src.moduler.adapter.cm_data_adapter import CmData


def __parseArgs():
    parser = argparse.ArgumentParser(description='Sample cached data for combined model')

    parser.add_argument("-o", "--originalCacheDir", required=False, default="./.original_cache",
                        help=u"original cache dir")
    parser.add_argument("-n", "--sampleCnt", type=int, required=False, default=1000, help=u"sample cnt")
    parser.add_argument("-s", "--sampledCacheDir", required=False, default="./.sampled_cache",
                        help=u"sampled cache dir")

    parser.add_argument("-L", "--logLevel", required=False, default="info", help=u"log level")

    parser.add_argument("-F", "--force", action="store_true", required=False, default=False,
                        help=u"general force option, for example force removing wkdir is already exist")

    g_options = parser.parse_args()
    return g_options


def main():
    options = __parseArgs()

    originalCacheDir = options.originalCacheDir
    sampleCnt = options.sampleCnt
    sampledCacheDir = options.sampledCacheDir

    logLevel = options.logLevel

    force = options.force

    logLevel = logLevel.upper()

    # TODO(20180630) check args

    logging.info("initing")
    rc = __init(logLevel, sampledCacheDir, force)
    if rc != 0:
        return rc

    logging.info("loading data from originalCacheDir: %s" % originalCacheDir)
    orgCmData = CmData.loadFromCacheDir(originalCacheDir)

    if sampleCnt >= orgCmData.exampleCnt:
        logging.warn("no need to sample %d examples from %d examples in originalCacheDir: %s"
                     % (sampleCnt, orgCmData.exampleCnt, originalCacheDir))
        return 1
    logging.info("sampling %d examples from %d examples" % (sampleCnt, orgCmData.exampleCnt))
    spCmData = orgCmData.randomSampleBatch(sampleCnt)
    assert sampleCnt == spCmData.exampleCnt
    del orgCmData

    logging.info("dumping to sampledCacheDir: %s" % sampledCacheDir)
    CmData.dumpToCacheDir(spCmData, sampledCacheDir)
    del spCmData

    logging.info("cleaning")
    __clean()

    return 0


def __init(logLevel, sampledCacheDir, force):
    conf.logLevel = conf.LogLevel[logLevel]

    if os.path.isdir(sampledCacheDir):
        logging.warn("sampledCacheDir already exist: %s" % sampledCacheDir)
        if not force:
            return 1
        logging.warn("delete and create")
        shutil.rmtree(sampledCacheDir)
    os.makedirs(sampledCacheDir)
    return 0


def __clean():
    pass


if __name__ == "__main__":
    sys.exit(main())
