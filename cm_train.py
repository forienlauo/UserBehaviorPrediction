# coding=utf-8
import logging
import os
import shutil
import sys

import conf
from src.moduler.adapter.cm_data_adapter import CmDataAdapter


def main():
    # OPT(20180630) support option
    logLevel = sys.argv[1].upper()
    featureFrame3dDir = sys.argv[2]
    targetBehaviorDir = sys.argv[3]
    cacheDir = sys.argv[4]
    wkdir = sys.argv[5]
    # TODO(20180630) check args

    conf.logLevel = conf.LogLevel[logLevel.upper()]

    # REFACTOR(20180701) remove to check args
    # TODO(20180701) support cache
    if os.path.isdir(wkdir):
        logging.warn("wkdir already exist, delete and create: %s" % wkdir)
        shutil.rmtree(wkdir)
    os.makedirs(wkdir)
    tmpDir = os.path.join(wkdir, ".tmp")
    os.makedirs(tmpDir)

    logging.info("initing common conf")
    conf.FeatureFrame3dDict.init()

    logging.info("loading data")
    cmData = CmDataAdapter(
        featureFrame3dDir=featureFrame3dDir, targetBehaviorDir=targetBehaviorDir,
        cacheDir=cacheDir,
    ).run()

    logging.info("cleaning")
    shutil.rmtree(tmpDir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
