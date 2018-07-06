# coding=utf-8
import glob
import json
import logging
import multiprocessing
import os
import shutil
import sys
import traceback

import base_conf as bconf


def main():
    # OPT(20180630) support option
    logLevel = sys.argv[1].upper()
    cdrDir = sys.argv[2]
    propertyDir = sys.argv[3]
    wkdir = sys.argv[4]
    workerCnt = int(sys.argv[5])
    maxTaskCnt = int(sys.argv[6])
    # TODO(20180630) check args

    # REFACTOR(20180701) remove to check args
    # TODO(20180701) support cache
    if os.path.isdir(wkdir):
        logging.warn("wkdir already exist, delete and create: %s" % wkdir)
        shutil.rmtree(wkdir)
    os.makedirs(wkdir)

    tmpDir = os.path.join(wkdir, ".tmp")
    os.mkdir(tmpDir)

    logging.info("mapping cdr")
    tmpCdrDir = os.path.join(tmpDir, "cdr")
    os.mkdir(tmpCdrDir)
    cdrDirsBy_task = mapCdr(cdrDir, maxTaskCnt, tmpCdrDir)
    taskCnt = len(cdrDirsBy_task)
    logging.info("mapped cdr into %d part" % taskCnt)

    tmpWorkDir = os.path.join(tmpDir, "work")
    os.mkdir(tmpWorkDir)
    workDirsBy_task = map(lambda taskNo: os.path.join(tmpWorkDir, os.path.basename(taskNo)), cdrDirsBy_task)

    taskResults = [None for _ in range(taskCnt)]

    def onWorkDone(resultTuple):
        ret, taskNo, e = resultTuple
        taskResults[taskNo] = resultTuple
        if ret == 0:
            logging.info("task-%d success" % taskNo)
        else:
            logging.error("task-%d return non-0 errcode(%s), caught exception in worker: \n%s" % (taskNo, ret, e,))

    logging.info("initing common conf")
    bconf.FeatureFrame3dDict.init()

    logging.info("start to submit task")
    pool = multiprocessing.Pool(workerCnt)
    for taskNo in range(taskCnt):
        pool.apply_async(
            work,
            args=(taskNo, logLevel, cdrDirsBy_task[taskNo], propertyDir, workDirsBy_task[taskNo],),
            # callback函数是由主进程去调用，大概是利用信号和共享内存实现
            callback=onWorkDone,
        )
    pool.close()
    pool.join()
    logging.info("all done")

    logging.info("checking result")
    for taskResult in taskResults:
        ret = taskResult[0]
        if ret != 0:
            return ret

    logging.info("merging result")
    merge2WorkDir(workDirsBy_task, wkdir)
    logging.info("cleaning")
    shutil.rmtree(tmpDir)

    return 0


def mapCdr(cdrDir, maxTaskCnt, tmpCdrDir):
    maxCdrDirsBy_task = [None for _ in range(maxTaskCnt)]
    cdrFilePaths = glob.glob(os.path.join(cdrDir, "*.%s" % bconf.DATA_FILE_SUFFIX))
    for cdrFilePath in cdrFilePaths:
        with open(cdrFilePath, "r") as rFile:
            for line in rFile:
                line = line.strip()
                calling = line.split(bconf.CdrDict.SEPERATOR)[bconf.CdrDict.Column.CALLING.value]
                orgHash = hash(calling)
                betterHash4SmallMode = (orgHash ^ (orgHash >> 32) ^ (orgHash >> 48) ^ (orgHash >> 56) ^ (orgHash >> 60))
                taskNo = betterHash4SmallMode % maxTaskCnt
                if maxCdrDirsBy_task[taskNo] is None:
                    cdrDirBy_task = os.path.join(tmpCdrDir, str(taskNo))
                    maxCdrDirsBy_task[taskNo] = cdrDirBy_task
                    os.mkdir(cdrDirBy_task)
                # OPT(20180706) cache file handle
                with open(os.path.join(maxCdrDirsBy_task[taskNo], os.path.basename(cdrFilePath)), "a") as aFile:
                    aFile.write(line)
                    aFile.write("\n")
    cdrDirsBy_task = list()
    for dir in maxCdrDirsBy_task:
        if dir is not None:
            cdrDirsBy_task.append(dir)
    return cdrDirsBy_task


def merge2WorkDir(workDirsBy_task, wkdir):
    tmpResultDir = os.path.join(wkdir, ".tmpResult")
    os.mkdir(tmpResultDir)

    for workDirBy_task in workDirsBy_task:
        if not os.path.isdir(workDirBy_task):
            continue
        # cleaner
        cleanCdrDir = os.path.join(workDirBy_task, "clean", "cdr")
        cleanPptDir = os.path.join(workDirBy_task, "clean", "property")
        dirtyCdrDir = os.path.join(workDirBy_task, "dirty", "cdr")
        dirtyPptDir = os.path.join(workDirBy_task, "dirty", "property")
        # translator
        translateCdrDir = os.path.join(workDirBy_task, "translate", "cdr")
        translatePptDir = os.path.join(workDirBy_task, "translate", "property")
        # aggregator
        aggregateCdrDir = os.path.join(workDirBy_task, "aggregate")
        # feature frame 3d constructor
        featureFrame3dDir = os.path.join(workDirBy_task, "featureFrame3d")
        # target behavior constructor
        targetBehaviorDir = os.path.join(workDirBy_task, "targetBehavior")

        dirsWithOverlap = [
            cleanCdrDir, cleanPptDir, dirtyCdrDir, dirtyPptDir,
            translateCdrDir, translatePptDir,
        ]
        for dirWithOverlap in dirsWithOverlap:
            if not os.path.isdir(dirWithOverlap):
                continue
            tgtDir = os.path.join(tmpResultDir, dirWithOverlap[len(workDirBy_task):].lstrip("/"))
            if not os.path.isdir(tgtDir):
                os.makedirs(tgtDir)
            mergeFmtFilesOnce(dirWithOverlap, tgtDir)
            mergeFilesWithOverlap(dirWithOverlap, tgtDir)

        dirsWithoutOverlap = [
            os.path.join(aggregateCdrDir, bconf.AGGREGATE_TIME_UNIT.name),
            featureFrame3dDir,
            targetBehaviorDir,
        ]
        for dirWithoutOverlap in dirsWithoutOverlap:
            if not os.path.isdir(dirWithoutOverlap):
                continue
            tgtDir = os.path.join(tmpResultDir, dirWithoutOverlap[len(workDirBy_task):].lstrip("/"))
            if not os.path.isdir(tgtDir):
                os.makedirs(tgtDir)
            mergeFmtFilesOnce(dirWithoutOverlap, tgtDir)
            mergeWithoutOverlap(dirWithoutOverlap, tgtDir)

    for path in glob.glob(os.path.join(tmpResultDir, "*")):
        os.rename(path, os.path.join(wkdir, os.path.basename(path)))
    shutil.rmtree(tmpResultDir)


def mergeFmtFilesOnce(srcDir, tgtDir):
    srcFmtFilePaths = glob.glob(os.path.join(srcDir, "*.%s" % bconf.FORMAT_FILE_SUFFIX))
    tgtFmtFilePaths = glob.glob(os.path.join(tgtDir, "*.%s" % bconf.FORMAT_FILE_SUFFIX))
    if len(tgtFmtFilePaths) > 0:
        assert map(os.path.basename, srcFmtFilePaths) == map(os.path.basename, tgtFmtFilePaths)
        fmtFileCnt = len(srcFmtFilePaths)
        for fmtFileNo in range(fmtFileCnt):
            assert checkFmt(srcFmtFilePaths[fmtFileNo], tgtFmtFilePaths[fmtFileNo])
    else:
        for srcFmtFilePath in srcFmtFilePaths:
            tgtFmtFilePath = os.path.join(tgtDir, os.path.basename(srcFmtFilePath))
            os.rename(srcFmtFilePath, tgtFmtFilePath)


def mergeFilesWithOverlap(srcDir, tgtDir):
    srcFilePaths = set(glob.glob(os.path.join(srcDir, "*")))
    tgtFilePaths = set(glob.glob(os.path.join(tgtDir, "*")))
    for srcFilePath in srcFilePaths:
        if not srcFilePath.endswith(bconf.DATA_FILE_SUFFIX):
            continue
        tgtFilePath = os.path.join(tgtDir, os.path.basename(srcFilePath))
        if tgtFilePath not in tgtFilePaths:
            os.rename(srcFilePath, tgtFilePath)
        else:
            with open(tgtFilePath, "a") as aFile:
                with open(srcFilePath, "r") as rFile:
                    aFile.write(bconf.ROW_SEPERATOR)
                    aFile.write(rFile.read(-1))


def mergeWithoutOverlap(srcDir, tgtDir):
    srcPaths = set(glob.glob(os.path.join(srcDir, "*")))
    tgtPaths = set(glob.glob(os.path.join(tgtDir, "*")))
    for srcPath in srcPaths:
        if not (srcPath.endswith(bconf.DATA_FILE_SUFFIX) or srcPath.endswith(bconf.KEY_DIR_SUFFIX)):
            continue
        tgtPath = os.path.join(tgtDir, os.path.basename(srcPath))
        assert tgtPath not in tgtPaths
        os.rename(srcPath, tgtPath)


def checkFmt(fmtFilePath1, fmtFilePath2):
    with open(fmtFilePath1, "r") as rFmtFile1, open(fmtFilePath2) as rFmtFile2:
        return json.loads(rFmtFile1.read(-1)) == json.loads(rFmtFile2.read(-1))


def work(taskNo, logLevel, cdrDir, propertyDir, wkdir):
    # set base conf(conf is constructed by base conf)
    bconf.wkdir = wkdir
    bconf.logLevel = bconf.LogLevel[logLevel.upper()]
    if os.path.isdir(wkdir):
        shutil.rmtree(wkdir)
    os.makedirs(wkdir)

    try:
        from src.processor import Processor
        Processor(cdrDir=cdrDir, propertyDir=propertyDir).run()
    except Exception as _:
        return (1, taskNo, getExceptionTrace())
    return (0, taskNo, None)


def getExceptionTrace():
    class SimpleFile(object):
        def __init__(self, ):
            super(SimpleFile, self).__init__()
            self.buffer = ""

        def write(self, str):
            self.buffer += str

        def read(self):
            return self.buffer

    simpleFile = SimpleFile()
    traceback.print_exc(file=simpleFile)
    return simpleFile.read()


if __name__ == "__main__":
    sys.exit(main())
