# coding=utf-8
import glob
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
    tmpCdrDir = os.path.join(tmpDir, "cdr")
    os.mkdir(tmpCdrDir)
    tmpWorkDir = os.path.join(tmpDir, "work")
    os.mkdir(tmpWorkDir)
    cdrDirsBy_task = mapCdr(cdrDir, maxTaskCnt, tmpCdrDir)
    workDirsBy_task = map(lambda taskNo: os.path.join(tmpWorkDir, os.path.basename(taskNo)), cdrDirsBy_task)
    taskCnt = len(cdrDirsBy_task)

    logging.info("start to submit task")
    pool = multiprocessing.Pool(workerCnt)
    for taskNo in range(taskCnt):
        pool.apply_async(
            work,
            args=(taskNo, logLevel, cdrDirsBy_task[taskNo], propertyDir, workDirsBy_task[taskNo],),
            # callback函数是由主进程去调用，大概是利用信号实现的和共享内存
            callback=onWorkDone,
        )
    pool.close()
    pool.join()
    logging.info("all done")

    # mergeAndMoveWorkDir(workDirsBy_task, wkdir)
    # shutil.rmtree(tmpDir)

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
                with open(os.path.join(maxCdrDirsBy_task[taskNo], os.path.basename(cdrFilePath)), "a") as wFile:
                    wFile.write(line)
                    wFile.write("\n")
    cdrDirsBy_task = list()
    for dir in maxCdrDirsBy_task:
        if dir is not None:
            cdrDirsBy_task.append(dir)
    return cdrDirsBy_task


def mergeAndMoveWorkDir(workDirsBy_task, wkdir):
    # TODO(20180706) read from all workDirsBy_task, write into the very wkdir
    raise NotImplementedError()


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


def onWorkDone(resultTuple):
    import json
    logging.info("args: %s" % json.dumps(resultTuple))
    ret, taskNo, e = resultTuple
    if ret == 0:
        logging.info("task-%d success" % taskNo)
    else:
        logging.error("task-%d return non-0 errcode(%s), caught exception in worker: \n%s" % (taskNo, ret, e,))


if __name__ == "__main__":
    sys.exit(main())
