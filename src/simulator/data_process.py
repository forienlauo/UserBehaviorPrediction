# coding=utf-8
import parallel_process as pc


def run(
        cdrDir=None, propertyDir=None,
        wkdir=None,
        workerCnt=None, maxTaskCnt=None,
):
    return pc.run(
        cdrDir=cdrDir, propertyDir=propertyDir,
        wkdir=wkdir,
        workerCnt=workerCnt, maxTaskCnt=maxTaskCnt,
        logLevel="DEBUG",
        force=True,
    )
