# coding=utf-8
import sys

import os

import src.simulator.data_process as dp
import src.simulator.model_train as mt
import src.simulator.behavior_predict as bp

cpuCoreCnt = 3
gpuNos = None
gpuMemFraction = None


def main():
    cdrDir = "demo/resource/cdr"
    propertyDir = "demo/resource/property"
    dp_wkdir = "demo/data_process"
    dp.run(
        cdrDir=cdrDir, propertyDir=propertyDir,
        wkdir=dp_wkdir,
        workerCnt=3, maxTaskCnt=6,
    )

    featureFrame3dDir = os.path.join(dp_wkdir, "featureFrame3d")
    targetBehaviorDir = os.path.join(dp_wkdir, "targetBehavior")
    cacheDir = "demo/__cache"
    mt_wkdir = "demo/model_train"
    mt.run(
        featureFrame3dDir=featureFrame3dDir, targetBehaviorDir=targetBehaviorDir, cacheDir=cacheDir,
        wkdir=mt_wkdir,
        cpuCoreCnt=cpuCoreCnt, gpuNos=gpuNos, gpuMemFraction=gpuMemFraction,
    )

    modelFilePath = os.path.join(mt_wkdir, "model", "cm")
    featureFrame3dDir = os.path.join(dp_wkdir, "featureFrame3d")
    targetBehaviorDir = os.path.join(dp_wkdir, "targetBehavior")
    cacheDir = "demo/__cache"
    bp_wkdir = "demo/behavior_predict"
    bp.run(
        modelFilePath=modelFilePath,
        featureFrame3dDir=featureFrame3dDir, targetBehaviorDir=targetBehaviorDir, cacheDir=cacheDir,
        wkdir=bp_wkdir,
        cpuCoreCnt=cpuCoreCnt, gpuNos=gpuNos, gpuMemFraction=gpuMemFraction,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
