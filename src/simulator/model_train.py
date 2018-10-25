# coding=utf-8
import cm_train as t


def run(
        featureFrame3dDir=None, targetBehaviorDir=None, cacheDir=None,
        wkdir=None,
        cpuCoreCnt=None, gpuNos=None, gpuMemFraction=None,
):
    return t.run(
        featureFrame3dDir=featureFrame3dDir, targetBehaviorDir=targetBehaviorDir, trainExampleP=0.8, cacheDir=cacheDir,
        wkdir=wkdir,
        convShape=[6, 2, 5], convStrides=[1, 1, 1], poolShape=[2, 2, 2], poolStrides=[2, 2, 2], convCnts=[16, 32, 64],
        lstmSize=10,
        batchSizeConf=5, keepProbConf=0.8,
        # batchSizeConf=40, keepProbConf=0.8,
        iteration=4,
        # iteration=1000,
        error="MSE", learnRate=0.0001, diff1=0, diff2=0,
        cpuCoreCnt=cpuCoreCnt, gpuNos=gpuNos, gpuMemFraction=gpuMemFraction,
        printProgressPerStepCnt=1, logLevel="DEBUG",
        force=True,
    )
