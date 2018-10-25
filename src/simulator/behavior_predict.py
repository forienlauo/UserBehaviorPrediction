# coding=utf-8
import cm_predict as pd


def run(
        modelFilePath=None,
        featureFrame3dDir=None, targetBehaviorDir=None, cacheDir=None,
        wkdir=None,
        cpuCoreCnt=None, gpuNos=None, gpuMemFraction=None,
):
    return pd.run(
        modelFilePath=modelFilePath,
        featureFrame3dDir=featureFrame3dDir, targetBehaviorDir=targetBehaviorDir, cacheDir=cacheDir,
        wkdir=wkdir,
        cpuCoreCnt=cpuCoreCnt, gpuNos=gpuNos, gpuMemFraction=gpuMemFraction,
        logLevel="DEBUG",
        force=True,
    )
