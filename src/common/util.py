# coding=utf-8
import json


def loadFormatDict(fmtFilePath):
    with open(fmtFilePath, "r") as rfile:
        return json.loads(rfile.read(-1))


def dumpFormatDict(fmtDict, fmtFilePath):
    with open(fmtFilePath, "w") as wfile:
        wfile.write(json.dumps(fmtDict))
