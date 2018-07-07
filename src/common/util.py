# coding=utf-8
import json
import traceback


class SimpleFile(object):
    def __init__(self, ):
        super(SimpleFile, self).__init__()
        self.buffer = ""

    def write(self, str):
        self.buffer += str

    def read(self):
        return self.buffer


def loadFormatDict(fmtFilePath):
    with open(fmtFilePath, "r") as rfile:
        return json.loads(rfile.read(-1))


def dumpFormatDict(fmtDict, fmtFilePath):
    with open(fmtFilePath, "w") as wfile:
        wfile.write(json.dumps(fmtDict))


def getExceptionTrace():
    simpleFile = SimpleFile()
    traceback.print_exc(file=simpleFile)
    return simpleFile.read()
