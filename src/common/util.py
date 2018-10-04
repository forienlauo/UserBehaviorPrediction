# coding=utf-8
import json
import os
import traceback

import tensorflow as tf


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


def dump_model(sess, model_file_path, ):
    """保存一个sess中的全部变量
    实际并不存在文件路径 __model_file_path, dirname(__model_file_path) 作为保存的目录, basename(__model_file_path) 作为模型的名字
    """
    saver = tf.train.Saver()
    saver.save(sess, model_file_path)


def load_model(sess, model_file_path, ):
    """加载一个sess中的全部变量
    实际并不存在文件路径 __model_file_path, dirname(__model_file_path) 作为保存的目录, basename(__model_file_path) 作为模型的名字
    @:return sess中的 gragh, 可以通过 __graph 取得所有 tensor 和 operation
    """
    meta_gragh_path = '%s.meta' % (model_file_path,)
    saver = tf.train.import_meta_graph(meta_gragh_path)
    saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(model_file_path)))
    return tf.get_default_graph()
