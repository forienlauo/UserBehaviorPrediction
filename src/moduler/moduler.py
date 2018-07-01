# coding=utf-8
import json


class Moduler(object):
    def __init__(self):
        super(Moduler, self).__init__()
        self.name = self.__class__.__name__


class Stat(object):
    def __init__(self):
        super(Stat, self).__init__()

    def __str__(self):
        return json.dumps(self.__dict__)
