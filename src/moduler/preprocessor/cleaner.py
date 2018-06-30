# coding=utf-8
import logging

import conf
from src.moduler.moduler import Moduler, Stat


# TODO(20180630) add permance monitor
class Cleaner(Moduler):
    def __init__(self):
        super(Cleaner, self).__init__()

    def run(self):
        logging.info("Start to clean cdr: %s" % conf.CDR_DIR)
        self.cleanCdr()
        logging.info("Start to clean property: %s" % conf.PROPERTY_DIR)
        self.cleanProperty()

    def cleanCdr(self):
        # TODO(20180630) impl
        raise NotImplementedError()

    def cleanProperty(self):
        # TODO(20180630) impl
        raise NotImplementedError()


class CdrStat(Stat):
    def __init__(self):
        super(CdrStat, self).__init__()
        self.dirtyCdrCnt = 0
        self.cleanCdrCnt = 0
        self.cleanCdrCntByDate = dict()

    def _update(self, cdr):
        # TODO(20180630) impl
        raise NotImplementedError()


class PropertyStat(object):
    def __init__(self):
        super(PropertyStat, self).__init__()
        self.dirtyPropertyCnt = 0
        self.cleanPropertyCnt = 0

    def _update(self, property):
        # TODO(20180630) impl
        raise NotImplementedError()
