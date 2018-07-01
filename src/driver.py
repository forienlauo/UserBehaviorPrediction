# coding=utf-8
from src.moduler.preprocessor.cleaner import Cleaner


class Driver(object):
    def __init__(self):
        super(Driver, self).__init__()

    def run(self):
        import conf
        Cleaner(
            cdrDir=conf.CDR_DIR, propertyDir=conf.PROPERTY_DIR,
            cleanCdrDir=conf.CLEAN_CDR_DIR, cleanPropertyDir=conf.CLEAN_PROPERTY_DIR,
            dirtyCdrDir=conf.DIRTY_CDR_DIR, dirtyPropertyDir=conf.DIRTY_PROPERTY_DIR,
        ).run()
        # TODO(20180701) translate
        # TODO(20180701) aggregate
        # TODO(20180701) impl others
