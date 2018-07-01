# coding=utf-8
from src.moduler.preprocessor.cleaner import Cleaner
from src.moduler.preprocessor.translator import Translator


class Driver(object):
    def __init__(self):
        super(Driver, self).__init__()

    def run(self):
        import conf
        Cleaner(
            cdrDir=conf.CDR_DIR, propertyDir=conf.PROPERTY_DIR,
            cleanCdrDir=conf.CLEAN_CDR_DIR, cleanPropertyDir=conf.CLEAN_PPT_DIR,
            dirtyCdrDir=conf.DIRTY_CDR_DIR, dirtyPropertyDir=conf.DIRTY_PPT_DIR,
            cleanPptFmtFileName=conf.CLEAN_PPT_FORMAT_FILE, cleanCdrFmtFileName=conf.CLEAN_CDR_FORMAT_FILE,
        ).run()
        Translator(
            cleanCdrDir=conf.CLEAN_CDR_DIR, cleanPropertyDir=conf.CLEAN_PPT_DIR,
            cleanPptFmtFileName=conf.CLEAN_PPT_FORMAT_FILE, cleanCdrFmtFileName=conf.CLEAN_CDR_FORMAT_FILE,
            translateCdrDir=conf.TRANSLATE_CDR_DIR, translatePropertyDir=conf.TRANSLATE_PPT_DIR,
            tlPptFmtFileName=conf.TL_PPT_FORMAT_FILE, tlCdrFmtFileName=conf.TL_CDR_FORMAT_FILE,
        ).run()
        # TODO(20180701) aggregate
        # TODO(20180701) impl others
