# coding=utf-8
from src.moduler.preprocessor.aggregator import CdrAggregator
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
            cleanCdrFmtFileName=conf.CLEAN_CDR_FORMAT_FILE, cleanPptFmtFileName=conf.CLEAN_PPT_FORMAT_FILE,
        ).run()
        Translator(
            cleanCdrDir=conf.CLEAN_CDR_DIR, cleanPropertyDir=conf.CLEAN_PPT_DIR,
            cleanCdrFmtFileName=conf.CLEAN_CDR_FORMAT_FILE, cleanPptFmtFileName=conf.CLEAN_PPT_FORMAT_FILE,
            translateCdrDir=conf.TRANSLATE_CDR_DIR, translatePropertyDir=conf.TRANSLATE_PPT_DIR,
            tlCdrFmtFileName=conf.TL_CDR_FORMAT_FILE, tlPptFmtFileName=conf.TL_PPT_FORMAT_FILE,
        ).run()
        CdrAggregator(
            translateCdrDir=conf.TRANSLATE_CDR_DIR, tlCdrFmtFileName=conf.TL_CDR_FORMAT_FILE,
            aggregateCdrDir=conf.AGGREGATE_CDR_DIR, aggCdrFmtFileName=conf.AGG_CDR_FORMAT_FILE,
            aggregateTimeUnit=conf.AggregateTimeUnit.HOUR_1,
        ).run()
        # TODO(20180701) impl others
