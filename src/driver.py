# coding=utf-8

from src.moduler.constructor.feature_frame_con import FeatureFrame3dConstructor
from src.moduler.constructor.target_behavior_con import TargetBehaviorConstructor
from src.moduler.preprocessor.aggregator import CdrAggregator
from src.moduler.preprocessor.cleaner import Cleaner
from src.moduler.preprocessor.translator import Translator


class Driver(object):
    def __init__(self, overwrite=False, ):
        super(Driver, self).__init__()
        self.overwrite = overwrite

    def run(self):
        import conf
        # OPT(20180705) pre-map cdr by uid(load balance), and process with multi processes parallel(so easy)
        # REFACTOR(20180705) extract prepare modulers
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
        FeatureFrame3dConstructor(
            aggregateCdrDir=conf.AGGREGATE_CDR_DIR, aggCdrFmtFileName=conf.AGG_CDR_FORMAT_FILE,
            aggregateTimeUnit=conf.AggregateTimeUnit.HOUR_1,
            translatePropertyDir=conf.TRANSLATE_PPT_DIR, tlPptFmtFileName=conf.TL_PPT_FORMAT_FILE,
            featureFrame3dDir=conf.FEATURE_FRAME_3D_DIR,
            shuffleFmtFileName=conf.SHUFFLE_FORMAT_FILE, ffFirstRowFmtFileName=conf.FF_FIRST_ROW_FORMAT_FILE,
        ).run()
        TargetBehaviorConstructor(
            featureFrame3dDir=conf.FEATURE_FRAME_3D_DIR, ffFirstRowFmtFileName=conf.FF_FIRST_ROW_FORMAT_FILE,
            targetBehaviorDir=conf.TARGET_BEHAVIOR_DIR,
        ).run()
        # TODO(20180701) impl others
