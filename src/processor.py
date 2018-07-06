# coding=utf-8
import logging

import base_conf as bconf
from src.moduler.constructor.feature_frame_con import FeatureFrame3dConstructor
from src.moduler.constructor.target_behavior_con import TargetBehaviorConstructor
from src.moduler.preprocessor.aggregator import CdrAggregator
from src.moduler.preprocessor.cleaner import Cleaner
from src.moduler.preprocessor.translator import Translator


class Processor(object):
    def __init__(self, cdrDir=None, propertyDir=None, ):
        super(Processor, self).__init__()
        self.cdrDir = cdrDir
        self.propertyDir = propertyDir

    def run(self):
        import conf
        cleaner = Cleaner(
            cdrDir=self.cdrDir, propertyDir=self.propertyDir,
            cleanCdrDir=conf.CLEAN_CDR_DIR, cleanPropertyDir=conf.CLEAN_PPT_DIR,
            dirtyCdrDir=conf.DIRTY_CDR_DIR, dirtyPropertyDir=conf.DIRTY_PPT_DIR,
            cleanCdrFmtFileName=conf.CLEAN_CDR_FORMAT_FILE, cleanPptFmtFileName=conf.CLEAN_PPT_FORMAT_FILE,
        )
        cleaner.run()
        if not cleaner.checkExistCleanData():
            logging.warn("No clean cdr or property")
            return
        Translator(
            cleanCdrDir=conf.CLEAN_CDR_DIR, cleanPropertyDir=conf.CLEAN_PPT_DIR,
            cleanCdrFmtFileName=conf.CLEAN_CDR_FORMAT_FILE, cleanPptFmtFileName=conf.CLEAN_PPT_FORMAT_FILE,
            translateCdrDir=conf.TRANSLATE_CDR_DIR, translatePropertyDir=conf.TRANSLATE_PPT_DIR,
            tlCdrFmtFileName=conf.TL_CDR_FORMAT_FILE, tlPptFmtFileName=conf.TL_PPT_FORMAT_FILE,
        ).run()
        CdrAggregator(
            translateCdrDir=conf.TRANSLATE_CDR_DIR, tlCdrFmtFileName=conf.TL_CDR_FORMAT_FILE,
            aggregateCdrDir=conf.AGGREGATE_CDR_DIR, aggCdrFmtFileName=conf.AGG_CDR_FORMAT_FILE,
            aggregateTimeUnit=bconf.AGGREGATE_TIME_UNIT,
        ).run()
        FeatureFrame3dConstructor(
            aggregateCdrDir=conf.AGGREGATE_CDR_DIR, aggCdrFmtFileName=conf.AGG_CDR_FORMAT_FILE,
            aggregateTimeUnit=bconf.AggregateTimeUnit.HOUR_1,
            translatePropertyDir=conf.TRANSLATE_PPT_DIR, tlPptFmtFileName=conf.TL_PPT_FORMAT_FILE,
            featureFrame3dDir=conf.FEATURE_FRAME_3D_DIR,
            shuffleFmtFileName=conf.SHUFFLE_FORMAT_FILE, ffFirstRowFmtFileName=conf.FF_FIRST_ROW_FORMAT_FILE,
            ffFirstRowFeatures=bconf.FeatureFrame3dDict.FIRST_ROW_FEATURES,
            ffFirstRowFmt=bconf.FeatureFrame3dDict.FIRST_ROW_FMT, shuffleFmt=bconf.FeatureFrame3dDict.SHUFFLE_FMT,
        ).run()
        TargetBehaviorConstructor(
            featureFrame3dDir=conf.FEATURE_FRAME_3D_DIR, ffFirstRowFmtFileName=conf.FF_FIRST_ROW_FORMAT_FILE,
            targetBehaviorDir=conf.TARGET_BEHAVIOR_DIR,
        ).run()
