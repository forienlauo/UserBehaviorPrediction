# coding=utf-8
import logging
import os

import conf
from src.moduler.constructor.feature_frame_con import FeatureFrame3dConstructor
from src.moduler.constructor.target_behavior_con import TargetBehaviorConstructor
from src.moduler.moduler import Moduler
from src.moduler.preprocessor.aggregator import CdrAggregator
from src.moduler.preprocessor.cleaner import Cleaner
from src.moduler.preprocessor.translator import Translator


class Processor(Moduler):
    def __init__(
            self,
            cdrDir=None, propertyDir=None,
            wkdir=None,
    ):
        super(Processor, self).__init__()
        self.cdrDir = cdrDir
        self.propertyDir = propertyDir
        self.wkdir = wkdir
        self.__initOtherMembers()

    def __initOtherMembers(self):
        wkdir = self.wkdir
        # cleaner
        self.cleanCdrDir = os.path.join(wkdir, "__clean", "cdr")
        self.cleanPptDir = os.path.join(wkdir, "__clean", "property")
        self.CLEAN_CDR_FORMAT_FILE = "__cdr.%s" % conf.FORMAT_FILE_SUFFIX
        self.CLEAN_PPT_FORMAT_FILE = "__property.%s" % conf.FORMAT_FILE_SUFFIX
        self.dirtyCdrDir = os.path.join(wkdir, "dirty", "cdr")
        self.dirtyPptDir = os.path.join(wkdir, "dirty", "property")
        # translator
        self.translateCdrDir = os.path.join(wkdir, "translate", "cdr")
        self.translatePptDir = os.path.join(wkdir, "translate", "property")
        self.TL_CDR_FORMAT_FILE = "__cdr.%s" % conf.FORMAT_FILE_SUFFIX
        self.TL_PPT_FORMAT_FILE = "__property.%s" % conf.FORMAT_FILE_SUFFIX
        # aggregator
        self.aggregateCdrDir = os.path.join(wkdir, "aggregate")
        self.AGG_CDR_FORMAT_FILE = "__cdr.%s" % conf.FORMAT_FILE_SUFFIX
        # feature frame 3d constructor
        self.featureFrame3dDir = os.path.join(wkdir, "featureFrame3d")
        self.SHUFFLE_FORMAT_FILE = "__shuffle.%s" % conf.FORMAT_FILE_SUFFIX
        self.FF_FIRST_ROW_FORMAT_FILE = "__ffFirstRow.%s" % conf.FORMAT_FILE_SUFFIX
        # target behavior constructor
        self.targetBehaviorDir = os.path.join(wkdir, "targetBehavior")

    def run(self):
        import conf
        # TODO(20180707) print progress
        cleaner = Cleaner(
            cdrDir=self.cdrDir, propertyDir=self.propertyDir,
            cleanCdrDir=self.cleanCdrDir, cleanPropertyDir=self.cleanPptDir,
            dirtyCdrDir=self.dirtyCdrDir, dirtyPropertyDir=self.dirtyPptDir,
            cleanCdrFmtFileName=self.CLEAN_CDR_FORMAT_FILE, cleanPptFmtFileName=self.CLEAN_PPT_FORMAT_FILE,
        )
        cleaner.run()
        if not cleaner.checkExistCleanData():
            logging.warn("No clean cdr or property")
            return

        Translator(
            cleanCdrDir=self.cleanCdrDir, cleanPropertyDir=self.cleanPptDir,
            cleanCdrFmtFileName=self.CLEAN_CDR_FORMAT_FILE, cleanPptFmtFileName=self.CLEAN_PPT_FORMAT_FILE,
            translateCdrDir=self.translateCdrDir, translatePropertyDir=self.translatePptDir,
            tlCdrFmtFileName=self.TL_CDR_FORMAT_FILE, tlPptFmtFileName=self.TL_PPT_FORMAT_FILE,
        ).run()
        CdrAggregator(
            translateCdrDir=self.translateCdrDir, tlCdrFmtFileName=self.TL_CDR_FORMAT_FILE,
            aggregateCdrDir=self.aggregateCdrDir, aggCdrFmtFileName=self.AGG_CDR_FORMAT_FILE,
            aggregateTimeUnit=conf.AggregateCdrDict.AGGREGATE_TIME_UNIT,
            aggregateFeatures=conf.AggregateCdrDict.AGGREGATE_FEATURES,
            aggregateFmt=conf.AggregateCdrDict.AGGREGATE_FMT,
        ).run()

        FeatureFrame3dConstructor(
            aggregateCdrDir=self.aggregateCdrDir, aggCdrFmtFileName=self.AGG_CDR_FORMAT_FILE,
            aggregateTimeUnit=conf.AggregateCdrDict.AGGREGATE_TIME_UNIT,
            translatePropertyDir=self.translatePptDir, tlPptFmtFileName=self.TL_PPT_FORMAT_FILE,
            featureFrame3dDir=self.featureFrame3dDir,
            depthTimeUnit=conf.FeatureFrame3dDict.DEPTH_TIME_UNIT,
            shuffleFmtFileName=self.SHUFFLE_FORMAT_FILE, ffFirstRowFmtFileName=self.FF_FIRST_ROW_FORMAT_FILE,
            ffFirstRowFeatures=conf.FeatureFrame3dDict.FIRST_ROW_FEATURES,
            ffFirstRowFmt=conf.FeatureFrame3dDict.FIRST_ROW_FMT, shuffleFmt=conf.FeatureFrame3dDict.SHUFFLE_FMT,
        ).run()
        TargetBehaviorConstructor(
            featureFrame3dDir=self.featureFrame3dDir, ffFirstRowFmtFileName=self.FF_FIRST_ROW_FORMAT_FILE,
            targetBehaviorDir=self.targetBehaviorDir,
        ).run()
