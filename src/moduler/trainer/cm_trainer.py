# coding=utf-8
import logging
import os

import tensorflow as tf

import conf
from src.moduler.moduler import Moduler


class CmTrainer(Moduler):
    SUMMARIZE_IMAGE = False

    def __init__(
            self,
            trainData=None, testData=None,
            wkdir=None,
            convShape=None, convStrides=None, poolShape=None, poolStrides=None, convCnts=None,
            lstmSize=None,
            batchSizeConf=None, keepProbConf=None,
            cpuCoreCnt=None, gpuNos=None, gpuMemFraction=None,
            iteration=None, printProgressPerStepCnt=None,
    ):
        super(CmTrainer, self).__init__()
        self.trainData = trainData
        self.testData = testData
        self.wkdir = wkdir
        self.summaryDir = os.path.join(self.wkdir, "summary")

        self.convShape = convShape
        self.convStrides = convStrides
        self.poolShape = poolShape
        self.poolStrides = poolStrides
        self.convCnts = convCnts

        self.lstmSize = lstmSize

        self.batchSizeConf = batchSizeConf
        self.keepProbConf = keepProbConf

        self.cpuCoreCnt = cpuCoreCnt
        self.gpuNos = gpuNos
        self.gpuMemFraction = gpuMemFraction
        self.iteration = iteration
        self.printProgressPerStepCnt = printProgressPerStepCnt

    def run(self):
        self.__init()
        runConf, runInput, trainer, evaluator = self.__construct()

        if self.gpuNos is not None:
            assert self.gpuMemFraction is not None
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpuNos
            gpuOptions = tf.GPUOptions(
                per_process_gpu_memory_fraction=self.gpuMemFraction,
                allow_growth=True,
            )
            config = tf.ConfigProto(gpu_options=gpuOptions)
        else:
            assert self.cpuCoreCnt is not None
            config = tf.ConfigProto(
                device_count={"CPU": self.cpuCoreCnt},
                inter_op_parallelism_threads=self.cpuCoreCnt,
                intra_op_parallelism_threads=self.cpuCoreCnt,
            )

        logging.info("start to train.")
        with tf.Session(config=config) as sess:
            runConf.set(self.batchSizeConf, self.keepProbConf)
            trainer.train(
                sess, self.summaryDir,
                runConf, runInput,
                self.trainData, self.iteration, self.printProgressPerStepCnt,
                self.testData, evaluator,
            )

            # TODO(20180722) dump model

            # logging.info("start to evaluate.")
            # evaluate_result = evaluator.evaluate(
            #     sess,
            #     runConf, runInput,
            #     self.testData,
            # )
            # logging.info("final evaluate_result: %s" % (evaluate_result,))

    def __init(self):
        os.mkdir(self.summaryDir)

    def __construct(self, ):
        learnDayCnt = conf.TrainerDict.LEARN_DAY_CNT
        ff3dShape = conf.FeatureFrame3dDict.FF3D_SHAPE
        fvLen = conf.FeatureVectorDict.FEATURE_VECTOR_LEN

        with tf.name_scope('conf') as _:
            # must use [] as shape
            batchSize = tf.placeholder(tf.int32, shape=[], name="batchSize")
            keepProb = tf.placeholder(tf.float32, name="keepProb")

            runConf = CmTrainer._RunConf(batchSize, keepProb)

        with tf.name_scope('input') as _:
            # TODO(20180722) assert batchSize == x.shape[0] and batchSize == y_.shape[0]
            x = tf.placeholder(tf.float32, shape=[None] + ([learnDayCnt] + ff3dShape), name='x')
            y_ = tf.placeholder(tf.float32, shape=[None] + [1], name="y_")

            tf.summary.histogram("realX", x)
            tf.summary.histogram("realY", y_)

            runInput = CmTrainer._RunInput(x, y_)

        with tf.name_scope('featureMap') as _:
            fv = self.__constructFeatureMapScope(batchSize, x, ff3dShape, fvLen, learnDayCnt)

            tf.summary.histogram("realFvBeforeDropout", fv)

        with tf.name_scope('dropout') as _:
            fv = tf.layers.dropout(fv, 1.0 - keepProb)

            tf.summary.histogram("realFvAfterDropout", fv)

        with tf.name_scope('featurePredict') as _:
            predictFv = self.__constructFeaturePredictScope(batchSize, fv, fvLen, learnDayCnt)

            tf.summary.histogram("predictFv", predictFv)

        with tf.name_scope('targetBehaviorResolve') as _:
            y = self.__constructtargetBehaviorResolve(predictFv, fvLen)

            tf.summary.histogram("predictY", y)

        with tf.name_scope('optimize') as _:
            lossMse, lossRmse, lossMae, lossR2, lossRrmse, lossMape = self.__constructLoss(y, y_, batchSize)

            with tf.name_scope("internalOptimize") as _:
                logging.info("optimize by lossMse")
                optimize = tf.train.AdamOptimizer().minimize(lossMse, name="optimize")

            evaluator = CmTrainer._Evaluator(lossMse, lossRmse, lossMae, lossR2, lossRrmse, lossMape)
            trainer = CmTrainer._Trainer(optimize)

        return runConf, runInput, trainer, evaluator

    def __constructFeatureMapScope(self, batchSize, x, ff3dShape, fvLen, learnDayCnt):
        convShape = self.convShape
        convStrides = self.convStrides
        poolShape = self.poolShape
        poolStrides = self.poolStrides
        convCnts = self.convCnts

        def conv3d(
                x, filter,
                strides, padding='SAME',
                name=None,
        ):
            return tf.nn.conv3d(x, filter, strides, padding, name=name, )

        def max_pool3d(
                x, ksize,
                strides, padding='SAME',
                name=None,
        ):
            return tf.nn.max_pool3d(x, ksize, strides, padding, name=name, )

        def addFf2d2summary(x, imageNamePrefix):
            if not CmTrainer.SUMMARIZE_IMAGE:
                return
            depth = x.get_shape()[1].value
            channels = x.get_shape()[4].value
            for depthNo in xrange(depth):
                for channelNo in xrange(channels):
                    # FIXME(20180730) covered old images with the same name
                    image = x[:, depthNo:depthNo + 1, :, :, channelNo:channelNo + 1]
                    image = tf.reshape(image, [-1] + map(lambda _: _.value, image.get_shape()[2:]))
                    imageName = '%s-%d-%d' % (imageNamePrefix, depthNo, channelNo,)
                    tf.summary.image(imageName, image)

        with tf.name_scope('internalInput') as _:
            ff3d = tf.reshape(x, [batchSize * learnDayCnt] + ff3dShape + [1], name="ff3d")
            out0 = ff3d

            addFf2d2summary(ff3d, "ff3d")

        with tf.name_scope('C1') as _:
            _in = out0
            _depth, _height, _width = _in.get_shape()[1].value, _in.get_shape()[2].value, _in.get_shape()[3].value
            _inChannels = _in.get_shape()[4].value
            _outChannels = convCnts[0]

            _weight = self.__weightVar(convShape + [_inChannels, _outChannels], name='weight', )
            _bia = self.__biaVar([_outChannels], name='bia', )
            # TODO(20180724) split conv node and activate node
            _convRs = tf.nn.relu(
                conv3d(_in, _weight, strides=[1] + convStrides + [1]) + _bia, name='convRs', )

            out = _convRs

            addFf2d2summary(out, "image")
            tf.summary.histogram('weight', _weight)
            tf.summary.histogram('bia', _bia)

        with tf.name_scope('S2') as _:
            _in = out

            _poolRs = max_pool3d(
                _in, [1] + poolShape + [1], strides=[1] + poolStrides + [1], name='poolRs', )

            out = _poolRs

            addFf2d2summary(out, "image")

        with tf.name_scope('C3') as _:
            _in = out
            _depth, _height, _width = _in.get_shape()[1].value, _in.get_shape()[2].value, _in.get_shape()[3].value
            _inChannels = _in.get_shape()[4].value
            _outChannels = convCnts[1]

            _weight = self.__weightVar(
                convShape + [_inChannels, _outChannels], name='weight', )
            _bia = self.__biaVar([_outChannels], name='bia', )
            _convRs = tf.nn.relu(
                conv3d(_in, _weight, strides=[1] + convStrides + [1]) + _bia, name='convRs', )

            out = _convRs

            addFf2d2summary(out, "image")
            tf.summary.histogram('weight', _weight)
            tf.summary.histogram('bia', _bia)

        with tf.name_scope('S4') as _:
            _in = out

            _poolRs = max_pool3d(
                _in, [1] + poolShape + [1], strides=[1] + poolStrides + [1], name='poolRs', )

            out = _poolRs

            addFf2d2summary(out, "image")

        with tf.name_scope('C5') as _:
            _in = out
            _depth, _height, _width = _in.get_shape()[1].value, _in.get_shape()[2].value, _in.get_shape()[3].value
            _inChannels = _in.get_shape()[4].value
            _outChannels = convCnts[2]

            _weight = self.__weightVar(
                convShape + [_inChannels, _outChannels], name='weight', )
            _bia = self.__biaVar([_outChannels], name='bia', )
            _convRs = tf.nn.relu(
                conv3d(_in, _weight, strides=[1] + convStrides + [1]) + _bia, name='convRs', )

            out = _convRs

            addFf2d2summary(out, "image")
            tf.summary.histogram('weight', _weight)
            tf.summary.histogram('bia', _bia)

        with tf.name_scope('F6') as _:
            _in = out
            _depth, _height, _width = _in.get_shape()[1].value, _in.get_shape()[2].value, _in.get_shape()[3].value
            _inChannels = _in.get_shape()[4].value
            _outWidth = 1024

            _weight = self.__weightVar([_depth * _height * _width * _inChannels, _outWidth], name='weight', )
            _bia = self.__biaVar([_outWidth], name='bia', )
            _flatRs = tf.reshape(_in, [-1, _depth * _height * _width * _inChannels], name='flatRs', )
            _fcRs = tf.nn.relu(tf.matmul(_flatRs, _weight) + _bia, name='fcRs', )

            out = _fcRs

            tf.summary.histogram('weight', _weight)
            tf.summary.histogram('bia', _bia)

        with tf.name_scope('internalOutput') as _:
            _in = out
            _inWidth = _in.get_shape()[1].value
            _outWidth = fvLen

            _weight = self.__weightVar([_inWidth, _outWidth], name='weight', )
            _bia = self.__biaVar([_outWidth], name='bia', )
            fv = tf.add(tf.matmul(_in, _weight), _bia, name='fv', )

            tf.summary.histogram('weight', _weight)
            tf.summary.histogram('bia', _bia)

        return fv

    def __constructFeaturePredictScope(self, batchSize, fv, fvLen, learnDayCnt):
        lstmSize = self.lstmSize

        def lstmCell(lstmSize):
            return tf.contrib.rnn.BasicLSTMCell(lstmSize)

        # TODO(20180729) impl
        def addVec2summary(vec, vecName):
            if not CmTrainer.SUMMARIZE_IMAGE:
                return
            # FIXME(20180730) covered old images with the same name
            vecLen = vec.shape[1].value
            vec = tf.reshape(vec, [-1] + [1, vecLen, 1])
            tf.summary.image(vecName, vec)

        with tf.name_scope('internalInput') as _:
            addVec2summary(fv, "historyFv")

            _mFv = tf.transpose(tf.reshape(fv, [batchSize] + [learnDayCnt, fvLen]), perm=[0, 2, 1], name="mFv")

        with tf.name_scope('lstmCells') as _:
            _cell = tf.contrib.rnn.MultiRNNCell([lstmCell(lstmSize) for _ in xrange(learnDayCnt)])
            _initialState = _cell.zero_state(batchSize, tf.float32)
            # _output: shape[None, fvLen, lstmSize]
            _output, _final_state = tf.nn.dynamic_rnn(_cell, _mFv, initial_state=_initialState)

        with tf.name_scope('internalOutput') as _:
            _output = tf.reshape(_output, [-1] + [fvLen * lstmSize])
            _weight = self.__weightVar([fvLen * lstmSize, fvLen], name='weight', )
            _bia = self.__biaVar([fvLen], name='bia', )
            predictFv = tf.add(tf.matmul(_output, _weight), _bia, name="predictFv")

            addVec2summary(fv, "predictFv")
            tf.summary.histogram('weight', _weight)
            tf.summary.histogram('bia', _bia)

        return predictFv

    def __constructLoss(self, y, y_, batchSize):
        # cannot define name in the way below
        # lossMse = tf.losses.mean_squared_error(y_, y)
        _batchSizeF = tf.cast(batchSize, tf.float32)
        with tf.name_scope("lossMse") as _:
            _lossSe = tf.square(tf.subtract(y, y_))  # squared error
            lossMse = tf.div(tf.reduce_sum(_lossSe), _batchSizeF, name="lossMse")  # mean squared error
        with tf.name_scope("lossRmse") as _:
            lossRmse = tf.sqrt(lossMse, name="lossRmse")  # root mean squared error
        with tf.name_scope("lossMae") as _:
            _lossAe = tf.abs(tf.subtract(y, y_))  # abstract error
            lossMae = tf.div(tf.reduce_sum(_lossAe), _batchSizeF, name="lossMae")  # mean abstract error
        with tf.name_scope("lossR2") as _:
            _lossSr = tf.square(tf.subtract(tf.reduce_mean(y), y_))  # squared residual
            lossR2 = tf.subtract(tf.ones([]), tf.div(tf.reduce_sum(_lossSe), tf.reduce_sum(_lossSr)),
                                 name="lossR2")  # r2
        with tf.name_scope("lossRrmse") as _:
            lossRrmse = tf.div(lossRmse, tf.reduce_mean(y_), name="lossRrmse")
        with tf.name_scope("lossMape") as _:
            lossMape = tf.div(tf.reduce_sum(tf.div(_lossAe, y_ + 0.01)), _batchSizeF,
                              name="lossMape")

        tf.summary.scalar('lossMse', lossMse)
        tf.summary.scalar('lossRmse', lossRmse)
        tf.summary.scalar('lossMae', lossMae)
        tf.summary.scalar('lossR2', lossR2)
        tf.summary.scalar('lossRrmse', lossRrmse)
        tf.summary.scalar('lossMape', lossMape)

        return lossMse, lossRmse, lossMae, lossR2, lossRrmse, lossMape

    def __constructtargetBehaviorResolve(self, predictFv, fvLen):
        _weight = self.__weightVar([fvLen, 1], name='weight', )
        _bia = self.__biaVar([1], name='bia', )
        y = tf.add(tf.matmul(predictFv, _weight), _bia, name="y")

        tf.summary.histogram("weight", _weight)
        tf.summary.histogram("bia", _bia)

        return y

    def __weightVar(self, shape, name=None, ):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name, )

    def __biaVar(self, shape, name=None, ):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name, )

    class _RunConf(object):
        def __init__(self, batchSize, keepProb):
            super(CmTrainer._RunConf, self).__init__()
            self.batchSize = batchSize
            self.keepProb = keepProb

            self.batchSizeConf = None
            self.keepProbConf = None

        def set(self, batchSizeConf, keepProbConf):
            self.batchSizeConf = batchSizeConf
            self.keepProbConf = keepProbConf

        def clear(self):
            self.batchSizeConf = None
            self.keepProbConf = None

    class _RunInput(object):
        def __init__(self, x, y_):
            super(CmTrainer._RunInput, self).__init__()
            self.x = x
            self.y_ = y_

    class _Trainer(object):
        def __init__(self, optimize):
            super(CmTrainer._Trainer, self).__init__()
            self.optimize = optimize

        def train(
                self,
                sess, summaryDir,
                runConf, runInput,
                trainData, iteration, printProgressPerStepCnt,
                testData, evaluator,
        ):
            optimize = self.optimize

            summaries = tf.summary.merge_all()
            trainSummaryWriter = tf.summary.FileWriter(logdir=os.path.join(summaryDir, "train"), graph=sess.graph)
            testSummaryWriter = tf.summary.FileWriter(logdir=os.path.join(summaryDir, "test"), graph=sess.graph)

            runOptions = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            runMetadata = tf.RunMetadata()

            sess.run(tf.global_variables_initializer())
            for stopNo in xrange(iteration):
                trainBatch = trainData.randomSampleBatch(runConf.batchSizeConf)
                trainFeedDict = {runConf.batchSize: runConf.batchSizeConf, runConf.keepProb: runConf.keepProbConf,
                                 runInput.x: trainBatch.learnMFf3ds, runInput.y_: trainBatch.predictTbs, }
                sess.run(optimize, feed_dict=trainFeedDict, options=runOptions, run_metadata=runMetadata)
                trainSummariesV = summaries.eval(feed_dict=trainFeedDict, session=sess)
                trainSummaryWriter.add_summary(trainSummariesV, global_step=stopNo)

                testBatch = testData.randomSampleBatch(runConf.batchSizeConf)
                testFeedDict = {runConf.batchSize: runConf.batchSizeConf, runConf.keepProb: 1.0,
                                runInput.x: testBatch.learnMFf3ds, runInput.y_: testBatch.predictTbs, }
                sess.run(optimize, feed_dict=testFeedDict, options=runOptions, run_metadata=runMetadata)
                testSummariesV = summaries.eval(feed_dict=testFeedDict, session=sess)
                testSummaryWriter.add_summary(testSummariesV, global_step=stopNo)

                # print progress
                if stopNo % printProgressPerStepCnt == 0:
                    trainEvlRs = evaluator.evaluate(
                        sess,
                        runConf, runInput,
                        trainBatch,
                    )
                    testEvlRs = evaluator.evaluate(
                        sess,
                        runConf, runInput,
                        testBatch,
                    )
                    logging.info(
                        "step %d before optimizing, training evaluate result: %s, testing evaluate result: %s"
                        % (stopNo, trainEvlRs, testEvlRs))

            trainSummaryWriter.close()
            testSummaryWriter.close()

    class _Evaluator(object):

        def __init__(
                self,
                lossMse, lossRmse, lossMae, lossR2, lossRrmse, lossMape,
        ):
            super(CmTrainer._Evaluator, self).__init__()
            self.lossMse = lossMse
            self.lossRmse = lossRmse
            self.lossMae = lossMae
            self.lossR2 = lossR2
            self.lossRrmse = lossRrmse
            self.lossMape = lossMape

        def evaluate(
                self,
                sess,
                runConf, runInput,
                cmData,
        ):
            feedDict = {runConf.batchSize: cmData.exampleCnt, runConf.keepProb: 1.0,
                        runInput.x: cmData.learnMFf3ds, runInput.y_: cmData.predictTbs, }
            lossMseV, lossRmseV, lossMaeV, lossR2V, lossRrmseV, lossMapeV = sess.run(
                [self.lossMse, self.lossRmse, self.lossMae, self.lossR2, self.lossRrmse, self.lossMape],
                feed_dict=feedDict, )
            return CmTrainer._Evaluator.Result(
                cmData.exampleCnt,
                lossMseV, lossRmseV, lossMaeV, lossR2V, lossRrmseV, lossMapeV,
            )

        class Result(object):
            def __init__(
                    self,
                    exampleCnt,
                    lossMse, lossRmse, lossMae, lossR2, lossRrmse, lossMape,
            ):
                super(CmTrainer._Evaluator.Result, self).__init__()
                self.exampleCnt = exampleCnt

                self.lossMse = lossMse
                self.lossRmse = lossRmse
                self.lossMae = lossMae
                self.lossR2 = lossR2
                self.lossRrmse = lossRrmse
                self.lossMape = lossMape

            def __str__(self):
                return (
                    "result {example_cnt: %d,"
                    " lossMse: %.6f, lossRmse: %.6f, lossMae: %.6f, lossR2: %.6f, lossRrmse: %.6f, lossMape: %.6f}"
                    % (self.exampleCnt,
                       self.lossMse, self.lossRmse, self.lossMae, self.lossR2, self.lossRrmse, self.lossMape,)
                )
