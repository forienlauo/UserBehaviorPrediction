# coding=utf-8
import logging
import os

import tensorflow as tf

import conf
from src.moduler.moduler import Moduler


class CmTrainer(Moduler):
    # cnn configuration
    CONV_DEPTH, CONV_HEIGHT, CONV_WIDTH = 6, 2, 5
    CONV_STRIDES_D, CONV_STRIDES_H, CONV_STRIDES_W = 1, 1, 1
    POOL_STRIDES_D, POOL_STRIDES_H, POOL_STRIDES_W = 2, 2, 2

    POOL_SHAPE = [1, 2, 2, 2, 1]  # [1, POOL_DEPTH, POOL_WIDTH, POOL_HEIGHT,1]

    def __init__(
            self,
            trainCmData=None, testCmData=None,
            wkdir=None,
            lstmSize=None, batchSizeConf=None, keepProbConf=None,
            cpuCoreCnt=None, gpuNos=None, gpuMemFraction=None,
            iteration=None, printProgressPerStepCnt=None,
    ):
        super(CmTrainer, self).__init__()
        self.trainCmData = trainCmData
        self.testCmData = testCmData
        self.wkdir = wkdir
        self.summaryDir = os.path.join(self.wkdir, "summary")

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
        batchSize, keepProb, x, y_, y, loss, optimizer = self.__construct()
        trainer = CmTrainer.Trainer(optimizer)
        evaluator = CmTrainer.Evaluator(loss)

        if self.gpuNos is not None:
            assert self.gpuMemFraction is not None
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpuNos
            gpuOptions = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpuMemFraction)
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
            trainer.train(
                sess, self.summaryDir,
                batchSize, keepProb, x, y_,
                self.batchSizeConf, self.keepProbConf,
                self.trainCmData, self.iteration, self.printProgressPerStepCnt,
                self.testCmData, evaluator,
            )

            # TODO(20180722) dump model

            logging.info("start to evaluate.")
            evaluate_result = evaluator.evaluate(
                sess,
                batchSize, keepProb, x, y_,
                self.testCmData,
            )
            logging.info("final evaluate_result: %s" % (evaluate_result,))

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

        with tf.name_scope('input') as _:
            # TODO(20180722) assert batchSize == x.shape[0] and batchSize == y_.shape[0]
            x = tf.placeholder(tf.float32, shape=[None] + ([learnDayCnt] + ff3dShape), name='x')
            y_ = tf.placeholder(tf.float32, shape=[None] + [1], name="y_")

        with tf.name_scope('featureMap') as _:
            # add channel
            ff3d = tf.reshape(x, [batchSize * learnDayCnt] + ff3dShape + [1], name="ff3d")
            fv = self.__constructFeatureMapScope(batchSize, ff3d, ff3dShape, fvLen, learnDayCnt)
            # fv = None

        with tf.name_scope('featurePredict') as _:
            predictFv = self.__constructFeaturePredictScope(batchSize, fv, fvLen, learnDayCnt)

        with tf.name_scope('dropout') as _:
            predictFv = tf.nn.dropout(predictFv, keepProb)

        with tf.name_scope('targetBehaviorResolve') as _:
            _tbrWeight = tf.Variable(tf.truncated_normal([fvLen, 1], stddev=0.01))
            _tbrBia = tf.zeros([1])
            y = tf.add(tf.matmul(predictFv, _tbrWeight), _tbrBia, name="y")

        with tf.name_scope('optimize') as _:
            # TODO(20180722) define name of loss
            loss = tf.losses.mean_squared_error(y_, y)
            optimizer = tf.train.AdamOptimizer().minimize(loss, name="optimizer")

        return batchSize, keepProb, x, y_, y, loss, optimizer

    def __constructFeatureMapScope(self, batchSize, ff3d, ff3dShape, fvLen, learnDayCnt):
        # # a simple impl
        # _elemCntPerFf3d = reduce(mul, ff3dShape)
        # _v = tf.reshape(ff3d, [batchSize * learnDayCnt] + [_elemCntPerFf3d])
        # _fmWeight = tf.Variable(tf.truncated_normal([_elemCntPerFf3d, fvLen], stddev=0.01))
        # _fmBia = tf.zeros([fvLen])
        # fv = tf.add(tf.matmul(_v, _fmWeight), _fmBia, name="fv")

        convDepth, convHeight, convWidth = CmTrainer.CONV_DEPTH, CmTrainer.CONV_HEIGHT, CmTrainer.CONV_WIDTH
        neuronsNums = [64, 128, 256]

        def weightVar(shape, name=None, ):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial, name=name, )

        def biaVar(shape, name=None, ):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial, name=name, )

        def conv3d(
                x, W,
                strides=(1, CmTrainer.CONV_STRIDES_D, CmTrainer.CONV_STRIDES_H, CmTrainer.CONV_STRIDES_W, 1),
                padding='SAME', name=None,
        ):
            return tf.nn.conv3d(x, W, strides, padding, name=name, )

        def max_pool3d(
                x, ksize,
                strides=(1, CmTrainer.POOL_STRIDES_D, CmTrainer.POOL_STRIDES_H, CmTrainer.POOL_STRIDES_W, 1),
                padding='SAME', name=None,
        ):
            return tf.nn.max_pool3d(x, ksize, strides, padding, name=name, )

        out0 = ff3d

        with tf.name_scope('C1') as _:
            _in = out0
            _depth, _height, _width = _in.get_shape()[1].value, _in.get_shape()[2].value, _in.get_shape()[3].value
            _inChannels = _in.get_shape()[4].value
            _outChannels = neuronsNums[0]

            _weight = weightVar(
                [convDepth, convHeight, convWidth, _inChannels, _outChannels], name='weight', )
            tf.summary.histogram('weight', _weight)
            _bia = biaVar([_outChannels], name='bia', )
            tf.summary.histogram('bia', _bia)
            # TODO(20180724) split conv node and activate node
            _convRs = tf.nn.relu(conv3d(_in, _weight) + _bia, name='convRs', )

            out = _convRs

        with tf.name_scope('S2') as _:
            _in = out

            _s2PoolRs = max_pool3d(_in, CmTrainer.POOL_SHAPE, name='s2PoolRs', )

            out = _s2PoolRs

        with tf.name_scope('C3') as _:
            _in = out
            _depth, _height, _width = _in.get_shape()[1].value, _in.get_shape()[2].value, _in.get_shape()[3].value
            _inChannels = _in.get_shape()[4].value
            _outChannels = neuronsNums[1]

            _weight = weightVar(
                [convDepth, convHeight, convWidth, _inChannels, _outChannels], name='weight', )
            tf.summary.histogram('weight', _weight)
            _bia = biaVar([_outChannels], name='bia', )
            tf.summary.histogram('bia', _bia)
            _convRs = tf.nn.relu(conv3d(_in, _weight) + _bia, name='convRs', )

            out = _convRs

        with tf.name_scope('S4') as _:
            _in = out

            _poolRs = max_pool3d(_in, CmTrainer.POOL_SHAPE, name='poolRs', )

            out = _poolRs

        with tf.name_scope('C5') as _:
            _in = out
            _depth, _height, _width = _in.get_shape()[1].value, _in.get_shape()[2].value, _in.get_shape()[3].value
            _inChannels = _in.get_shape()[4].value
            _outChannels = neuronsNums[2]

            _weight = weightVar(
                [convDepth, convHeight, convWidth, _inChannels, _outChannels], name='weight', )
            tf.summary.histogram('weight', _weight)
            _bia = biaVar([_outChannels], name='bia', )
            tf.summary.histogram('bia', _bia)
            _convRs = tf.nn.relu(conv3d(_in, _weight) + _bia, name='convRs', )

            out = _convRs

        with tf.name_scope('F6') as _:
            _in = out
            _depth, _height, _width = _in.get_shape()[1].value, _in.get_shape()[2].value, _in.get_shape()[3].value
            _inChannels = _in.get_shape()[4].value
            _outWidth = 1024

            _weight = weightVar([_depth * _height * _width * _inChannels, _outWidth], name='weight', )
            _bia = biaVar([_outWidth], name='bia', )
            _flatRs = tf.reshape(_in, [-1, _depth * _height * _width * _inChannels], name='flatRs', )
            _fcRs = tf.nn.relu(tf.matmul(_flatRs, _weight) + _bia, name='fcRs', )

            out = _fcRs

        with tf.name_scope('internalOutput') as _:
            _in = out
            _inWidth = _in.get_shape()[1].value
            _outWidth = fvLen

            _weight = weightVar([_inWidth, _outWidth], name='weight', )
            _bia = biaVar([_outWidth], name='bia', )
            fv = tf.add(tf.matmul(_in, _weight), _bia, name='fv', )

        return fv

    def __constructFeaturePredictScope(self, batchSize, fv, fvLen, learnDayCnt):
        _lstmSize = self.lstmSize
        _mFv = tf.transpose(tf.reshape(fv, [batchSize] + [learnDayCnt, fvLen]), perm=[0, 2, 1], name="mFv")
        _cell = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.BasicLSTMCell(_lstmSize) for _ in xrange(learnDayCnt)])
        # # dropout
        # _cell = tf.contrib.rnn.DropoutWrapper(_cell, output_keep_prob=keepProbRnn)
        _initialState = _cell.zero_state(batchSize, tf.float32)
        # predictFV: shape[None, fvLen, _lstmSize]
        _output, _final_state = tf.nn.dynamic_rnn(_cell, _mFv, initial_state=_initialState)
        _output = tf.reshape(_output, [-1] + [fvLen * _lstmSize], name="output")
        _fpWeight = tf.Variable(tf.truncated_normal([fvLen * _lstmSize, fvLen], stddev=0.01))
        _fpBia = tf.zeros([fvLen])
        predictFV = tf.add(tf.matmul(_output, _fpWeight), _fpBia, name="predictFV")
        return predictFV

    class Trainer(object):
        def __init__(self, optimizer):
            super(CmTrainer.Trainer, self).__init__()
            self.optimizer = optimizer

        def train(
                self,
                sess, summaryDir,
                batchSize, keepProb, x, y_,
                batchSizeConf, keepProbConf,
                trainCmData, iteration, printProgressPerStepCnt,
                testCmData, evaluator,
        ):
            optimizer = self.optimizer

            # summaries = tf.summary.merge_all()
            # summaryWriter = tf.summary.FileWriter(logdir=summaryDir, graph=sess.graph)

            sess.run(tf.global_variables_initializer())
            for stopNo in xrange(iteration):
                trainCmBatch = trainCmData.randomSampleBatch(batchSizeConf)
                # print progress
                if stopNo % printProgressPerStepCnt == 0:
                    trainEvlRs = evaluator.evaluate(
                        sess,
                        batchSize, keepProb, x, y_,
                        trainCmBatch,
                    )
                    testCmBatch = testCmData.randomSampleBatch(batchSizeConf)
                    testEvlRs = evaluator.evaluate(
                        sess,
                        batchSize, keepProb, x, y_,
                        testCmBatch,
                    )
                    logging.info(
                        "step %d before optimizing, training evaluate result: %s, testing evaluate result: %s"
                        % (stopNo, trainEvlRs, testEvlRs))
                # train
                feedDict = {batchSize: batchSizeConf, keepProb: keepProbConf,
                            x: trainCmBatch.learnMFf3ds, y_: trainCmBatch.predictTbs, }
                optimizer.run(feed_dict=feedDict, session=sess)
                # summarize
                # summariesV = summaries.eval(feed_dict=feedDict, session=sess)
                # summaryWriter.add_summary(summariesV, global_step=stopNo)

                # summaryWriter.close()

    class Evaluator(object):

        def __init__(self, loss, ):
            super(CmTrainer.Evaluator, self).__init__()
            self.loss = loss

        def evaluate(
                self,
                sess,
                batchSize, keepProb, x, y_,
                cmData,
        ):
            feedDict = {batchSize: cmData.exampleCnt, keepProb: 1.0, x: cmData.learnMFf3ds, y_: cmData.predictTbs, }
            lossV = self.loss.eval(feed_dict=feedDict, session=sess)  # v means value
            return CmTrainer.Evaluator.Result(cmData.exampleCnt, lossV, )

        class Result(object):
            def __init__(self, exampleCnt, lossV, ):
                super(CmTrainer.Evaluator.Result, self).__init__()
                self.lossV = lossV
                self.exampleCnt = exampleCnt

            def __str__(self):
                return "result {lossV: %.2f, example_cnt: %d}" % (self.lossV, self.exampleCnt,)
