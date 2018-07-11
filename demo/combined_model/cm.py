# coding=utf-8

import numpy as np
import tensorflow as tf


class DataTool(object):
    def __init__(self, binary_dim, x_dimension):
        super(DataTool, self).__init__()
        self.binary_dim = binary_dim
        self.x_dimension = x_dimension
        self.int2binary = None
        self.largest_number = None
        assert self.x_dimension == 2

    def init_data(self):
        # 一个字典，隐射一个数字到其二进制的表示
        # 例如 int2binary[3] = [0,0,0,0,0,0,1,1]
        self.int2binary = {}

        # 在8位情况下，最大数为2^8 = 256
        self.largest_number = pow(2, self.binary_dim)

        # 将[0,256)所有数表示成二进制
        binary = np.unpackbits(
            np.array([range(self.largest_number)], dtype=np.uint8).T, axis=1)

        # 建立字典
        for i in range(self.largest_number):
            self.int2binary[i] = binary[i]

    def binary_generation(self, numbers, reverse=False):
        '''
        返回numbers中所有数的二进制表达，
        例如 numbers = [3, 2, 1]
        返回 ：[[0,0,0,0,0,0,1,1],
                [0,0,0,0,0,0,1,0],
                [0,0,0,0,0,0,0,1]'

        如果 reverse = True, 二进制表达式前后颠倒，
        这么做是为训练方便，因为训练的输入顺序是从低位开始的

        numbers : 一组数字
        reverse : 是否将其二进制表示进行前后翻转
        '''
        binary_x = np.array([self.int2binary[num] for num in numbers], dtype=np.uint8)

        if reverse:
            binary_x = np.fliplr(binary_x)

        return binary_x

    def batch_generation(self, batch_size):
        '''
        生成batch_size大小的数据，用于训练或者验证

        batch_x 大小为[batch_size, biniary_dim, 2]
        batch_y 大小为[batch_size, biniray_dim]
        '''

        # 随机生成batch_size个数
        n1 = np.random.randint(0, self.largest_number // self.x_dimension, batch_size)
        n2 = np.random.randint(0, self.largest_number // self.x_dimension, batch_size)
        # 计算加法结果
        add = n1 + n2

        # int to binary
        binary_n1 = self.binary_generation(n1, True)
        binary_n2 = self.binary_generation(n2, True)
        batch_y = self.binary_generation(add, True)

        batch_x = np.ndarray([self.x_dimension * batch_size, self.binary_dim], dtype=np.uint8)
        for batchNo in xrange(batch_size):
            batch_x[batchNo * self.x_dimension] = binary_n1[batchNo]
            batch_x[batchNo * self.x_dimension + 1] = binary_n2[batchNo]
        assert self.binary_dim == FF_WIDTH
        batch_x = batch_x.reshape([self.x_dimension * batch_size, FF_DEPTH, FF_HEIGHT, FF_WIDTH, FF_CHANNEL])

        return batch_x, batch_y, n1, n2, add

    def binary2int(self, binary_array):
        '''
        将一个二进制数组转为整数
        '''
        out = 0
        for index, x in enumerate(reversed(binary_array)):
            out += x * pow(2, index)
        return out


FF_WIDTH = 8
FF_HEIGHT = 1
FF_DEPTH = 1
FF_CHANNEL = 1

DAY_CNT = 2
FV_LENGTH = 8

# TODO(20180710) remove dependence on BATCH_SIZE in rnn
BATCH_SIZE = 50

with tf.name_scope("Input") as _:
    x = tf.placeholder(tf.float32, [DAY_CNT * BATCH_SIZE, FF_DEPTH, FF_HEIGHT, FF_WIDTH, FF_CHANNEL], name='x')
    y_ = tf.placeholder(tf.float32, [BATCH_SIZE, FF_WIDTH], name='y_')
    keepProb3dCnn = tf.placeholder(tf.float32, name='keepProb3dCnn')
    keepProbRnn = tf.placeholder(tf.float32, name='keepProbRnn')

# with tf.name_scope("3dCnn") as _:
#     CONV_WIDTH = 2
#     CONV_HEIGHT = 1
#     CONV_DEPTH = 1
#
#     CONV_STRIDES_W = 1
#     CONV_STRIDES_H = 1
#     CONV_STRIDES_D = 1
#
#     POOL_WIDTH = 2
#     POOL_HEIGHT = 1
#     POOL_DEPTH = 1
#
#     POOL_STRIDES_W = 2
#     POOL_STRIDES_H = 1
#     POOL_STRIDES_D = 1
#
#     in1 = x
#     wConv1 = tf.Variable(tf.truncated_normal(
#         [CONV_DEPTH, CONV_HEIGHT, CONV_WIDTH, FF_CHANNEL, 1], stddev=0.1))
#     bConv1 = tf.Variable(tf.constant(0.1, shape=[1]))
#     hConv1 = tf.nn.relu(tf.nn.conv3d(
#         in1, wConv1, (1, CONV_STRIDES_D, CONV_STRIDES_H, CONV_STRIDES_W, 1), padding='SAME') + bConv1)
#     hPool1 = tf.nn.max_pool3d(
#         hConv1, [1, POOL_DEPTH, POOL_HEIGHT, POOL_WIDTH, 1],
#         strides=(1, POOL_STRIDES_D, POOL_STRIDES_H, POOL_STRIDES_W, 1), padding='SAME')
#
#     inFc = tf.reshape(
#         hPool1,
#         [DAY_CNT * BATCH_SIZE,
#          (hPool1.shape[1:][0] * hPool1.shape[1:][1] * hPool1.shape[1:][2] * hPool1.shape[1:][3]).value])
#     wFc2 = tf.Variable(tf.truncated_normal(
#         [inFc.shape[1:][0].value, FV_LENGTH], stddev=0.1))
#     bFc2 = tf.Variable(tf.constant(0.1, shape=[FV_LENGTH]))
#     output = tf.matmul(inFc, wFc2) + bFc2
#
#     v = tf.nn.dropout(output, keepProb3dCnn)

# with tf.name_scope("3dCnn") as _:
#     tmp = tf.reshape(x, [DAY_CNT * BATCH_SIZE, FF_DEPTH * FF_HEIGHT * FF_WIDTH * FF_CHANNEL])
#     weight = tf.Variable(tf.truncated_normal(
#         [FF_DEPTH * FF_HEIGHT * FF_WIDTH * FF_CHANNEL, FV_LENGTH], stddev=0.01))
#     bia = tf.zeros([FV_LENGTH])
#     v = tf.matmul(tmp, weight) + bia

with tf.name_scope("3dCnn") as _:
    assert FF_WIDTH <= FV_LENGTH
    v = tf.concat([
        tf.reshape(x, [DAY_CNT * BATCH_SIZE, FF_DEPTH * FF_HEIGHT * FF_WIDTH * FF_CHANNEL]),
        tf.zeros([DAY_CNT * BATCH_SIZE, FV_LENGTH - FF_WIDTH])], 1)

with tf.name_scope("Rnn") as _:
    LSTM_SIZE = 20

    combinedV = range(BATCH_SIZE)
    for batchNo in xrange(BATCH_SIZE):
        combinedV[batchNo] = tf.transpose(v[batchNo * DAY_CNT:(batchNo + 1) * DAY_CNT])
    combinedV = tf.convert_to_tensor(combinedV)

    cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(LSTM_SIZE) for _ in xrange(DAY_CNT)])
    # dropout
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keepProbRnn)
    # 进行forward，得到隐层的输出
    # 初始状态，可以理解为初始记忆
    initial_state = cell.zero_state(BATCH_SIZE, tf.float32)
    # outputs 大小为[batch_size, lstm_size*binary_dim]
    output, final_state = tf.nn.dynamic_rnn(cell, combinedV, initial_state=initial_state)

with tf.name_scope("Output") as _:
    # 建立输出层
    # [BATCH_SIZE, FV_LENGTH, LSTM_SIZE] ==> # [BATCH_SIZE, FV_LENGTH * LSTM_SIZE]
    output = tf.reshape(output, [BATCH_SIZE, FV_LENGTH * LSTM_SIZE])
    weight = tf.Variable(tf.truncated_normal([FV_LENGTH * LSTM_SIZE, FF_WIDTH], stddev=0.01))
    bia = tf.zeros([FF_WIDTH])
    # 得到输出, logits大小为[batch_size*binary_dim, 1]
    logit = tf.sigmoid(tf.matmul(output, weight) + bia)
    # [batch_size*binary_dim, 1] ==> [batch_size, binary_dim]
    prediction = tf.reshape(logit, [BATCH_SIZE, FF_WIDTH], name="prediction")

# with tf.name_scope("Output") as _:
#     # [BATCH_SIZE, FV_LENGTH, LSTM_SIZE] ==> # [BATCH_SIZE * FV_LENGTH, LSTM_SIZE]
#     output = tf.reshape(output, [BATCH_SIZE * FV_LENGTH, LSTM_SIZE])
#     weight1 = tf.Variable(tf.truncated_normal([LSTM_SIZE, 1], stddev=0.01))
#     bia1 = tf.zeros([1])
#     output = tf.matmul(output, weight1) + bia1
#
#     output = tf.reshape(output, [BATCH_SIZE, FV_LENGTH])
#     weight2 = tf.Variable(tf.truncated_normal([FV_LENGTH, FF_WIDTH], stddev=0.01))
#     bia2 = tf.zeros([FF_WIDTH])
#     output = tf.matmul(output, weight2) + bia2
#
#     logit = tf.sigmoid(output)
#     # [batch_size*binary_dim, 1] ==> [batch_size, binary_dim]
#     prediction = tf.reshape(logit, [BATCH_SIZE, FF_WIDTH], name="prediction")

with tf.name_scope("Optimizer") as _:
    loss = tf.losses.mean_squared_error(y_, prediction)
    optimizer = tf.train.AdamOptimizer().minimize(loss, name="optimizer")

dt = DataTool(FF_WIDTH, DAY_CNT)
dt.init_data()

with tf.Session() as sess:
    STEPS = 3000
    sess.run(tf.global_variables_initializer())
    for iter in xrange(1, 1 + STEPS):
        _input_x, _input_y, _, _, _ = dt.batch_generation(BATCH_SIZE)
        _combinedV, _, _loss = sess.run(
            [combinedV, optimizer, loss],
            feed_dict={x: _input_x, y_: _input_y, keepProb3dCnn: 0.5, keepProbRnn: 0.5})
        # _, _loss = sess.run(
        #     [optimizer, loss],
        #     feed_dict={x: _input_x, y_: _input_y, keepProb3dCnn: 0.5, keepProbRnn: 0.5})

        # flatten_input_x = np.reshape(_input_x, [DAY_CNT * BATCH_SIZE, FF_WIDTH])
        # xs = range(BATCH_SIZE)
        # for batchNo in xrange(BATCH_SIZE):
        #     xs[batchNo] = np.transpose(flatten_input_x[batchNo * DAY_CNT:(batchNo + 1) * DAY_CNT])
        # cvs = _combinedV
        # eqs = np.equal(xs, cvs)
        # if eqs.sum() != eqs.size:
        #     print("xs:\n%s" % xs)
        #     print("cvs:\n%s" % cvs)
        #     print("eqs:\n%s" % eqs)
        #     import sys
        #
        #     sys.exit(1)

        if iter % 100 == 0:
            print('Iter:{}, Loss:{}'.format(iter, _loss))
        iter += 1

    # 训练结束，进行测试
    val_x, val_y, n1, n2, add = dt.batch_generation(BATCH_SIZE)
    _prediction, _loss = sess.run(
        [prediction, loss],
        feed_dict={x: val_x, y_: val_y, keepProb3dCnn: 1.0, keepProbRnn: 1.0})
    # 左右翻转二进制数组。因为输出的结果是低位在前，而正常的表达是高位在前，因此进行翻转
    _prediction = np.fliplr(np.round(_prediction))
    _prediction = _prediction.astype(np.int32)
    print('Evaluate, Loss:{}'.format(_loss))
    val_x = val_x.reshape([DAY_CNT * BATCH_SIZE, FF_WIDTH])
    for batchNo in xrange(BATCH_SIZE):
        print('{}:{}'.format(val_x[batchNo, :], n1[batchNo]))
        print('{}:{}'.format(val_x[batchNo + 1, :], n2[batchNo]))
        print('{}:{}\n'.format(val_y[batchNo], dt.binary2int(_prediction[batchNo])))
