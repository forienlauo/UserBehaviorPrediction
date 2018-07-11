# coding=utf-8
import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
# mnist = read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf


def demo_3d_cnn_with_mnist():
    # import tensorflow.contrib.slim as slim
    # configuration
    INITIAL_DEPTH, INITIAL_HEIGHT, INITIAL_WIDTH, INITIAL_IN_CHANELS = 4, 14, 14, 1
    TARGET_CLASS_CNT = 10

    CONV_STRIDES_D, CONV_STRIDES_H, CONV_STRIDES_W = 1, 1, 1
    CONV_DEPTH, CONV_HEIGHT, CONV_WIDTH = 2, 5, 5

    POOL_STRIDES_D, POOL_STRIDES_H, POOL_STRIDES_W = 2, 2, 2
    POOL_SHAPE = [1, 2, 2, 2, 1]  # [1, POOL_DEPTH, POOL_WIDTH, POOL_HEIGHT,1]

    KEEP_PROB = 0.5

    ITERATION = 1000
    BATCH_SIZE = 50

    # input and labels
    x = tf.placeholder(tf.float32, shape=[None, INITIAL_DEPTH * INITIAL_HEIGHT * INITIAL_WIDTH * INITIAL_IN_CHANELS])
    y_ = tf.placeholder(tf.float32, shape=[None, TARGET_CLASS_CNT])

    def weight_variable(shape):
        """
        :param shape: usually used as shape of filter
            filter: A `Tensor`. Must have the same type as `input`.
          A 4-D tensor of shape
          `[filter_height, filter_width, in_channels, out_channels]`
        :return:
        """
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        """

        :param shape: usually used as shape of bias
            bias: A `Tensor`. Must have the same type as `output`.
          A 1-D tensor of shape
          `[out_channels]`
        :return:
        """
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv3d(x, W, strides=(1, CONV_STRIDES_D, CONV_STRIDES_H, CONV_STRIDES_W, 1), padding='SAME'):
        r"""Computes a 3-D convolution given 5-D `input` and `filter` tensors.

          In signal processing, cross-correlation is a measure of similarity of
          two waveforms as a function of a time-lag applied to one of them. This
          is also known as a sliding dot product or sliding inner-product.

          Our Conv3D implements a form of cross-correlation.

          Args:
            input: A `Tensor`. Must be one of the following types: `float32`, `float64`.
              Shape `[batch, in_depth, in_height, in_width, in_channels]`.
            filter: A `Tensor`. Must have the same type as `input`.
              Shape `[filter_depth, filter_height, filter_width, in_channels,
              out_channels]`. `in_channels` must match between `input` and `filter`.
            strides: A list of `ints` that has length `>= 5`.
              1-D tensor of length 5. The stride of the sliding window for each
              dimension of `input`. Must have `strides[0] = strides[4] = 1`.
            padding: A `string` from: `"SAME", "VALID"`.
              The type of padding algorithm to use.
            data_format: An optional `string` from: `"NDHWC", "NCDHW"`. Defaults to `"NDHWC"`.
              The data format of the input and output data. With the
              default format "NDHWC", the data is stored in the order of:
                  [batch, in_depth, in_height, in_width, in_channels].
              Alternatively, the format could be "NCDHW", the data storage order is:
                  [batch, in_channels, in_depth, in_height, in_width].
            name: A name for the operation (optional).

          Returns:
            A `Tensor`. Has the same type as `input`.
          """
        # 如果使用默认步长strides[depth, height, width]=[1,1,1], 则卷积后的图片大小不变
        return tf.nn.conv3d(x, W, strides, padding)

    def max_pool3d(x, ksize, strides=(1, POOL_STRIDES_D, POOL_STRIDES_H, POOL_STRIDES_W, 1), padding='SAME'):
        r"""Performs 3D max pooling on the input.

        Args:
          input: A `Tensor`. Must be one of the following types: `float32`.
            Shape `[batch, depth, rows, cols, channels]` tensor to pool over.
          ksize: A list of `ints` that has length `>= 5`.
            1-D tensor of length 5. The size of the window for each dimension of
            the input tensor. Must have `ksize[0] = ksize[4] = 1`.
          strides: A list of `ints` that has length `>= 5`.
            1-D tensor of length 5. The stride of the sliding window for each
            dimension of `input`. Must have `strides[0] = strides[4] = 1`.
          padding: A `string` from: `"SAME", "VALID"`.
            The type of padding algorithm to use.
          data_format: An optional `string` from: `"NDHWC", "NCDHW"`. Defaults to `"NDHWC"`.
            The data format of the input and output data. With the
            default format "NDHWC", the data is stored in the order of:
                [batch, in_depth, in_height, in_width, in_channels].
            Alternatively, the format could be "NCDHW", the data storage order is:
                [batch, in_channels, in_depth, in_height, in_width].
          name: A name for the operation (optional).

        Returns:
          A `Tensor`. Has the same type as `input`. The max pooled output tensor.
        """
        # 如果使用默认步长strides[depth, height, width]=[2,2,2], 则卷积后的图片大小变为原来的 1/8
        return tf.nn.max_pool3d(x, ksize, strides, padding)

    # First Convolutional Layer
    _in_channels1 = INITIAL_IN_CHANELS
    _out_channels1 = 32
    _example_cnt, _depth1, _height1, _width1 = -1, INITIAL_DEPTH, INITIAL_HEIGHT, INITIAL_WIDTH
    x_image = tf.reshape(x, [_example_cnt, _depth1, _height1, _width1, _in_channels1])
    W_conv1 = weight_variable([CONV_DEPTH, CONV_HEIGHT, CONV_WIDTH, _in_channels1, _out_channels1])
    b_conv1 = bias_variable([_out_channels1])
    in1 = x_image
    h_conv1 = tf.nn.relu(conv3d(in1, W_conv1) + b_conv1)
    h_pool1 = max_pool3d(h_conv1, POOL_SHAPE)

    out1 = h_pool1

    # Second Convolutional Layer
    in2 = out1
    _example_cnt, _depth2, _height2, _width2 = \
        -1, in2.get_shape()[1].value, in2.get_shape()[2].value, in2.get_shape()[3].value
    _in_channels2 = _out_channels1
    _out_channels2 = 64

    W_conv2 = weight_variable([CONV_DEPTH, CONV_HEIGHT, CONV_WIDTH, _in_channels2, _out_channels2])
    b_conv2 = bias_variable([_out_channels2])
    h_conv2 = tf.nn.relu(conv3d(in2, W_conv2) + b_conv2)
    h_pool2 = max_pool3d(h_conv2, POOL_SHAPE)

    out2 = h_pool2

    # Densely Connected Layer(Full Connected Layer)
    in3 = out2
    _example_cnt, _depth3, _height3, _width3 = \
        -1, in3.get_shape()[1].value, in3.get_shape()[2].value, in3.get_shape()[3].value
    _in_channels3 = _out_channels2
    _out_channels3 = 1024

    W_fc1 = weight_variable([_depth3 * _height3 * _width3 * _in_channels3, _out_channels3])
    b_fc1 = bias_variable([_out_channels3])
    h_pool2_flat = tf.reshape(in3, [_example_cnt, _depth3 * _height3 * _width3 * _in_channels3])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # matrix multiple

    out3 = h_fc1

    # Dropout
    keep_prob = tf.placeholder(tf.float32)
    tmp_in = out3
    h_fc1_drop = tf.nn.dropout(tmp_in, keep_prob)
    tmp_out = h_fc1_drop

    out3 = tmp_out

    # Readout Layer
    in4 = out3
    _in_width4 = in4.get_shape()[1].value
    _out_width4 = TARGET_CLASS_CNT
    W_fc2 = weight_variable([_in_width4, _out_width4])
    b_fc2 = bias_variable([_out_width4])
    y_conv = tf.matmul(in4, W_fc2) + b_fc2  # matrix multiple

    out4 = y_conv

    # Train
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_per_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(ITERATION):
            # batch = mnist.fit.next_batch(BATCH_SIZE)
            batch = mnist.train.next_batch(BATCH_SIZE)
            # print progress
            if i % 10 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                print("step %d, training accuracy %g" % (i, train_accuracy))
            train_per_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: KEEP_PROB})

        # Evaluate
        print("test accuracy %g" % accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
        # test accuracy 0.9914


if __name__ == '__main__':
    # demo_softmax_with_mnist()
    # demo_2d_cnn_with_mnist()
    demo_3d_cnn_with_mnist()
    pass
