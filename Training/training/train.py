# -*- coding: utf-8 -*-
"""
@author ozzi7
"""

import numpy as np
import tensorflow as tf


class Trainer:
    def __init__(self):
        self.NOF_FILTERS = 64
        self.NOF_RESIDUAL_BLOCKS = 6
        self.NOF_PLANES = 2
        self.BOARD_SIZE_X = 3
        self.BOARD_SIZE_Y = 3

        self.weights = [] # to export
        construct_net(self.NOF_PLANES)

    def train(self):
        pass

    def residual_block(self, inputs, nof_filters):
        """
        :param inputs:
        :param nof_filters:
        :return:
        """
        # First convnet
        passthrough = tf.identity(inputs)
        conv_weights_1 = weight_variable([3, 3, nof_filters, nof_filters])
        conv_biases_1 = bn_bias_variable([nof_filters])
        self.weights.append(conv_weights_1)
        self.weights.append(conv_biases_1)


        weight_key_1 = self.get_batchnorm_key()
        self.weights.append(weight_key_1 + "/batch_normalization/moving_mean:0")
        self.weights.append(weight_key_1 + "/batch_normalization/moving_variance:0")

        # Second convnet
        conv_weights_2 = weight_variable([3, 3, nof_filters, nof_filters])
        conv_biases_2 = bn_bias_variable([nof_filters])
        self.weights.append(conv_weights_2)
        self.weights.append(conv_biases_2)

        weight_key_2 = self.get_batchnorm_key()
        self.weights.append(weight_key_2 + "/batch_normalization/moving_mean:0")
        self.weights.append(weight_key_2 + "/batch_normalization/moving_variance:0")

        with tf.variable_scope(weight_key_1):
            h_bn1 = \
                tf.layers.batch_normalization(
                    conv2d(inputs, W_conv_1),
                    epsilon=1e-5, axis=1, fused=True,
                    center=False, scale=False,
                    training=self.training)
        h_out_1 = tf.nn.relu(h_bn1)
        with tf.variable_scope(weight_key_2):
            h_bn2 = \
                tf.layers.batch_normalization(
                    conv2d(h_out_1, W_conv_2),
                    epsilon=1e-5, axis=1, fused=True,
                    center=False, scale=False,
                    training=self.training)
        h_out_2 = tf.nn.relu(tf.add(h_bn2, orig))
        return h_out_2

        def weight_variable(shape):
            """
            The idea is to initialize weights such that data passed through the network does not change its magnitude
            dramatically
            :param shape:
            :return:
            """
            """Xavier initialization"""
            stddev = np.sqrt(2.0 / (sum(shape)))
            initial = tf.truncated_normal(shape, stddev=stddev)
            weights = tf.Variable(initial)
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weights)
            return weights

        # No point in learning bias weights as they are cancelled
        # out by the BatchNorm layers's mean adjustment.
        def bn_bias_variable(shape):
            initial = tf.constant(0.0, shape=shape)
            return tf.Variable(initial, trainable=False)

    def construct_net(self, data):
        input_boards = tf.reshape(planes, [None, self.NOF_PLANES, self.BOARD_SIZE_Y, self.BOARD_SIZE_X])

        # Input convolution
        flow = self.conv_block(input_boards, filter_size=3, input_channels=self.NOF_PLANES, output_channels=self.NOF_FILTERS)

        # Residual tower
        for _ in range(0, self.RESIDUAL_BLOCKS):
            flow = self.residual_block(flow, self.NOF_FILTERS)

        # Policy head
        conv_pol = self.conv_block(flow, filter_size=1,
                                   input_channels=self.RESIDUAL_FILTERS,
                                   output_channels=32)
        h_conv_pol_flat = tf.reshape(conv_pol, [-1, 32*8*8])
        W_fc1 = weight_variable([32*8*8, 1858])
        b_fc1 = bias_variable([1858])
        self.weights.append(W_fc1)
        self.weights.append(b_fc1)
        h_fc1 = tf.add(tf.matmul(h_conv_pol_flat, W_fc1), b_fc1, name='policy_head')

        # Value head
        conv_val = self.conv_block(flow, filter_size=1,
                                   input_channels=self.RESIDUAL_FILTERS,
                                   output_channels=32)
        h_conv_val_flat = tf.reshape(conv_val, [-1, 32*8*8])
        W_fc2 = weight_variable([32 * 8 * 8, 128])
        b_fc2 = bias_variable([128])
        self.weights.append(W_fc2)
        self.weights.append(b_fc2)
        h_fc2 = tf.nn.relu(tf.add(tf.matmul(h_conv_val_flat, W_fc2), b_fc2))
        W_fc3 = weight_variable([128, 1])
        b_fc3 = bias_variable([1])
        self.weights.append(W_fc3)
        self.weights.append(b_fc3)
        h_fc3 = tf.nn.tanh(tf.add(tf.matmul(h_fc2, W_fc3), b_fc3), name='value_head')

        return h_fc1, h_fc3


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()