# -*- coding: utf-8 -*-
"""
@author github/suragnair, ozzi7
"""

import os
import shutil
import time
import random
import numpy as np
import math
import sys
import tensorflow as tf


class TicTacToeNet():
    def __init__(self, args):
        # game params
        self.board_x, self.board_y = 3,3
        self.action_size = 9
        self.args = args

        # Renaming functions
        Relu = tf.nn.relu
        Tanh = tf.nn.tanh
        BatchNormalization = tf.layers.batch_normalization
        Dropout = tf.layers.dropout
        Dense = tf.layers.dense

        # Neural Net
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.input_boards = tf.placeholder(tf.float32, shape=[None, self.board_x, self.board_y])    # s: batch_size x board_x x board_y
            self.dropout = tf.placeholder(tf.float32)
            self.isTraining = tf.placeholder(tf.bool, name="is_training")

            x_image = tf.reshape(self.input_boards, [-1, self.board_x, self.board_y, 1])                    # batch_size  x board_x x board_y x 1
            h_conv1 = Relu(BatchNormalization(self.conv2d(x_image, args.num_channels, 'same'), axis=3, training=self.isTraining))     # batch_size  x board_x x board_y x num_channels
            h_conv2 = Relu(BatchNormalization(self.conv2d(h_conv1, args.num_channels, 'same'), axis=3, training=self.isTraining))     # batch_size  x board_x x board_y x num_channels
            h_conv3 = Relu(BatchNormalization(self.conv2d(h_conv2, args.num_channels, 'valid'), axis=3, training=self.isTraining))    # batch_size  x (board_x-2) x (board_y-2) x num_channels
            h_conv4 = Relu(BatchNormalization(self.conv2d(h_conv3, args.num_channels, 'valid'), axis=3, training=self.isTraining))    # batch_size  x (board_x-4) x (board_y-4) x num_channels
            h_conv4_flat = tf.reshape(h_conv4, [-1, args.num_channels*(self.board_x-4)*(self.board_y-4)])
            s_fc1 = Dropout(Relu(BatchNormalization(Dense(h_conv4_flat, 1024), axis=1, training=self.isTraining)), rate=self.dropout) # batch_size x 1024
            s_fc2 = Dropout(Relu(BatchNormalization(Dense(s_fc1, 512), axis=1, training=self.isTraining)), rate=self.dropout)         # batch_size x 512
            self.pi = Dense(s_fc2, self.action_size)                                                        # batch_size x self.action_size
            self.prob = tf.nn.softmax(self.pi)
            self.v = Tanh(Dense(s_fc2, 1))                                                               # batch_size x 1

            self.calculate_loss()

    def conv2d(self, x, out_channels, padding):
      return tf.layers.conv2d(x, out_channels, kernel_size=[3,3], padding=padding)

    def calculate_loss(self):
        self.target_pis = tf.placeholder(tf.float32, shape=[None, self.action_size])
        self.target_vs = tf.placeholder(tf.float32, shape=[None])
        self.loss_pi =  tf.losses.softmax_cross_entropy(self.target_pis, self.pi)
        self.loss_v = tf.losses.mean_squared_error(self.target_vs, tf.reshape(self.v, shape=[-1,]))
        self.total_loss = self.loss_pi + self.loss_v
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.train.AdamOptimizer(self.args.lr).minimize(self.total_loss)

####

class Trainer:
    def __init__(self):
        self.NOF_FILTERS = 64
        self.NOF_RESIDUAL_BLOCKS = 6
        self.NOF_PLANES = 2
        self.BOARD_SIZE_X = 3
        self.BOARD_SIZE_Y = 3
        self.NOF_POLICY_PLANES = 32
        self.NOF_VALUE_PLANES = 32

        self.NUM_STEP_TRAIN = 200
        self.NUM_STEP_TEST = 2000
        self.lr_values = [0.02, 0.002, 0.0005]
        self.lr_boundaries = [100000,130000]
        self.total_steps = 140000

        self.weights = [] # to export

        # TF variables
        self.handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(
            self.handle, dataset.output_types, dataset.output_shapes)
        self.next_batch = iterator.get_next()
        self.train_handle = self.session.run(train_iterator.string_handle())
        self.test_handle = self.session.run(test_iterator.string_handle())

        self.init_net(self.next_batch)
        self.construct_net(self.NOF_PLANES)

    def init_net(self, next_batch):
        self.x = next_batch[0]  # tf.placeholder(tf.float32, [None, 112, 8*8])
        self.y_ = next_batch[1]  # tf.placeholder(tf.float32, [None, 1858])
        self.z_ = next_batch[2]  # tf.placeholder(tf.float32, [None, 1])
        self.batch_norm_count = 0
        self.y_conv, self.z_conv = self.construct_net(self.x)

        # Calculate loss on policy head
        cross_entropy = \
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y_,
                                                    logits=self.y_conv)
        self.policy_loss = tf.reduce_mean(cross_entropy)

        # Loss on value head
        self.mse_loss = \
            tf.reduce_mean(tf.squared_difference(self.z_, self.z_conv))

        # Regularizer
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.reg_term = \
            tf.contrib.layers.apply_regularization(regularizer, reg_variables)

        # For training from a (smaller) dataset of strong players, you will
        # want to reduce the factor in front of self.mse_loss here.
        pol_loss_w = self.cfg['training']['policy_loss_weight']
        val_loss_w = self.cfg['training']['value_loss_weight']
        loss = pol_loss_w * self.policy_loss + val_loss_w * self.mse_loss + self.reg_term

        # Set adaptive learning rate during training
        self.cfg['training']['lr_boundaries'].sort()
        self.cfg['training']['lr_values'].sort(reverse=True)
        self.lr = self.cfg['training']['lr_values'][0]

        # You need to change the learning rate here if you are training
        # from a self-play training set, for example start with 0.005 instead.
        opt_op = tf.train.MomentumOptimizer(
            learning_rate=self.learning_rate, momentum=0.9, use_nesterov=True)

        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.train_op = \
                opt_op.minimize(loss, global_step=self.global_step)

        correct_prediction = \
            tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        self.accuracy = tf.reduce_mean(correct_prediction)

        self.avg_policy_loss = []
        self.avg_mse_loss = []
        self.avg_reg_term = []
        self.time_start = None

        # Summary part
        self.test_writer = tf.summary.FileWriter(
            os.path.join(os.getcwd(), "leelalogs/{}-test".format(self.cfg['name'])), self.session.graph)
        self.train_writer = tf.summary.FileWriter(
            os.path.join(os.getcwd(), "leelalogs/{}-train".format(self.cfg['name'])), self.session.graph)

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

        self.session.run(self.init)

    def train(self, batch_size, test_batches):
        if not self.time_start:
            self.time_start = time.time()

        # Run training for this batch
        policy_loss, mse_loss, reg_term, _, _ = self.session.run(
            [self.policy_loss, self.mse_loss, self.reg_term, self.train_op,
             self.next_batch],
            feed_dict={self.training: True, self.learning_rate: self.lr, self.handle: self.train_handle})

        steps = tf.train.global_step(self.session, self.global_step)

        # Determine learning rate
        steps_total = (steps - 1) % self.total_steps
        self.lr = self.lr_values[bisect.bisect_right(self.lr_boundaries, steps_total)]

        # Keep running averages
        # Google's paper scales MSE by 1/4 to a [0, 1] range, so do the same to
        # get comparable values.
        mse_loss /= 4.0
        self.avg_policy_loss.append(policy_loss)
        self.avg_mse_loss.append(mse_loss)
        self.avg_reg_term.append(reg_term)
        if steps % self.NUM_STEP_TRAIN == 0:
            pol_loss_w = self.cfg['training']['policy_loss_weight']
            val_loss_w = self.cfg['training']['value_loss_weight']
            time_end = time.time()
            speed = 0
            if self.time_start:
                elapsed = time_end - self.time_start
                speed = batch_size * (self.NUM_STEP_TRAIN / elapsed)
            avg_policy_loss = np.mean(self.avg_policy_loss or [0])
            avg_mse_loss = np.mean(self.avg_mse_loss or [0])
            avg_reg_term = np.mean(self.avg_reg_term or [0])
            print("step {}, lr={:g} policy={:g} mse={:g} reg={:g} total={:g} ({:g} pos/s)".format(
                steps, self.lr, avg_policy_loss, avg_mse_loss, avg_reg_term,
                # Scale mse_loss back to the original to reflect the actual
                # value being optimized.
                # If you changed the factor in the loss formula above, you need
                # to change it here as well for correct outputs.
                pol_loss_w * avg_policy_loss + val_loss_w * avg_mse_loss + avg_reg_term,
                speed))
            train_summaries = tf.Summary(value=[
                tf.Summary.Value(tag="Policy Loss", simple_value=avg_policy_loss),
                tf.Summary.Value(tag="MSE Loss", simple_value=avg_mse_loss)])
            self.train_writer.add_summary(train_summaries, steps)
            self.time_start = time_end
            self.avg_policy_loss, self.avg_mse_loss, self.avg_reg_term = [], [], []

        if steps % self.NUM_STEP_TEST == 0:
            sum_accuracy = 0
            sum_mse = 0
            sum_policy = 0
            for _ in range(0, test_batches):
                test_policy, test_accuracy, test_mse, _ = self.session.run(
                    [self.policy_loss, self.accuracy, self.mse_loss,
                     self.next_batch],
                    feed_dict={self.training: False,
                               self.handle: self.test_handle})
                sum_accuracy += test_accuracy
                sum_mse += test_mse
                sum_policy += test_policy
            sum_accuracy /= test_batches
            sum_accuracy *= 100
            sum_policy /= test_batches
            # Additionally rescale to [0, 1] so divide by 4
            sum_mse /= (4.0 * test_batches)
            test_summaries = tf.Summary(value=[
                tf.Summary.Value(tag="Accuracy", simple_value=sum_accuracy),
                tf.Summary.Value(tag="Policy Loss", simple_value=sum_policy),
                tf.Summary.Value(tag="MSE Loss", simple_value=sum_mse)])
            self.test_writer.add_summary(test_summaries, steps)
            print("step {}, policy={:g} training accuracy={:g}%, mse={:g}". \
                  format(steps, sum_policy, sum_accuracy, sum_mse))
            path = os.path.join(self.root_dir, self.cfg['name'])
            save_path = self.saver.save(self.session, path, global_step=steps)
            print("Model saved in file: {}".format(save_path))
            leela_path = path + "-" + str(steps) + ".txt"
            self.save_leelaz_weights(leela_path)
            print("Weights saved in file: {}".format(leela_path))

    def save_weights(self, filename):
        """
        TODO: implement
        :param filename:
        :return:
        """
        with open(filename, "w") as file:
            for weights in self.weights:
                # Newline unless last line (single bias)
                file.write("\n")
                work_weights = None
                # Keyed batchnorm weights
                if isinstance(weights, str):
                    work_weights = tf.get_default_graph().get_tensor_by_name(weights)
                elif weights.shape.ndims == 4:
                    # Convolution weights need a transpose
                    #
                    # TF (kYXInputOutput)
                    # [filter_height, filter_width, in_channels, out_channels]
                    #
                    # Leela/cuDNN/Caffe (kOutputInputYX)
                    # [output, input, filter_size, filter_size]
                    work_weights = tf.transpose(weights, [3, 2, 0, 1])
                elif weights.shape.ndims == 2:
                    # Fully connected layers are [in, out] in TF
                    #
                    # [out, in] in Leela
                    #
                    work_weights = tf.transpose(weights, [1, 0])
                else:
                    # Biases, batchnorm etc
                    work_weights = weights
                nparray = work_weights.eval(session=self.session)
                wt_str = [str(wt) for wt in np.ravel(nparray)]
                file.write(" ".join(wt_str))

    def get_batchnorm_key(self):
        result = "bn" + str(self.batch_norm_count)
        self.batch_norm_count += 1
        return result

    def conv_block(self, inputs, filter_size, input_channels, output_channels):
        W_conv = self.weight_variable([filter_size, filter_size, input_channels, output_channels])
        b_conv = self.bn_bias_variable([output_channels])

        self.weights.append(W_conv)
        self.weights.append(b_conv)
        # The weights are internal to the batchnorm layer, so apply
        # a unique scope that we can store, and use to look them back up
        # later on.
        weight_key = self.get_batchnorm_key()
        self.weights.append(weight_key + "/batch_normalization/moving_mean:0")
        self.weights.append(weight_key + "/batch_normalization/moving_variance:0")

        with tf.variable_scope(weight_key):
            h_bn = tf.layers.batch_normalization(
                    self.conv2d(inputs, W_conv),
                    epsilon=1e-5, axis=1, fused=True,
                    center=False, scale=False,
                    training=self.training)
        h_conv = tf.nn.relu(h_bn)
        return h_conv

    def residual_block(self, inputs, nof_filters):
        """
        :param inputs:
        :param nof_filters:
        :return:
        """
        # First convnet
        passthrough = tf.identity(inputs)
        conv_weights_1 = self.weight_variable([3, 3, nof_filters, nof_filters])
        conv_biases_1 = self.bn_bias_variable([nof_filters])
        self.weights.append(conv_weights_1)
        self.weights.append(conv_biases_1)

        weight_key_1 = self.get_batchnorm_key()
        self.weights.append(weight_key_1 + "/batch_normalization/moving_mean:0")
        self.weights.append(weight_key_1 + "/batch_normalization/moving_variance:0")

        # Second convnet
        conv_weights_2 = self.weight_variable([3, 3, nof_filters, nof_filters])
        conv_biases_2 = self.bn_bias_variable([nof_filters])
        self.weights.append(conv_weights_2)
        self.weights.append(conv_biases_2)

        weight_key_2 = self.get_batchnorm_key()
        self.weights.append(weight_key_2 + "/batch_normalization/moving_mean:0")
        self.weights.append(weight_key_2 + "/batch_normalization/moving_variance:0")

        with tf.variable_scope(weight_key_1):
            h_bn1 = tf.layers.batch_normalization(
                    self.conv2d(inputs, conv_weights_1),
                    epsilon=1e-5, axis=1, fused=True,
                    center=False, scale=False,
                    training=self.training)
        h_out_1 = tf.nn.relu(h_bn1)
        with tf.variable_scope(weight_key_2):
            h_bn2 = tf.layers.batch_normalization(
                    self.conv2d(h_out_1, conv_weights_2),
                    epsilon=1e-5, axis=1, fused=True,
                    center=False, scale=False,
                    training=self.training)
        h_res_out = tf.nn.relu(tf.add(h_bn2, passthrough))
        return h_res_out

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

    def bn_bias_variable(shape):
        """
        # No point in learning bias weights as they are cancelled
        # out by the BatchNorm layers's mean adjustment.
        :return:
        """
        initial = tf.constant(0.0, shape=shape)
        return tf.Variable(initial, trainable=False)

    def bias_variable(shape):
        """
        # Bias weights for layers not followed by BatchNorm
        # We do not regularlize biases, so they are not
        # added to the regularlizer collection
        :return:
        """
        initial = tf.constant(0.0, shape=shape)
        return tf.Variable(initial)

    def construct_net(self, data):
        # Shape the input data
        input_boards = tf.reshape(data, [-1, self.NOF_PLANES, self.BOARD_SIZE_Y, self.BOARD_SIZE_X])

        # Convolutional layer
        flow = self.conv_block(input_boards, filter_size=3, input_channels=self.NOF_PLANES, output_channels=self.NOF_FILTERS)

        # Residual tower
        for _ in range(0, self.RESIDUAL_BLOCKS):
            flow = self.residual_block(flow, self.NOF_FILTERS)

        # Policy head
        conv_pol = self.conv_block(flow, filter_size=1, input_channels=self.RESIDUAL_FILTERS, output_channels=self.NOF_POLICY_PLANES)

        h_conv_pol_flat = tf.reshape(conv_pol, [-1, self.NOF_POLICY_PLANES*self.BOARD_SIZE_Y*self.BOARD_SIZE_X])
        W_fully_connected1 = self.weight_variable([self.NOF_POLICY_PLANES*self.BOARD_SIZE_Y*self.BOARD_SIZE_X, self.NOF_OUTPUT_POLICIES])
        b_fully_connected1 = self.bias_variable([self.NOF_OUTPUT_POLICIES])
        self.weights.append(W_fully_connected1)
        self.weights.append(b_fully_connected1)
        h_policy = tf.add(tf.matmul(h_conv_pol_flat, W_fully_connected1), b_fully_connected1, name='policy_head')

        # Value head
        conv_val = self.conv_block(flow, filter_size=1, input_channels=self.RESIDUAL_FILTERS, output_channels=self.NOF_VALUE_PLANES)
        h_conv_val_flat = tf.reshape(conv_val, [-1, self.NOF_POLICY_PLANES*self.BOARD_SIZE_Y*self.BOARD_SIZE_X])
        convolution_weights_value1 = self.weight_variable([self.NOF_VALUE_PLANES * self.BOARD_SIZE_Y * self.BOARD_SIZE_X , 128])
        value_biases = self.bias_variable([128])
        self.weights.append(convolution_weights_value1)
        self.weights.append(value_biases)
        h_fc2 = tf.nn.relu(tf.add(tf.matmul(h_conv_val_flat, convolution_weights_value1), value_biases))
        convolution_weights_value2 = self.weight_variable([128, 1])
        value_bias_last = self.bias_variable([1])
        self.weights.append(convolution_weights_value2)
        self.weights.append(value_bias_last)
        h_value = tf.nn.tanh(tf.add(tf.matmul(h_fc2, convolution_weights_value2), value_bias_last), name='value_head')

        return h_policy, h_value


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()