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

        self.input_states = tf.placeholder(
            tf.float32, shape=[None, self.NOF_PLANES, self.BOARD_SIZE_Y3, self.BOARD_SIZE_X]) # depth*width*height

        self.conv = tf.layers.conv2d(inputs=self.input_states_reshaped,
                                      filters=self.NOF_FILTERS, kernel_size=[3, 3],
                                      padding="same", activation=tf.nn.relu)

    def train(self):
        pass

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()