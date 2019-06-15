# -*- coding: utf-8 -*-

"""
NeuralNet for the game of TicTacToe.
Authors: Evgeny Tyurin, github.com/evg-tyurin, github @suragnair @ozzi7, possibly others

Based on the OthelloNNet by SourKream and Surag Nair.

conv2d weights
That's the tensorflow convention, the kernel shape is (kernel_height, kernel_width, input_channels, output_channels).
(output_channels, input_channels, kernel_height, kernel_width)
To reverse it, you can always just get the value and transpose it appropriately,
e.g. with filters = model.get_weights()[0][:, :, 0, :].transpose((2, 1, 0)).

"""

import argparse
from keras.models import *
from keras.layers import *
from keras.optimizers import *

# game params
BOARD_X=5
BOARD_Y=5
NOF_INPUT_PLANES=3
NOF_POLICIES=25
NOF_FILTERS=8
NOF_VALUE_FILTERS=1
NOF_POLICY_FILTERS=8
NOF_FC_NEURONS_VAL_LAYER=32
NOF_RES_LAYERS=4


class TicTacToeNet():
    def __init__(self):
        self.input_boards = Input(shape=(BOARD_X, BOARD_Y, NOF_INPUT_PLANES))  # s: batch_size x board_x x board_y
        x = Conv2D(filters=NOF_FILTERS,
                   kernel_size=(3,3),
                   padding='same', activation='linear', use_bias=False,
                   data_format="channels_last")(self.input_boards)

        x = BatchNormalization(axis=3)(x) # axis -1 is equal to 3 here, means last dimension
        x = LeakyReLU()(x)
        for _ in range(NOF_RES_LAYERS):
            x = self.residual_layer(x, NOF_FILTERS,
                                    (3,3))
        self.value_head = self.value_head(x)
        self.policy_head = self.policy_head(x)

        self.model = Model(inputs=[self.input_boards], outputs=[self.policy_head, self.value_head])
        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(0.001))
        self.model.summary()

    def conv_layer(self, x, filters, kernel_size):
        x = Conv2D(
            filters=filters
            , kernel_size=kernel_size
            , padding='same'
            , activation='linear'
            , use_bias=False
        )(x)

        x = BatchNormalization(axis=3)(x)
        x = LeakyReLU()(x)

        return (x)

    def residual_layer(self, input_block, filters, kernel_size):
        """
        The residual layer
        :param input_block: input of CNN
        :param filters: how many filters?
        :param kernel_size: the kernel of the CNN
        :return:
        """
        x = self.conv_layer(input_block, filters, kernel_size)

        x = Conv2D(
            filters=filters
            , kernel_size=kernel_size
            , padding='same'
            , activation='linear'
            , use_bias=False
        )(x)

        x = BatchNormalization(axis=3)(x)
        x = add([input_block, x])
        x = LeakyReLU()(x)

        return (x)

    def value_head(self, x):
        """
        The value head that will be optimized with the reward as the target
        Using tanh as the activation function.
        :param x: the input from the residual layer
        :return:
        """
        x = Conv2D(
            filters=NOF_VALUE_FILTERS
            , kernel_size=(1,1)
            , padding='same'
            , activation='linear'
            , use_bias=False
        )(x)

        x = BatchNormalization(axis=3)(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        x = Dense(
            NOF_FC_NEURONS_VAL_LAYER
            , activation='linear'
        )(x)

        x = LeakyReLU()(x)
        x = Dense(
            1
            , activation='tanh'
            , name='value_head'
        )(x)

        return (x)

    def policy_head(self, x):
        """
            The policy head that will be optimized with the action prob as the target.
            Using softmax as the activation function.
            :param x: the input from the residual layer
            :return:
        """
        x = Conv2D(
            filters = NOF_POLICY_FILTERS
            , kernel_size = (1,1)
            , padding = 'same'
            , activation='linear', use_bias=False)(x)

        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        x = Dense( # this is equivalent to dense layer + softmax layer, but combined
            NOF_POLICIES
            , activation='softmax'
            , name ='policy_head'
            )(x)

        return (x)

    def dump_weights(self):
        f = open("weights.txt", "a")
        for layer in self.model.layers:
            weights = layer.get_weights()  # list of numpy arrays
            if weights: # if no weights
                for w in weights: # sometimes there is bias as well
                    print(w.shape)
                    #w = np.transpose(w, (3,0,1,2))
                    try:
                        w2 = np.transpose(w, (3, 2, 0, 1)) # (3,..) = move that was last to first place
                        np.savetxt(f, w2.flatten(order='C'), delimiter=',', newline=" ")
                        f.write("\n")

                    except:
                        np.savetxt(f, w.flatten(order='C'), delimiter=',', newline=" ")
                        f.write("\n")
        f.close()

    def calculate_loss(self):
        self.target_pis = tf.placeholder(tf.float32, shape=[None, NOF_POLICIES])
        self.target_vs = tf.placeholder(tf.float32, shape=[None])
        self.loss_pi =  tf.losses.softmax_cross_entropy(self.target_pis, self.pi)
        self.loss_v = tf.losses.mean_squared_error(self.target_vs, tf.reshape(self.v, shape=[-1,]))
        self.total_loss = self.loss_pi + self.loss_v
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.train.AdamOptimizer(self.args.lr).minimize(self.total_loss)