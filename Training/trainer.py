# -*- coding: utf-8 -*-
"""
@author ozzi7
"""

import os

from tictactoe_nn import *


class Trainer():
    def __init__(self):
        self.nnet = TicTacToeNet()
        self.BATCH_SIZE = 100
        self.EPOCHS_FIT = 15

        #print(K.image_data_format())  # print current format

    def save_init_weights(self):

        print("Initializing network weights")
        self.nnet.model.save("best_model.hd5f")
        self.nnet.dump_weights()

    def train(self, inputs, output_values, output_policies):

        try:
            with open('./best_model.hd5f', 'r') as fh:
                pass
        except FileNotFoundError:
            self.save_init_weights()

        print("====================================================================================")

        self.nnet.model.load_weights("best_model.hd5f")
        self.nnet.model.fit(np.array(inputs), [np.array(output_policies), np.array(output_values)],
                                 batch_size=self.BATCH_SIZE,
                                 epochs=self.EPOCHS_FIT,
                                 verbose=1)
        self.nnet.model.save("best_model.hd5f")

        self.nnet.dump_weights()

    def predict(self,  input):
        self.nnet.model.load_weights("best_model.hd5f")

        #self.nnet.dump_weights()

        print("Prediction for: " + str(input))

        inp = self.nnet.model.input  # input placeholder
        outputs = [layer.output for layer in  self.nnet.model.layers][1:]  # all layer outputs except first (input) layer
        functor = K.function([inp], outputs)  # evaluation function

        #Testing
        print(np.transpose(input[0], (2, 0, 1)).flatten(order='C'))  # correct
        layer_outs = functor([input,1.0])
        for layer in layer_outs:
            try:
                print(np.transpose(layer, (0, 3, 1, 2)).flatten(order='C'))
            except:
                print(layer)

        prediction = self.nnet.model.predict([input])
        np.set_printoptions(threshold=np.inf)
        np.set_printoptions(suppress=True)
        print("Output: ")
        print(prediction[0])
        print(prediction[1])
