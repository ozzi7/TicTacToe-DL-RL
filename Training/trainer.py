# -*- coding: utf-8 -*-
"""
@author ozzi7
"""

import os

from tictactoe_nn import *


class Trainer():
    def __init__(self):
        self.nnet = TicTacToeNet()
        self.EPOCHS = 1
        self.BATCH_SIZE = 100
        self.EPOCHS_FIT = 30

        print(K.image_data_format())  # print current format

    def train(self, inputs, output_values, output_policies):

        for eps in range(self.EPOCHS):
            print("Episode %d" % (eps))
            print("====================================================================================")

            #self.nnet.model.load_weights("best_model.hd5f")
            self.nnet.model.fit([inputs], [output_policies, output_values],
                                     batch_size=self.BATCH_SIZE,
                                     epochs=self.EPOCHS_FIT,
                                     verbose=1)
            self.nnet.model.save("best_model.hd5f")

        self.nnet.dump_weights()

    def predict(self,  input):
        self.nnet.model.load_weights("best_model.hd5f")

        #self.nnet.dump_weights()

        print("Prediction for: ")
        print(input)

        inp = self.nnet.model.input  # input placeholder
        outputs = [layer.output for layer in  self.nnet.model.layers][1:]  # all layer outputs except first (input) layer
        functor = K.function([inp], outputs)  # evaluation function

        # Testing
        print(np.transpose(input[0], (2, 0, 1)).flatten(order='C'))  # correct
        layer_outs = functor([input,1.0])
        for layer in layer_outs:
            try:
                print(np.transpose(layer, (0, 3,1,2)).flatten(order='C'))
            except:
                print(layer)

        prediction = self.nnet.model.predict([input])
        np.set_printoptions(threshold=np.inf)
        np.set_printoptions(suppress=True)
        print("Output: ")
        print(prediction[0])
        print(prediction[1])

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        if self.saver == None:
            self.saver = tf.train.Saver(self.nnet.graph.get_collection('variables'))
        with self.nnet.graph.as_default():
            self.saver.save(self.sess, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath+'.meta'):
            raise("No model in path {}".format(filepath))
        with self.nnet.graph.as_default():
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, filepath)