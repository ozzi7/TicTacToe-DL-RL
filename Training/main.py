# -*- coding: utf-8 -*-
"""
@author ozzi7

"""

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from collections import deque
from trainer import Trainer
import numpy as np
from ast import literal_eval as createTuple
import re
import sys
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"


def read_samples(filename):
    # read training data from file
    with open(filename) as f:
        games = f.readlines()
        inputs = []
        output_values = []
        output_policies = []

        line_count = 0
        for line in games:
            # read value
            if line_count % 3 == 0:
                output_value = float(line)

            # read moves
            elif line_count % 3 == 1:
                # empty board first
                input = np.zeros((5, 5, 3))
                input[:, :, 2].fill(1)
                inputs.append(input)
                # add rotated boards
                inputs.append(np.rot90(np.copy(input), axes=(1,0),k=1))
                inputs.append(np.rot90(np.copy(input), axes=(1, 0), k=2))
                inputs.append(np.rot90(np.copy(input), axes=(1, 0), k=3))

                pattern = '\((\d+, \d+)\)'
                data = re.findall(pattern, line)
                moves = []
                for item in data:
                    moves.append(tuple(map(lambda x: int(x), item.split(','))))

                # construct the input
                player = -1
                input = np.zeros((5, 5, 3))
                for i in range(len(moves)-1): # ignore last move, we dont have visit counts there so we dont train it
                    move = moves[i]

                    if player == 1:
                        input[move[1],move[0], 1] = 1 # x, y, channel
                        input[:, :,2].fill(1)

                    elif player == -1:
                        input[move[1], move[0],0] = 1
                        input[:, :,2].fill(0)

                    inputs.append(np.copy(input))
                    # add rotated boards
                    inputs.append(np.rot90(np.copy(input), axes=(1, 0), k=1))
                    inputs.append(np.rot90(np.copy(input), axes=(1, 0), k=2))
                    inputs.append(np.rot90(np.copy(input), axes=(1, 0), k=3))

                    player *= -1

            # read policy
            elif line_count % 3 == 2:
                line = line.replace("(","").replace(")","").replace(",", " ")
                policies = [float(number) for number in line.split()]

                for move in range(len(moves)):
                    policy = np.zeros((25))
                    for i in range(25):
                        policy[i] = policies[move*25+i]

                    output_values.append(np.array([output_value])) # output val is from the view of player X
                    output_values.append(np.array([output_value]))  # output val is from the view of player X
                    output_values.append(np.array([output_value]))  # output val is from the view of player X
                    output_values.append(np.array([output_value]))  # output val is from the view of player X

                    output_policies.append(policy)
                    output_policies.append((np.rot90(np.reshape(np.copy(policy), (5, 5)),k=1)).flatten())
                    output_policies.append((np.rot90(np.reshape(np.copy(policy), (5, 5)), k=2)).flatten())
                    output_policies.append((np.rot90(np.reshape(np.copy(policy), (5, 5)), k=3)).flatten())

            line_count += 1

    return (inputs,output_values, output_policies)

if __name__ == '__main__':

    os.chdir(os.path.dirname(sys.argv[0]))
    trainer = Trainer()
    #trainer.save_init_weights()
    trainer.train(*read_samples(r'Z:/CloudStation/GitHub Projects/TicTacToe-DL-RL/Training/' + sys.argv[1]))
    #(inputs, output_values, output_policies) = read_samples(r'Z:/CloudStation/GitHub Projects/TicTacToe-DL-RL/Training/' + sys.argv[1])
    #trainer.predict([inputs[0]])