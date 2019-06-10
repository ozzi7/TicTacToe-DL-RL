# -*- coding: utf-8 -*-
"""
@author ozzi7
"""

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from collections import deque
from train import Trainer
import numpy as np
from ast import literal_eval as createTuple
import re


def read_samples():
    # read training data from file
    with open("./training_games.txt") as f:
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
                pattern = '\((\d+, \d+)\)'
                data = re.findall(pattern, line)
                moves = []
                for item in data:
                    moves.append(tuple(map(lambda x: int(x), item.split(','))))

                # construct the input
                player = 1
                for move in moves:
                    input = np.zeros((5, 5, 3))  # input size
                    if player == 1:
                        input[move[0], move[1], 0] = 1
                        input[:, :, 2].fill(1)
                        output_values.append(np.array([output_value]))
                    elif player == -1:
                        input[move[0], move[1], 1] = 1
                        input[:, :, 2].fill(0)
                        output_values.append(np.array([1-output_value]))
                    inputs.append(input)
                    player *= -1

            # read policy
            elif line_count % 3 == 2:
                policies = []
                line = line.replace("(","").replace(")","").replace(",", " ")
                policies = [float(number) for number in line.split()]

                for move in range(len(moves)):
                    policy = np.zeros((25))
                    for i in range(25):
                        policy[i] = policies[move*25+i]
                    output_policies.append(policy)

            line_count += 1
    return (inputs,output_values, output_policies)

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train(*read_samples())
