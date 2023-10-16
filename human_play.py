# -*- coding: utf-8 -*-
"""
human VS AI models
Input your move in the format: 2,3

@author: Junxiao Song
"""

from __future__ import print_function
import pickle

import numpy as np

from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_numpy import PolicyValueNetNumpy
# from policy_value_net import PolicyValueNet  # Theano and Lasagne
from policy_value_net_pytorch import PolicyValueNet  # Pytorch
# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
# from policy_value_net_keras import PolicyValueNet  # Keras
import pickle
import torch
from collections import OrderedDict

import mcts_alphaZero


class Human(object):
    """
    human player
    """

    def __init__(self):
        self.player = None
        self.mcts_hidden = None

    def set_player_ind(self, p):
        self.player = p

    def set_hidden_player(self, board, model="best_policy_6_6_knobby_1008.model"):
        best_policy = PolicyValueNet(board.width, board.height, model_file=model)
        self.mcts_hidden = MCTSPlayer(best_policy.policy_value_fn,
                                      c_puct=5,
                                      n_playout=400)

    def get_hidden_probability(self, board, temp):
        move_probs = np.zeros(board.width * board.height)
        acts, probs = self.mcts_hidden.mcts.get_move_probs(board, temp)
        move_probs[list(acts)] = probs
        return move_probs


    def get_action(self, board, temp=0.75, return_prob=0):
        # temp (0, 1] needs to be larger for detailed probability map

        # --- get the action from a MCTS player for evaluating player move

        sensible_moves = board.availables
        self.set_hidden_player(board)
        move_probs = np.zeros(board.width * board.height)

        if len(sensible_moves) > 0:
            move_probs = self.get_hidden_probability(board, temp)
        else:
            move_probs = None

        # ---

        try:
            location = input("Your move: ")
            if isinstance(location, str):  # for python3
                location = [int(n, 10) for n in location.split(",")]
            move = board.location_to_move(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move = self.get_action(board)

        # allow human player get actions to return probabilities too
        if return_prob and move_probs is not None:
            return move, move_probs
        else:
            return move

    def __str__(self):
        return "Human {}".format(self.player)


def run():
    n = 4
    width, height = 6, 6
    model_file = 'best_policy_6_6_knobby_1011_mid.model'
    #model_file = 'best_policy_6_6_knobby_1008.model'
    try:
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)

        """
        param_theano = pickle.load(open(model_file, 'rb'))
        keys = ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias', 'conv3.weight', 'conv3.bias'
            , 'act_conv1.weight', 'act_conv1.bias', 'act_fc1.weight', 'act_fc1.bias'
            , 'val_conv1.weight', 'val_conv1.bias', 'val_fc1.weight', 'val_fc1.bias', 'val_fc2.weight', 'val_fc2.bias']
        param_pytorch = OrderedDict()
        for key, value in zip(keys, param_theano):
            if 'fc' in key and 'weight' in key:
                param_pytorch[key] = torch.FloatTensor(value.T)
            elif 'conv' in key and 'weight' in key:
                param_pytorch[key] = torch.FloatTensor(value[:, :, ::-1, ::-1].copy())
            else:
                param_pytorch[key] = torch.FloatTensor(value)
        """
        # ############### human VS AI ###################
        # load the trained policy_value_net in either Theano/Lasagne, PyTorch or TensorFlow

        # best_policy = PolicyValueNet(width, height, model_file = model_file)
        # mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)

        # load the provided model (trained in Theano/Lasagne) into a MCTS player written in pure numpy
        try:
            policy_param = model_file
            # policy_param = pickle.load(open(model_file, 'rb'))
        except:
            policy_param = pickle.load(open(model_file, 'rb'),
                                       encoding='bytes')  # To support python3
        best_policy = PolicyValueNet(width, height, policy_param)
        mcts_player = MCTSPlayer(best_policy.policy_value_fn,
                                 c_puct=5,
                                 n_playout=400)  # set larger n_playout for better performance

        # uncomment the following line to play with pure MCTS (it's much weaker even with a larger n_playout)
        # mcts_player = MCTS_Pure(c_puct=5, n_playout=1000)

        # human player, input your move in the format: 2,3
        human = Human()

        # set start_player=0 for human first
        game.start_play(human, mcts_player, start_player=1, is_shown=1)
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()
