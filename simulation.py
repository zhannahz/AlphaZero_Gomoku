# Loop through the board difference files of each participant
# and get the board probability maps for each move
#
from collections import defaultdict
from mcts_alphaZero import MCTSPlayer
import os
import numpy as np
from game import Board, Game
from human_play import Human

player = Human()
board = Board(width=6, height=6, n_in_row=4)
game = Game(board)


temp = 0.75
move_probs_fiar = np.zeros(6 * 6)
move_probs_knobby = np.zeros(6 * 6)
root = os.path.dirname(os.path.realpath(__file__))

def convert_to_state(matrix):
    last_move = matrix[1] - matrix[0]
    state = np.zeros((4, 6, 6))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[1][i, j] == 2:
                state[0, i, j] = 1.0  # Represents the moves of the AI player
            elif matrix[1][i, j] == 1:
                state[1, i, j] = 1.0  # Represents the moves of the human player
    state[2] = last_move

    return get_board_prob_maps(state)

def get_board_prob_maps(state):
    board.states = state
    # get the board probability map from boardDifference[0]
    player.set_hidden_player(board, 0)
    move_prob_fiar = player.get_hidden_probability(board, temp)
    player.set_hidden_player(board, 1)
    move_prob_knobby = player.get_hidden_probability(board, temp)
    #
    # print("move_prob_fiar", move_prob_fiar)
    # print("move_prob_knobby", move_prob_knobby)

    # mask the board
    move_prob_fiar = move_prob_fiar * state[2]
    move_prob_knobby = move_prob_knobby * state[2]

    return move_prob_fiar, move_prob_knobby