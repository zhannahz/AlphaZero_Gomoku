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
policy_param = 'best_policy_6_6_k_0311.model'

def convert_to_state(matrix):
    last_move = matrix[1] - matrix[0]
    state = np.zeros((4, matrix.shape[0], matrix.shape[1]))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] == 2:
                state[0, i, j] = 1.0  # Represents the moves of the AI player
            elif matrix[i, j] == 1:
                state[1, i, j] = 1.0  # Represents the moves of the human player
    state[2] = last_move
    return state

def retain_board_state(move):
    # do move
    board.do_move(board, move)
    # get current board state
    state_matrices = board.current_state(board)
    player_moves = state_matrices[1]
    ai_moves = state_matrices[0]
    ai_moves[ai_moves == 1] = 2
    last_move = state_matrices[2]
    previous_matrix = player_moves + ai_moves - last_move
    current_matrix = player_moves + ai_moves
    mask = last_move != 0


    return board
def get_board_prob_maps(boardDifference):
    # get the board probability map from boardDifference[0]
    player.set_hidden_player(board, 0)
    move_prob_fiar = player.get_hidden_probability(board, temp)

    player.set_hidden_player(board, 1)
    move_prob_knobby = player.get_hidden_probability(board, temp)
    # mask the board with the boardDifference[1]
