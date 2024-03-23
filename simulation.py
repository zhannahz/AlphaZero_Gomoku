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

def state_to_prob(matrix):
    # board states stored as a dict,
    # key: move as location on the board,
    # value: player as pieces type
    state = defaultdict()
    for i in range(6):
        for j in range(6):
            if matrix[1][i, j] == 2:
                m = board.location_to_move([i, j])
                state[m] = 2 # player 2 is the AI player
            elif matrix[1][i, j] == 1:
                m = board.location_to_move([i, j])
                state[m] = 1 # player 1 is the human player
    last_m = matrix[1] - matrix[0]
    last_m = np.where(last_m == 1)

    board.last_move = board.location_to_move([last_m[0][0], last_m[1][0]])

    return get_board_prob_maps(state)

def get_board_prob_maps(state):
    board.states = state
    board.current_player = 1
    print("state", state)
    availables = [i for i in range(36) if i not in state.keys()]
    board.availables = availables
    # get the board probability map from boardDifference[0]
    player.set_hidden_player(board, 0)
    move_prob_fiar = player.get_hidden_probability(board, temp)
    player.set_hidden_player(board, 1)
    move_prob_knobby = player.get_hidden_probability(board, temp)

    print("move_prob_fiar", move_prob_fiar)
    print("move_prob_knobby", move_prob_knobby)

    # mask the board

    move_prob_fiar = move_prob_fiar * state[2]
    move_prob_knobby = move_prob_knobby * state[2]

    return move_prob_fiar, move_prob_knobby