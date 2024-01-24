import os
import numpy as np
import json
from collections import defaultdict
import matplotlib.pyplot as plt


deprecated_id = ['p03', 'p05', 'p06', 'p07', 'p08', 'p09', 'p10', 'p11']

def get_board_prob(id, round):
    # return
    # 1) the prob number
    # 2) the move position
    i = round
    file_probKnobby = id+"_probKnobby_"+i+".npy"
    file_probFour = id+"_probFouriar_"+i+".npy"
    path = f"Data/{id}/{file_probKnobby}"
    probKnobby = np.load(path)
    probFour = np.load(path)

    # get the location and value of the probability that's not 0
    fmove_K = probKnobby[0]
    fmove_F = probFour[0]

    # Find the position (row, column) and value
    max_position_K = np.unravel_index(np.argmax(fmove_K), fmove_K.shape)
    max_value_K = fmove_K[max_position_K]
    max_position_F = np.unravel_index(np.argmax(fmove_F), fmove_F.shape)
    max_value_F = fmove_F[max_position_F]

    return max_value_K, max_position_K, max_value_F, max_position_F

# given a id and a given game, retrieve the move prob for that game
a, b, c, d = get_board_prob("p27", 1)