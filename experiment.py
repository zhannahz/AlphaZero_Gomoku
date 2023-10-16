"""


@author: Hannah Zeng
"""

from __future__ import print_function
from enum import Enum
import os
import random
import numpy as np
import csv
import pandas as pd  # To manipulate and save data
from sympy.printing.numpy import const

class Condition(Enum):
    block = 'block'
    interchange = 'interchange'

four_model = "best_policy_6_6_4_1010_mid.model"
knob_model = "best_policy_6_6_knobby_1011_mid.model"

params = {
    "participant_id": "",
    "condition": Condition('block'),
}

def main():
    inputs = input("Enter parameters for setting up the experiment (space as deliminator): \n")
    inputs = inputs.split(" ")
    if len(inputs) != len(params):
        print("The experiment needs ", len(params), "parameters to set up. Please try again.")
        return

    for idx, param in enumerate(inputs, 0):
        participant_id = inputs[0]
        condition = Condition(inputs[1])
    print("Participant ID: ", participant_id)
    print("Condition: ", condition)

# helper function
def board_to_matrix(self):
    """Converts current board state to a matrix representation."""
    matrix = np.zeros((self.width, self.height), dtype=int)
    for move, player in self.states.items():
        h, w = self.move_to_location(move)
        matrix[h][w] = player
    return matrix

def next_filename(append_step, base="data", extension=".txt"):
    """Generate the next filename based on existing files."""
    append_id = str(params["participant_id"])
    count = 1
    abs_dir = os.path.dirname(os.path.abspath(__file__)) + "/Data/"
    dir = os.path.join(abs_dir, f"{append_id}_{base}_{append_step}{extension}")
    while os.path.exists(dir):
        count += 1
        return os.path.join(abs_dir, f"{append_id}_{base}_{append_step}-{count}{extension}")
    return os.path.join(dir)
def save_data(data_list, append_step):
    filename = next_filename(append_step)
    with open(filename, "w") as file:
        for item in data_list:
            file.write(str(item) + "\n")
    return filename  # Return the filename to inform the caller about the generated file

if __name__ == "__main__":
    main()