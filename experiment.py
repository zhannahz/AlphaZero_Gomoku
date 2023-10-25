"""


@author: Hannah Zeng
"""
# -*- coding: utf-8 -*-

from __future__ import print_function

import fnmatch
import shutil
import subprocess

from enum import Enum
import os
import random
import numpy as np
import json
import shutil
import glob

n_inputs = 3
#because i have non-english characters in my path, i have to use the raw string
path = r"C:\Users\汉那\gitHubTracking\AlphaZero_Gomoku"


def main():
    #print("\033[31mThis is red text!")
    #print("\033[32mThis is green text!")
    #print("\033[0mThis is reset to default color!")

    global params
    params = {
        "participant_id": "",
        "condition": 0,  # 0: block, 1: interchange
        "model": 0,  # 0: four-in-a-row, 1: knobby
        "state": 0,  # 0: init, 1: in_trial, 2: idle, 3: end
        "trials_fouriar": 2,
        "trials_knobby": 2,
        "fouriar_complete": False,
        "knobby_complete": False,
        "games_count": 0,
    }
    inputs = input("Enter parameters for setting up the experiment (space as deliminator): \n")
    inputs = inputs.split(" ")
    if len(inputs) != n_inputs:
        print("The experiment needs ", n_inputs, "parameters to set up. Please try again.")
        return

    for idx, param in enumerate(inputs, 0):
        params["participant_id"] = inputs[0]
        params["condition"] = int(inputs[1])
        params["model"] = int(inputs[2])

    print("Participant ID: ", params["participant_id"])

    if params["condition"] == 0:
        print("Condition is learning by block")
    else:
        print("Condition is learning by interchange")

    if params["model"] == 0:
        print("Rule to start with is four-in-a-row")
    else:
        print("Rule to start with is knobby")
    store_params_to_file()

    update_with_condition(True)


def update_with_condition(is_first=False):
    global params
    params = load_params_from_file()

    # block condition
    if params["condition"] == 0:
        if params["model"] == 0:
            if params["fouriar_complete"]:
                params["model"] = 1
        elif params["model"] == 1:
            if params["knobby_complete"]:
                params["model"] = 0

    # interchange condition
    elif params["condition"] == 1:
        if is_first:
            pass
        else:
            if params["model"] == 1:
                params["model"] = 0
            elif params["model"] == 0:
                params["model"] = 1

    store_params_to_file()

    start_a_game()


def start_a_game():
    global params
    params = load_params_from_file()

    # init
    if params["state"] == 0:
        print("Current state: init")
        count_trials()
        params["state"] = 1
        params["games_count"] += 1
        store_params_to_file()

        call_human_play()

    # in_trial
    elif params["state"] == 1:
        print("Current state: in_trial")
        params["state"] = 2
        store_params_to_file()
        start_a_game()

    # idle (between-trials)
    elif params["state"] == 2:
        print("Current state: idle")
        if params["trials_fouriar"] == 0 and params["trials_knobby"] == 0:
            print("All trials completed")
            params["state"] = 3
            store_params_to_file()
            start_a_game()

        else:
            print("- - - Starting another game - - -")
            count_trials()
            params["state"] = 1
            params["games_count"] += 1
            store_params_to_file()

            call_human_play()

    elif params["state"] == 3:
        print("Current state: end")
        store_params_to_file()
        end_experiment()

        return

def end_experiment():
    params = load_params_from_file()
    move_files_with_id(params["participant_id"])
    print("\033[31mExperiment is complete. Thank you for participating!")

def count_trials():
    global params
    params = load_params_from_file()

    if (params["model"] == 0):
        params["trials_fouriar"] -= 1
    elif (params["model"] == 1):
        params["trials_knobby"] -= 1

    if (params["trials_fouriar"] == 0):
        params["fouriar_complete"] = True
    if (params["trials_knobby"] == 0):
        params["knobby_complete"] = True

    print("Current trials left:", "4iar =", params["trials_fouriar"], "knobby =", params["trials_knobby"])


def call_human_play():
    if params["model"] == 0:
        print("\033[31mCurrent game rule is four in a row")
    else:
        print("\033[31mCurrent game rule is knobby")
    subprocess.call(['python', 'human_play.py'])


# helper function
def load_params_from_file(filename="params.json"):
    with open(filename, 'r') as file:
        return json.load(file)

def store_params_to_file(filename="params.json"):
    with open(filename, 'w') as file:
        json.dump(params, file)

def move_files_with_id(participant_id):
    # Specify the pattern for files starting with the participant_id
    pattern = f"{participant_id}_*"

    destination_dir = f"Data/{participant_id}/"
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Move each file matching the pattern to the destination directory
    file_path = os.path.join(os.getcwd(), "Data")
    files = os.listdir(file_path)
    for filename in files:
        #print(filename)
        if filename.startswith(f"{participant_id}_"):
            #print(filename, "is moved")
            f = os.path.join(file_path, filename)
            shutil.move(f, destination_dir)

    # Also move the params.json file
    if os.path.exists("params.json"):
        shutil.move("params.json", destination_dir)

def board_to_matrix(self):
    """Converts current board state to a matrix representation."""
    matrix = np.zeros((self.width, self.height), dtype=int)
    for move, player in self.states.items():
        h, w = self.move_to_location(move)
        matrix[h][w] = player
    return matrix


def next_filename(append_step, base="data", extension=".txt"):
    """Generate the next filename based on existing files."""
    global params
    params = load_params_from_file()

    append_id = str(params["participant_id"])
    games_count = str(params["games_count"])
    count = 1
    abs_dir = os.path.dirname(os.path.abspath(__file__)) + "/Data/"
    dir = os.path.join(abs_dir, f"{append_id}_{base}_{games_count}_{append_step}{extension}")
    while os.path.exists(dir):
        count += 1
        return os.path.join(abs_dir, f"{append_id}_{base}_{games_count}_{append_step}-({count}){extension}")
    return os.path.join(dir)


def save_board_data(data_list, append_step):
    filename = next_filename(append_step)
    with open(filename, "w") as file:
        for item in data_list:
            file.write(str(item) + "\n")
    return filename  # Return the filename to inform the caller about the generated file


if __name__ == "__main__":
    main()
