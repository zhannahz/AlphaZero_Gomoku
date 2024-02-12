import os
import numpy as np
import json
from collections import defaultdict
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial



deprecated_id = ['p03', 'p05', 'p06', 'p07', 'p08', 'p09', 'p10', 'p11']
params_list = []
paths_blocked = []
paths_interleaved = []
id_blocked = []
id_interleaved = []

win_rate_blocked_1 = []
win_rate_interleaved_1 = []
win_rate_blocked_2 = []
win_rate_interleaved_2 = []

root = os.path.dirname(os.path.realpath(__file__))


# return
# 1) the prob number
# 2) the move position

def find_duplicate_params():
    global root

    params_dict = defaultdict(list)

    for root, dirs, files in os.walk(root):
        for file_name in files:
            if file_name == 'params.json':
                params_path = os.path.join(root, file_name)
                params_dict[file_name].append(params_path)

    # Filter out items in the dictionary where the values are not unique
    duplicate_params = {k: v for k, v in params_dict.items() if len(v) > 1}

    return duplicate_params


# group data by condition (blocked=0 vs interleaved=1)
def group_by_condition(params_list):
    global paths_blocked, paths_interleaved

    for file_name, file_paths in params_list.items():
        for path in file_paths:
            with open(path, 'r') as file:
                params_data = json.load(file)

            condition = params_data.get('condition', 0)
            this_id = params_data.get('participant_id', 0)
            if condition == 0:
                paths_blocked.append(path)
                id_blocked.append(this_id)
            else:
                paths_interleaved.append(path)
                id_interleaved.append(this_id)
    print("blocked ids", id_blocked, "\ninterleaved ids", id_interleaved)

# return
# 1) the first game rule played
# 2) the results of the first game
# 3) the results of the second game
def calculate_win(params_path):
    with open(params_path, 'r') as file:
        params_data = json.load(file)

    if (params_data['participant_id'] in deprecated_id):
        return None, None, None

    # Extract relevant information
    id = params_data['participant_id']
    games_condition = params_data.get('condition', 0) # 0 = blcok, 1 = interleaved
    games_rule = params_data.get('games_rule', [])
    games_results = params_data.get('games_results', [])
    games_count = params_data.get('games_count', 0)

    # which game is played first
    first_game = games_rule[0]

    # List of all win games
    results_four = []
    results_knobby = []

    # Seperate results into two lists
    for result, rule in zip(games_results, games_rule):
        if rule == 0:  # Four-in-a-row
            results_four.append(result)
        elif rule == 1:  # Knobby
            results_knobby.append(result)

    if (first_game == 0):
        return first_game, results_four, results_knobby
    else:
        return first_game, results_knobby, results_four

# return the win rate for all games
# given the cumulative results
def get_win_rate_all(data):
    win_rate = []
    sum_win = [0] * 50
    count = [0] * 50
    n = len(data)

    for i in range(n):
        result = data[i]
        result = [0 if r != 1 else 1 for r in result]
        for j in range(len(result)):
            sum_win[j] += result[j]
            count[j] += 1

    count = count[:next((i for i, x in enumerate(reversed(count)) if x != 0), len(count))]
    count = [x for x in count if x == 7]
    sum_win = sum_win[:next((i for i, x in enumerate(reversed(sum_win)) if x != 0), len(sum_win))]

    for i in range(len(sum_win)):
        if i < len(count) and i < len(sum_win):
            w = round(sum_win[i] / count[i], 3)
            win_rate.append(w)

    return win_rate

# Plot 4 figures:
# win rates for blocked & interleaved conditions
# in the first & second games
def plot_win_rate():
    global win_rate_blocked_1, win_rate_interleaved_1, win_rate_blocked_2, win_rate_interleaved_2

    fig, ((ax1_1, ax1_2), (ax2_1, ax2_2)) = plt.subplots(2, 2, figsize=(16, 8), sharex=True, sharey=True)

    # Create x-axis
    x1_1 = list(range(1, len(win_rate_blocked_1) + 1))
    x2_1 = list(range(1, len(win_rate_interleaved_1) + 1))
    x1_2 = list(range(1, len(win_rate_blocked_2) + 1))
    x2_2 = list(range(1, len(win_rate_interleaved_2) + 1))
    # print("x1_1", x1_1, "x2_1", x2_1, "x1_2", x1_2, "x2_2", x2_2)

    # Fitting polynomial
    poly_1_1 = np.poly1d(np.polyfit(x1_1, win_rate_blocked_1, 3))
    smooth_b1 = poly_1_1(x1_1)
    poly_2_1 = np.poly1d(np.polyfit(x2_1, win_rate_interleaved_1, 3))
    smooth_i1 = poly_2_1(x2_1)
    poly_1_2 = np.poly1d(np.polyfit(x1_2, win_rate_blocked_2, 3))
    smooth_b2 = poly_1_2(x1_2)
    poly_2_2 = np.poly1d(np.polyfit(x2_2, win_rate_interleaved_2, 3))
    smooth_i2 = poly_2_2(x2_2)

    # Blocked condition
    ax1_1.plot(x1_1, win_rate_blocked_1, label="Win rate (raw)")
    ax1_1.plot(x1_1, smooth_b1, color='red')
    ax1_2.plot(x1_2, win_rate_blocked_2, label="Win rate (raw)")
    ax1_2.plot(x1_2, smooth_b2, color='red')

    # Interleaved condition
    ax2_1.plot(x2_1, win_rate_interleaved_1, label="Win rate (raw)")
    ax2_1.plot(x2_1, smooth_i1, color='red')
    ax2_2.plot(x2_2, win_rate_interleaved_2, label="Win rate (raw)")
    ax2_2.plot(x2_2, smooth_i2, color='red')

    # Add title and labels
    ax1_1.set_title("Block-learning (first game)")
    ax1_2.set_title("Block-learning (second game)")

    ax2_1.set_title("Interleaved-learning (first game)")
    ax2_1.set_xlabel("Game Count")
    ax2_1.set_ylabel("Win rate")
    ax2_2.set_title("Interleaved-learning (second game)")

    # Add legend
    ax1_1.legend()
    ax1_2.legend()
    ax2_1.legend()
    ax2_2.legend()

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Show plot
    plt.show()

def get_all_move_prob(id):
    global root

    all_moves_k = []
    all_moves_f = []

    for round in range(1, 50):
        i = str(round)
        file_probKnobby = id+"_probKnobby_"+i+".npy"
        file_probFour = id+"_probFouriar_"+i+".npy"
        file_move = id+"_boardDifference_"+i+".npy"
        path_k = os.path.join(root, f"{id}\\{file_probKnobby}")
        path_f = os.path.join(root, f"{id}\\{file_probFour}")
        path_move = os.path.join(root, f"{id}\\{file_move}")
        # check if the file exists
        if not os.path.exists(path_k) or not os.path.exists(path_f) or not os.path.exists(path_move):
            continue
        else:
            probKnobby = np.load(path_k, allow_pickle=True)
            probFour = np.load(path_f, allow_pickle=True)
            move = np.load(path_move, allow_pickle=True)

        for m in range(len(move)):
            step = move[m][1] - move[m][0]


        # keep only the non-zero matrices
        probKnobby = np.array([x for x in probKnobby if np.count_nonzero(x) > 0])
        probFour = np.array([x for x in probFour if np.count_nonzero(x) > 0])
        # print("probKnobby", probKnobby.shape, "probFour", probFour.shape)

        for k in probKnobby:
            all_moves_k.append(k)
        for f in probFour:
            all_moves_f.append(f)

        # Find the position (row, column) and value
        # for move in all_moves_k:
        #     max_position_K = np.unravel_index(np.argmax(move), move.shape)
        #     max_value_K = move[max_position_K]
        # for move in all_moves_f:
        #     max_position_F = np.unravel_index(np.argmax(move), move.shape)
        #     max_value_F = move[max_position_F]

    return all_moves_f, all_moves_k


def main():
    global params_list, \
        paths_blocked, \
        paths_interleaved, \
        sum_blocked, \
        sum_interleaved, \
        win_rate_blocked_1, \
        win_rate_interleaved_1, \
        win_rate_blocked_2, \
        win_rate_interleaved_2, \
        id_blocked, \
        id_interleaved

    # test
    # f, k = get_all_move_prob("p27")

    # Find params in each folder
    params_list = find_duplicate_params()
    
    # Group data by condition (blocked vs interleaved)
    group_by_condition(params_list)

    data_blocked_1 = []
    data_interleaved_1 = []
    data_blocked_2 = []
    data_interleaved_2 = []

    count_first_4iar = 0
    count_first_knobby = 0
    # Retrieve first & second game results for blocked condition
    for params_path in paths_blocked:
        first_game, results_knobby, results_four = calculate_win(params_path)
        if (first_game == 0):
            data_blocked_1.append(results_four)
            data_blocked_2.append(results_knobby)
            count_first_4iar += 1
        elif (first_game == 1):
            data_blocked_1.append(results_knobby)
            data_blocked_2.append(results_four)
            count_first_knobby += 1
    print("count_first_4iar", count_first_4iar, "count_first_knobby", count_first_knobby)

    # Retrieve first & second game results for interleaved condition
    for params_path in paths_interleaved:
        first_game, results_knobby, results_four = calculate_win(params_path)
        if (first_game == 0):
            data_interleaved_1.append(results_four)
            data_interleaved_2.append(results_knobby)
        elif (first_game == 1):
            data_interleaved_1.append(results_knobby)
            data_interleaved_2.append(results_four)

    win_rate_blocked_1 = get_win_rate_all(data_blocked_1)
    win_rate_interleaved_1 = get_win_rate_all(data_interleaved_1)
    win_rate_blocked_2 = get_win_rate_all(data_blocked_2)
    win_rate_interleaved_2 = get_win_rate_all(data_interleaved_2)

    plot_win_rate()


    # Retrieve the probability of all 100 moves for each condition
    # for id in id_blocked:
    #     for i in range(1, 47):
    #         a, b, c, d = get_all_move_prob(id, i)
    #         print(f"round {i}")



# given a id and a given game, retrieve the move prob for that game
# for t in range(1, 47):
#     a, b, c, d = get_all_move_prob("p27", t)
#     print(f"round {t}")

if __name__ == "__main__":
    main()