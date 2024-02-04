import os
import numpy as np
import json
from collections import defaultdict
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial



deprecated_id = ['p03', 'p05', 'p06', 'p07', 'p08', 'p09', 'p10', 'p11']
params_list = []
group_blocked = []
group_interleaved = []
sum_blocked = [0] * 50
sum_interleaved = [0] * 50
win_rate_blocked = []
win_rate_interleaved = []

root = os.path.dirname(os.path.realpath(__file__))


# return
# 1) the prob number
# 2) the move position
def get_board_prob(id, round):
    global root

    i = str(round)
    file_probKnobby = id+"_probKnobby_"+i+".npy"
    file_probFour = id+"_probFouriar_"+i+".npy"
    path_k = os.path.join(root, f"{id}/{file_probKnobby}")
    path_f = os.path.join(root, f"{id}/{file_probFour}")
    probKnobby = np.load(path_k, allow_pickle=True)
    probFour = np.load(path_f, allow_pickle=True)

    # get the location and value of the probability that's not 0
    fmove_K = probKnobby[0]
    fmove_F = probFour[0]

    # Find the position (row, column) and value
    max_position_K = np.unravel_index(np.argmax(fmove_K), fmove_K.shape)
    max_value_K = fmove_K[max_position_K]
    max_position_F = np.unravel_index(np.argmax(fmove_F), fmove_F.shape)
    max_value_F = fmove_F[max_position_F]

    return max_value_K, max_position_K, max_value_F, max_position_F


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
    global group_blocked, group_interleaved

    for file_name, file_paths in params_list.items():
        for path in file_paths:
            with open(path, 'r') as file:
                params_data = json.load(file)

            condition = params_data.get('condition', 0)
            if condition == 0:
                group_blocked.append(path)
            else:
                group_interleaved.append(path)


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
    games_condition = params_data.get('condition', 0) # 0 = blcok, 1 = interleaved
    games_rule = params_data.get('games_rule', [])
    games_results = params_data.get('games_results', [])
    games_count = params_data.get('games_count', 0)

    # Calculate overall win rate
    # win_rate_overall = sum(result == 1 for result in games_results) / games_count

    # Judge which game is played first
    first_game = games_rule[0]

    # List of all win games
    # Initialize lists for results of each game type
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

def main():
    global params_list, group_blocked, group_interleaved, sum_blocked, sum_interleaved

    params_list = find_duplicate_params()
    group_by_condition(params_list)

    data_blocked = []
    data_interleaved = []

    # Calculate win rates for blocked condition
    for params_path in group_blocked:
        first_game, results_knobby, results_four = calculate_win(params_path)
        if (first_game == 0):
            data_blocked.append(results_four)
        elif (first_game == 1):
            data_blocked.append(results_knobby)

    # Calculate win rates for interleaved condition
    for params_path in group_interleaved:
        first_game, results_knobby, results_four = calculate_win(params_path)
        if (first_game == 0):
            data_interleaved.append(results_four)
        elif (first_game == 1):
            data_interleaved.append(results_knobby)

    # print("data blocked: ", data_blocked)
    # print("data interleaved: ", data_interleaved)

# BLOCKED
    count = [0] * 50
    n = len(data_blocked)

    for i in range(n):
        result = data_blocked[i]
        result = [0 if r != 1 else 1 for r in result] # keep only the 1s (human wins)
        for j in range(len(result)):
            sum_blocked[j] += result[j]
            count[j] += 1

    # Trim to remove trailing zeros
    count = count[:next((i for i, x in enumerate(reversed(count)) if x != 0), len(count))]
    count = [x for x in count if x == 7]
    sum_blocked = sum_blocked[:next((i for i, x in enumerate(reversed(sum_blocked)) if x != 0), len(sum_blocked))]

    print("sum_blocked: ", sum_blocked)
    print("count: ", count)

    for i in range(len(sum_blocked)):
        if i < len(count) and i < len(sum_blocked):
            w = round(sum_blocked[i] / count[i], 3)
            # print("w: ", w, " at ", i, "(", sum_blocked[i], "/", count[i], ")" )
            win_rate_blocked.append(w)

# INTERLEAVED
    count = [0] * 50
    n = len(data_interleaved)

    for i in range(n):
        result = data_interleaved[i]
        result = [0 if r != 1 else 1 for r in result]
        for j in range(len(result)):
            sum_interleaved[j] += result[j]
            count[j] += 1

    # Trim to remove trailing zeros
    count = count[:next((i for i, x in enumerate(reversed(count)) if x != 0), len(count))]
    count = [x for x in count if x == 7]
    sum_interleaved = sum_interleaved[:next((i for i, x in enumerate(reversed(sum_interleaved)) if x != 0), len(sum_interleaved))]
    print("sum_interleaved: ", sum_interleaved)
    print("count: ", count)

    for i in range(len(sum_interleaved)):
        if i < len(count) and i < len(sum_interleaved):
            w = round(sum_interleaved[i] / count[i], 3)
            win_rate_interleaved.append(w)

    plot_win_rate()


def calculate_win_rate_all(sum_data):
    win_rate = []
    count = [0] * 50
    n = len(sum_data)
    for i in range(n):
        result = sum_data[i]
        result = [0 if r != 1 else 1 for r in result]
        for j in range(len(result)):
            sum_data[j] += result[j]
            count[j] += 1
    count = count[:next((i for i, x in enumerate(reversed(count)) if x != 0), len(count))]
    count = [x for x in count if x == 7]
    sum_data = sum_data[:next((i for i, x in enumerate(reversed(sum_data)) if x != 0), len(sum_data))]
    for i in range(len(sum_data)):
        if i < len(count) and i < len(sum_data):
            w = round(sum_data[i] / count[i], 3)
            win_rate.append(w)

    return win_rate

def plot_win_rate():
    global win_rate_blocked, win_rate_interleaved

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True, sharey=True)

    # Create x-axis
    x1 = list(range(1, len(win_rate_blocked) + 1))
    x2 = list(range(1, len(win_rate_interleaved) + 1))

    # Fitting a 3rd-degree polynomial
    poly_1 = np.poly1d(np.polyfit(x1, win_rate_blocked, 3))
    smooth_1 = poly_1(x1)
    poly_2 = np.poly1d(np.polyfit(x2, win_rate_interleaved, 3))
    smooth_2 = poly_2(x2)

    ax1.plot(x1, win_rate_blocked, label="Blocked")
    ax1.plot(x1, smooth_1, label='Blocked*', color='red')

    ax2.plot(x2, win_rate_interleaved, label="Interleaved")
    ax2.plot(x2, smooth_2, label='Interleaved*', color='red')

    # Add title and labels
    ax1.set_title("Block-learning")
    ax1.set_xlabel("Game Count")
    ax1.set_ylabel("Win rate")

    ax2.set_title("Interleaved-learning")
    ax2.set_xlabel("Game Count")
    ax2.set_ylabel("Interleaved")

    # Add legend
    ax1.legend()
    ax2.legend()

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Show plot
    plt.show()

# given a id and a given game, retrieve the move prob for that game
# for t in range(1, 47):
#     a, b, c, d = get_board_prob("p27", t)
#     print(f"round {t}")

if __name__ == "__main__":
    main()