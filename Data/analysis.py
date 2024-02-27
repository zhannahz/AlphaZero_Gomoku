import math
import os
import numpy as np
import json
from collections import defaultdict
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial



deprecated_id = ['p03', 'p05', 'p06', 'p07', 'p08', 'p09', 'p10', 'p11', 'p35', 'p16', 'p18']
#p35 - problematic params
#p16 - lose params
params_list = []
paths_blocked = []
paths_interleaved = []
id_blocked = []
id_interleaved = []

win_rate_blocked_1 = []
win_rate_interleaved_1 = []
win_rate_blocked_2 = []
win_rate_interleaved_2 = []

aggregated_y1 = []
aggregated_y2 = []

root = os.path.dirname(os.path.realpath(__file__))


# return
# 1) the prob number
# 2) the move position

def find_duplicate_params():
    global root

    params_dict = defaultdict(list)
    temp_root = root
    for temp_root, dirs, files in os.walk(temp_root):
        for file_name in files:
            if file_name == 'params.json':
                params_path = os.path.join(temp_root, file_name)
                # Extract participant ID from the file path
                participant_id = os.path.basename(os.path.dirname(params_path))
                if (participant_id not in deprecated_id):
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

    if games_rule == []: #
        print("No games_rule for", id)
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
        for j in range(len(result)): # j = game number
            sum_win[j] += result[j]
            count[j] += 1

    while count and count[-1] == 0:
        count.pop()
    max = count[0]
    count = [x for x in count if x == max] # make sure to look at the same number of games for each participant
    sum_win = sum_win[:len(count)] # slice to the same length as count

    for i in range(len(sum_win)):
        if count[i] != 0:
            w = round(sum_win[i] / count[i], 3)
            win_rate.append(w)

    return win_rate, max

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

    all_moves_dict = defaultdict(list)
    all_prob_dict = defaultdict(list)

    # iterate through all games played by the participant
    for round in range(1, 60): #60 is the max #games played so far
        # create a dictionary to store the move and prob for each round/game
        move_dict = defaultdict(list)
        prob_dict = defaultdict(list)

        i = str(round)
        file_probKnobby = id+"_probKnobby_"+i+".npy"
        file_probFour = id+"_probFouriar_"+i+".npy"
        file_move = id+"_boardDifference_"+i+".npy"

        path_k = os.path.join(root, f"{id}\\{file_probKnobby}")
        path_f = os.path.join(root, f"{id}\\{file_probFour}")
        path_move = os.path.join(root, f"{id}\\{file_move}")
        # print("root", root, "id", id)
        # print("path_k", path_k, "path_f", path_f, "path_move", path_move)

        # check if the file exists
        if not os.path.exists(path_k) or not os.path.exists(path_f) or not os.path.exists(path_move):
            continue
        else:
            probKnobby = np.load(path_k, allow_pickle=True)
            probFour = np.load(path_f, allow_pickle=True)
            move = np.load(path_move, allow_pickle=True)

        move = move[:20] # all participants finish within 20 moves, so remove redundant 0s
        n_steps = len(move)
        for s in range(n_steps):
            if np.all(move[s][0] == None) or np.all(move[s][1] == None):
                continue
            step = move[s][1] - move[s][0]
            move_dict[s] = step
            prob_dict[s] = probFour[s], probKnobby[s]

        # print("move_dict size:", len(move_dict), "for round", i)
        # print("prob_dict size:", len(prob_dict), "for round", i)

        n = len(all_moves_dict)
        for i in range(len(move_dict)):
            all_moves_dict[n+i] = move_dict[i]
            all_prob_dict[n+i] = prob_dict[i]

    # mask the probability matrices with the move matrices
    for m in range(0, len(all_moves_dict)):
        move = all_moves_dict[m]
        mask = move.astype(bool)
        all_prob_dict[m] = [all_prob_dict[m][0][mask], all_prob_dict[m][1][mask]]


    return all_prob_dict, all_moves_dict

def plot_move_prob_comparison(ax1, ax2, data_prob):
    x = list(range(1, len(data_prob) + 1))  # move number (about 100*2=200)

    y1_raw = []  # move prob for first rule
    for i in range(len(data_prob)):
        y1_raw.append(data_prob[i][0])
    y2_raw = []  # move prob for second rule
    for i in range(len(data_prob)):
        y2_raw.append(data_prob[i][1])

    y1_raw = [0 if v is None else v for v in y1_raw]
    y2_raw = [0 if v is None else v for v in y2_raw]

    # convert nested list to 1D list
    y1 = [item[0] for item in y1_raw]
    y2 = [item[0] for item in y2_raw]

    # fit polynomial
    poly_1 = np.poly1d(np.polyfit(x, y1, 3))
    smooth_y1 = poly_1(x)
    poly_2 = np.poly1d(np.polyfit(x, y2, 3))
    smooth_y2 = poly_2(x)

    # Plot the move prob for the first rule using dots
    ax1.plot(x, y1, 'o', color='black', alpha=0.3)  # 'o' for circular markers
    ax1.plot(x, smooth_y1, color='red', alpha=0.5)
    ax1.set_xlabel('Move Number')
    ax1.set_ylabel('Probability')
    ax1.set_title('Move Probability (First Rule)')

    # Plot the move prob for the second rule using dots
    ax2.plot(x, y2, 'o', color='black', alpha=0.3)  # 'o' for circular markers
    ax2.plot(x, smooth_y2, color='red', alpha=0.5)
    ax2.set_xlabel('Move Number')
    ax2.set_ylabel('Probability')
    ax2.set_title('Move Probability (Second Rule)')


def normalize_probability(prob):
    for i in range(len(prob)):
        max_prob = max(prob[i][0], prob[i][1])
        if (max_prob == 0):
            # print("max_prob is 0 at move", i, "prob[i]", prob[i])
            continue
        prob[i] = [prob[i][0] / max_prob, prob[i][1] / max_prob]
    return prob
def check_data_quality(all_data):
    # a dictionary to store the win rate for each participant
    win_rate_dict = {}

    for file_name, file_paths in all_data.items():
        for path in file_paths:
            with open(path, 'r') as file:
                data = json.load(file)
            if (data['participant_id'] not in deprecated_id):
                # calculate individual win rate
                results = data.get('games_results', [])
                win_result = [0 if r != 1 else 1 for r in results]
                game_total = len(results)
                win_rate = round(sum(win_result) / game_total, 3)

                # print("id", data['participant_id'], "win_rate", win_rate, "for game_total", game_total)

                win_rate_dict[data['participant_id']] = win_rate

    # plot win rate distribution
    n_bin = round(math.sqrt(len(win_rate_dict)))
    plt.hist(win_rate_dict.values(), bins=5, alpha=0.5, color='black', edgecolor='white')
    plt.xlabel('Win Rate (within individual)')
    plt.ylabel('Frequency')
    plt.yticks(range(0, 10, 1))
    plt.title('Win Rate Distribution (n={})'.format(len(win_rate_dict)))
    plt.show()

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

    # Find params in each folder
    params_list = find_duplicate_params()

    # test overall data quality
    check_data_quality(params_list)
    
    # Group data by condition (blocked vs interleaved)
    group_by_condition(params_list)

    # people's first game results and second game results
    data_blocked_1 = []
    data_interleaved_1 = []
    data_blocked_2 = []
    data_interleaved_2 = []

    # how many people played a specific game first in a condition
    count_blocked_first_4iar = 0
    count_blocked_first_knobby = 0
    count_mix_first_4iar = 0
    count_mix_first_knobby = 0

    # 1) compare first & second game results
    for params_path in paths_blocked:
        first_game, results_knobby, results_four = calculate_win(params_path)
        if (first_game == 0):
            data_blocked_1.append(results_four)
            data_blocked_2.append(results_knobby)
            count_blocked_first_4iar += 1
        elif (first_game == 1):
            data_blocked_1.append(results_knobby)
            data_blocked_2.append(results_four)
            count_blocked_first_knobby += 1
    print("block-learning", len(paths_blocked))
    print("4iar first:", count_blocked_first_4iar, "knobby first:", count_blocked_first_knobby)
    # Retrieve first & second game results for interleaved condition
    for params_path in paths_interleaved:
        first_game, results_knobby, results_four = calculate_win(params_path)
        if (first_game == 0):
            data_interleaved_1.append(results_four)
            data_interleaved_2.append(results_knobby)
            count_mix_first_4iar += 1
        elif (first_game == 1):
            data_interleaved_1.append(results_knobby)
            data_interleaved_2.append(results_four)
            count_mix_first_knobby += 1
    print("interleaved-learning", len(paths_interleaved))
    print("4iar first:", count_mix_first_4iar, "knobby first:", count_mix_first_knobby)

    win_rate_blocked_1, max_b_1 = get_win_rate_all(data_blocked_1)
    win_rate_interleaved_1, max_i_1 = get_win_rate_all(data_interleaved_1)
    win_rate_blocked_2, max_b_2 = get_win_rate_all(data_blocked_2)
    win_rate_interleaved_2, max_i_2 = get_win_rate_all(data_interleaved_2)

    print("block's 1st rule total # games - ", len(win_rate_blocked_1), "at least have # games", max_b_1)
    print("interleaved's 1st rule total # - ", len(win_rate_interleaved_1), "at least have # games", max_i_1)
    print("block's 2nd rule total # games - ", len(win_rate_blocked_2), "at least have # games", max_b_2)
    print("interleaved's 2nd rule total # - ", len(win_rate_interleaved_2), "at least have # games", max_i_2)

    plot_win_rate()


    # 2) compare the probabilities of all 100 moves for each rule

    fig_blocked, (ax_b_1, ax_b_2) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    fig_interleaved, (ax_i_1, ax_i_2) = plt.subplots(1, 2, figsize=(16, 8))
    for id in id_blocked:
        print("blocked - id", id)
        prob, move = get_all_move_prob(id)
        prob = normalize_probability(prob)
        plot_move_prob_comparison(ax_b_1, ax_b_2, prob)

    for id in id_interleaved:
        print("interleaved - id", id)
        prob, move = get_all_move_prob(id)
        prob = normalize_probability(prob)
        plot_move_prob_comparison(ax_i_1, ax_i_2, prob)

    fig_blocked.suptitle('Blocked Condition', fontsize=16)
    fig_interleaved.suptitle('Interleaved Condition', fontsize=16)
    fig_blocked.show()
    fig_interleaved.show()


if __name__ == "__main__":
    main()