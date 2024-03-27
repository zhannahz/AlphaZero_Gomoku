import math
import os
import numpy as np
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from experiment import save_game_data_simple
from simulation import state_to_prob

deprecated_id = ['p03', 'p05', 'p06', 'p07', 'p08', 'p09', 'p10', 'p11', 'p35', 'p16', 'p18']
# p35 - problematic params
# p16 - lose params
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
                    params_dict[participant_id].append(params_path)

    return params_dict


# group data by condition (blocked=0 vs interleaved=1)
def group_by_condition(params_list):
    global paths_blocked, paths_interleaved, id_blocked, id_interleaved

    for id, file_paths in params_list.items():
        #print("id", id, "file_paths", file_paths)
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


def get_frist_game_by_id(id):
    # find corresponding params file
    for file_name, file_paths in params_list.items():
        for path in file_paths:
            with open(path, 'r') as file:
                params_data = json.load(file)
            if (params_data['participant_id'] == id):
                return params_data.get('games_rule', [])[0]  # return the first game played


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
    games_condition = params_data.get('condition', 0)  # 0 = blcok, 1 = interleaved
    games_rule = params_data.get('games_rule', [])
    games_results = params_data.get('games_results', [])
    games_count = params_data.get('games_count', 0)

    if games_rule == []:  #
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
        for j in range(len(result)):  # j = game number
            sum_win[j] += result[j]
            count[j] += 1

    while count and count[-1] == 0:
        count.pop()
    max = count[0]
    count = [x for x in count if x == max]  # make sure to look at the same number of games for each participant
    sum_win = sum_win[:len(count)]  # slice to the same length as count

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
    print("id", id)
    all_moves_dict = defaultdict(list)
    all_prob_dict = defaultdict(list)
    all_prob_dict_new = defaultdict(list)
    all_games_round = []

    # iterate through all games played by the participant
    for round in range(1, 60):  # 60 is the max #games played so far
        # create a dictionary to store the move and prob for each round/game
        move_dict = defaultdict(list)
        prob_dict = defaultdict(list)
        prob_dict_new = defaultdict(list)

        i = str(round)
        file_probKnobby = id + "_fullProbKnobby_" + i + ".npy"
        file_probFour = id + "_fullProbFiar_" + i + ".npy"
        file_move = id + "_boardDifference_" + i + ".npy"
        new_file_probKnobby = id + "_new_fullProbKnobby_" + i + ".npy"
        new_file_probFour = id + "_new_fullProbFiar_" + i + ".npy"
        # new_file_probKnobby = id + "_new_fullProbKnobby_" + i
        # new_file_probFour = id + "_new_fullProbFiar_" + i

        path_k = os.path.join(root, f"{id}\\{file_probKnobby}")
        path_f = os.path.join(root, f"{id}\\{file_probFour}")
        path_move = os.path.join(root, f"{id}\\{file_move}")
        path_k_new = os.path.join(root, f"{id}\\{new_file_probKnobby}")
        path_f_new = os.path.join(root, f"{id}\\{new_file_probFour}")

        # check if the file exists
        if not os.path.exists(path_move):
            continue
        else:
            probKnobby = np.load(path_k, allow_pickle=True)
            probFour = np.load(path_f, allow_pickle=True)
            move = np.load(path_move, allow_pickle=True)
            probKnobby_new = np.load(path_k_new, allow_pickle=True)
            probFour_new = np.load(path_f_new, allow_pickle=True)

            move = move[:20]  # all participants finish within 20 moves, so remove redundant 0s
            n_steps = len(move)

            # -------------------------
            # full_fiar = []
            # full_knobby = []
            # check_path_k = path_k_new + ".npy"
            # check_path_f = path_f_new + ".npy"
            # -------------------------

            for s in range(n_steps):
                if np.all(move[s][0] == None) or np.all(move[s][1] == None): # skip if the move is None
                    continue
                step = move[s][1] - move[s][0]
                move_dict[s] = step

                probFour[s] = np.flip(probFour[s], 0) # flip row to match coordinate system
                probKnobby[s] = np.flip(probKnobby[s], 0) # flip row

                prob_dict[s] = probFour[s], probKnobby[s]
                prob_dict_new[s] = probFour_new[s], probKnobby_new[s]
                all_games_round.append(round)

                # NEW get new probability matrices if haven't
                # -------------------------
                # if not os.path.exists(check_path_k) or not os.path.exists(check_path_f):
                # f, k = state_to_prob(move[s]) # for each step
                # full_fiar.append(f) # for each game
                # full_knobby.append(k)
                # -------------------------

            n = len(all_moves_dict)
            for i in range(len(move_dict)):
                all_moves_dict[n + i] = move_dict[i]
                all_prob_dict[n + i] = prob_dict[i]
                all_prob_dict_new[n + i] = prob_dict_new[i]

            # NEW save if new prob files are not found
            # -------------------------
            # if not os.path.exists(check_path_k) or not os.path.exists(check_path_f):
            # save_game_data_simple(full_knobby, path_k_new)
            # save_game_data_simple(full_fiar, path_f_new)
            # -------------------------

    # mask the probability matrices with the move matrices
    for m in range(0, len(all_moves_dict)):
        move = all_moves_dict[m]
        mask = move.astype(bool)
        if len(all_prob_dict[m]) != 0:
            all_prob_dict[m] = [all_prob_dict[m][0][mask], all_prob_dict[m][1][mask]]
        if len(all_prob_dict_new[m]) != 0:
            all_prob_dict_new[m] = [all_prob_dict_new[m][0][mask], all_prob_dict_new[m][1][mask]]

    return all_prob_dict, all_moves_dict, all_prob_dict_new, all_games_round


def plot_mixed_effect_model(df_blocked, df_interleaved, ax_blocked_1, ax_blocked_2, ax_interleaved_1, ax_interleaved_2):
    # Blocked condition
    # (frac_moves vs. prob_diff)
    df_blocked_first_half = df_blocked[df_blocked['frac_moves'] < 0.5].copy()
    df_blocked_second_half = df_blocked[df_blocked['frac_moves'] >= 0.5].copy()

    # First half of blocked data
    ax_blocked_1.scatter('frac_moves', 'prob_diff_new', data=df_blocked_first_half,
                         color='black', alpha=0.3, label='Blocked (First Half)')
    model_blocked_first_half = smf.mixedlm('prob_diff_new ~ frac_moves', df_blocked_first_half, groups=df_blocked_first_half['id'])
    result_blocked_first_half = model_blocked_first_half.fit()
    print("result_blocked_first_half\n")
    print(result_blocked_first_half.summary())
    df_blocked_first_half.loc[:, 'fittedvalues'] = result_blocked_first_half.fittedvalues
    ax_blocked_1.plot('frac_moves', 'fittedvalues', data=df_blocked_first_half, color='red', label='Blocked (First Half - LMEM)')

    # Second half of blocked data
    ax_blocked_2.scatter('frac_moves', 'prob_diff_new', data=df_blocked_second_half,
                         color='black', alpha=0.3, label='Blocked (Second Half)')
    model_blocked_second_half = smf.mixedlm('prob_diff_new ~ frac_moves', df_blocked_second_half, groups=df_blocked_second_half['id'])
    result_blocked_second_half = model_blocked_second_half.fit()
    print("result_blocked_second_half\n")
    print(result_blocked_second_half.summary())
    df_blocked_second_half.loc[:, 'fittedvalues'] = result_blocked_second_half.fittedvalues
    ax_blocked_2.plot('frac_moves', 'fittedvalues', data=df_blocked_second_half, color='red', label='Blocked (Second Half - LMEM)')

    # Interleaved condition
    # (odd and even games)
    df_interleaved_odd = df_interleaved[df_interleaved['round'] % 2 != 0].copy()
    df_interleaved_even = df_interleaved[df_interleaved['round'] % 2 == 0].copy()

    # Odd indices of interleaved data
    ax_interleaved_1.scatter('move_number', 'prob_diff_new', data=df_interleaved_odd,
                             color='black', alpha=0.3, label='Interleaved (Odd)')
    model_interleaved_odd = smf.mixedlm('prob_diff_new ~ move_number', df_interleaved_odd, groups=df_interleaved_odd['id'])
    result_interleaved_odd = model_interleaved_odd.fit()
    print("result_interleaved_odd\n")
    print(result_interleaved_odd.summary())
    df_interleaved_odd['fittedvalues'] = result_interleaved_odd.fittedvalues
    ax_interleaved_1.plot('move_number', 'fittedvalues', data=df_interleaved_odd, color='red', label='Interleaved (Odd - LMEM)')

    # Even indices of interleaved data
    ax_interleaved_2.scatter('move_number', 'prob_diff_new', data=df_interleaved_even, alpha=0.5, label='Interleaved (Even)')
    model_interleaved_even = smf.mixedlm('prob_diff_new ~ move_number', df_interleaved_even,
                                         groups=df_interleaved_even['id'])
    result_interleaved_even = model_interleaved_even.fit()
    print("result_interleaved_even\n")
    print(result_interleaved_even.summary())
    df_interleaved_even['fittedvalues'] = result_interleaved_even.fittedvalues
    ax_interleaved_2.plot('move_number', 'fittedvalues', data=df_interleaved_even, color='red', label='Interleaved (Even - LMEM)')

    # Set labels and titles for all subplots
    for ax in [ax_blocked_1, ax_blocked_2, ax_interleaved_1, ax_interleaved_2]:
        ax.set_ylabel('Probability Difference')
        ax.legend()

    ax_blocked_1.set_title('Blocked (First Half)')
    ax_blocked_2.set_title('Blocked (Second Half)')
    ax_interleaved_1.set_title('Interleaved (Odd)')
    ax_interleaved_2.set_title('Interleaved (Even)')

def plot_move_prob_comparison(ax1, ax2, dataFrame, condition):
    y_diff = []
    all_round = []
    all_move = []
    all_move_fraction = []
    max_moves = dataFrame.groupby('id').size().max()

    for i in range(len(dataFrame)):
        all_round.append(dataFrame['round'].iloc[i])
        all_move.append(dataFrame['move_number'].iloc[i])
        all_move_fraction.append(dataFrame['frac_moves'].iloc[i])
        y_diff.append(dataFrame['prob_diff_new'].iloc[i])

    x = np.array(all_move)
    y_diff = [0 if v is None else v for v in y_diff]

    y_diff = np.array(y_diff, dtype=float)

    x_1 = []
    y_1 = []
    x_2 = []
    y_2 = []

    half_fraction = 0.5
    if (condition == 0):  # blocked learning
        for i in all_move_fraction:
            index = all_move_fraction.index(i)
            if i < half_fraction:
                x_1.append(i)
                y_1.append(y_diff[index])
            if i >= half_fraction:
                x_2.append(i)
                y_2.append(y_diff[index])

    elif (condition == 1):  # interleaved learning
        # plot odd and even data separately
        for i, round_num in enumerate(all_round): # i = index, move_num = move number
            if (round_num % 2 != 0): # first game
                x_1.append(x[i])
                y_1.append(y_diff[i])
            else:
                x_2.append(x[i])
                y_2.append(y_diff[i])

    # PLAN A: only kept inliers
    # inliers_1 = remove_outliers(np.array(y_1))
    # x_1 = np.array(x_1)[inliers_1]
    # y_filtered_1 = np.array(y_1)[inliers_1]
    # inliers_2 = remove_outliers(np.array(y_2))
    # x_2 = np.array(x_2)[inliers_2]
    # y_filtered_2 = np.array(y_2)[inliers_2]

    # PLAN B: keep all data
    y_filtered_1 = np.array(y_1)
    y_filtered_2 = np.array(y_2)
    x_1 = np.array(x_1)
    x_2 = np.array(x_2)

    model_1 = np.poly1d(np.polyfit(x_1, y_filtered_1, 1))
    y_smooth_1 = model_1(x_1)
    model_2 = np.poly1d(np.polyfit(x_2, y_filtered_2, 1))
    y_smooth_2 = model_2(x_2)

    # Extracting slopes from the models
    slope_1 = model_1.coefficients[0]  # The first coefficient is the slope for np.poly1d models
    slope_2 = model_2.coefficients[0]
    slope_1_formatted = "{:.2e}".format(slope_1)
    slope_2_formatted = "{:.2e}".format(slope_2)

    ax1.plot(x_1, y_filtered_1, 'o', color='black', markersize=2, alpha=0.1)
    ax2.plot(x_2, y_filtered_2, 'o', color='black', markersize=2, alpha=0.1)
    ax1.plot(x_1, y_smooth_1, color='red', label=f'Slope: {slope_1_formatted}')
    ax2.plot(x_2, y_smooth_2, color='red', label=f'Slope: {slope_2_formatted}')

    if (condition == 0):
        ax1.set_title('Blocked Condition (first game)')
        ax2.set_title('Blocked Condition (second game)')
        ax1.set_xlabel('Move Fraction (0 - 1)')
        ax1.set_xlim(0, half_fraction)
        ax2.set_xlim(half_fraction, 1)
    elif (condition == 1):
        ax1.set_title('Interleaved Condition (first game)')
        ax2.set_title('Interleaved Condition (second game)')
        ax1.set_xlabel('Total Number of Moves')
        ax1.set_xlim(0, 250)
        ax2.set_xlim(0, 250)
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')

def normalize_and_diff_prob(prob, first_game):
    # a new list with length of prob
    diff = [0] * len(prob)

    for i in range(len(prob)):
        prob_fiar = prob[i][0]
        prob_knobby = prob[i][1]
        if (prob_fiar != 0):
            prob_fiar = math.log(prob_fiar[0], 10)
        else:
            prob_fiar = 0

        if (prob_knobby != 0):
            prob_knobby = math.log(prob_knobby[0], 10)
        else:
            prob_knobby = 0

        if (first_game == 0):  # fiar
            diff[i] = prob_fiar - prob_knobby
            diff[i] = float(diff[i])
        elif (first_game == 1):  # knobby
            diff[i] = prob_knobby - prob_fiar
            diff[i] = float(diff[i])

    return diff


def remove_outliers(data):
    # remove outliers
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    threshold = 1.5

    # Identify outliers
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    # outliers = (data < lower_bound) | (data > upper_bound)
    # data_filtered = [0 if d in outliers else d for d in data]
    # return data_filtered

    # ALTERNATIVE: Use a boolean mask for indices within the interquartile range
    inliers = (data >= lower_bound) & (data <= upper_bound)

    return inliers


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

def create_dataframe(id_blocked, id_interleaved):
    data = []
    for id in id_blocked + id_interleaved:
        prob, move, prob_new, round = get_all_move_prob(id)
        first_game = get_frist_game_by_id(id)
        prob = normalize_and_diff_prob(prob, first_game)
        prob_new = normalize_and_diff_prob(prob_new, first_game)
        condition = 'blocked' if id in id_blocked else 'interleaved'
        first_game = 'fiar' if first_game == 0 else 'knobby'
        for i in range(len(prob)):
            frac_moves = (i+1) / len(prob)
            data.append([id, condition, first_game, round[i],
                         i+1, frac_moves, prob[i], prob_new[i]])

    df = pd.DataFrame(data,
                columns=['id', 'condition', 'first_game', 'round',
                         'move_number', 'frac_moves', 'prob_diff', 'prob_diff_new'])

    # test =================
    # Calculate descriptive statistics for prob_diff
    stats_prob_diff = df['prob_diff'].describe()

    # Calculate descriptive statistics for prob_diff_new
    stats_prob_diff_new = df['prob_diff_new'].describe()

    # Print the descriptive statistics
    print("Descriptive Statistics for prob_diff:")
    print(stats_prob_diff)
    print("\nDescriptive Statistics for prob_diff_new:")
    print(stats_prob_diff_new)
    plt.figure(figsize=(12, 6))

    # Histogram for prob_diff
    plt.subplot(1, 2, 1)
    plt.hist(df['prob_diff'].dropna(), bins=30, color='skyblue', edgecolor='black')
    plt.title('Histogram of prob_diff')
    plt.xlabel('prob_diff')
    plt.ylabel('Frequency')

    # Histogram for prob_diff_new
    plt.subplot(1, 2, 2)
    plt.hist(df['prob_diff_new'].dropna(), bins=30, color='orange', edgecolor='black')
    plt.title('Histogram of prob_diff_new')
    plt.xlabel('prob_diff_new')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))

    # Box plot for prob_diff and prob_diff_new
    plt.boxplot([df['prob_diff'].dropna(), df['prob_diff_new'].dropna()], labels=['prob_diff', 'prob_diff_new'])
    plt.title('Box Plot of prob_diff and prob_diff_new')
    plt.ylabel('Values')

    plt.show()

    # ======================
    return df

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

    win_rate_blocked_1, max_b_1 = get_win_rate_all(data_blocked_1)
    win_rate_interleaved_1, max_i_1 = get_win_rate_all(data_interleaved_1)
    win_rate_blocked_2, max_b_2 = get_win_rate_all(data_blocked_2)
    win_rate_interleaved_2, max_i_2 = get_win_rate_all(data_interleaved_2)

    plot_win_rate()
    
    # create dataframe for plotting
    df = create_dataframe(id_blocked, id_interleaved)

    # 2) compare the probabilities of all 100 moves for each rule
    fig, ((ax_b_1, ax_b_2), (ax_i_1, ax_i_2)) = plt.subplots(2, 2, figsize=(12, 8))
    plt.tight_layout(pad=5, h_pad=5, w_pad=1)  # Adjust the padding, 'h_pad' controls the height between rows

    # get block and interleaved data
    all_prob_blocked = df[df['condition'] == 'blocked'].copy()
    all_prob_interleaved = df[df['condition'] == 'interleaved'].copy()

    plot_move_prob_comparison(ax_b_1, ax_b_2, all_prob_blocked, 0)
    plot_move_prob_comparison(ax_i_1, ax_i_2, all_prob_interleaved, 1)

    fig.show()

    # 3) mixed effect model
    df_blocked_copy = df[df['condition'] == 'blocked'].copy()
    df_interleaved_copy = df[df['condition'] == 'interleaved'].copy()

    # access the id column
    df_blocked_copy.loc[:, 'id'] = df_blocked_copy['id'].astype(str)
    df_blocked_copy.loc[:, 'id'] = pd.Categorical(df_blocked_copy['id'])
    df_interleaved_copy.loc[:, 'id'] = df_interleaved_copy['id'].astype(str)
    df_interleaved_copy.loc[:, 'id'] = pd.Categorical(df_interleaved_copy['id'])

    fig_mle, ((ax_blocked_1, ax_blocked_2), (ax_interleaved_1, ax_interleaved_2)) = plt.subplots(2, 2, figsize=(12, 8))
    plot_mixed_effect_model(df_blocked_copy, df_interleaved_copy,
                            ax_blocked_1, ax_blocked_2,
                            ax_interleaved_1, ax_interleaved_2)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()