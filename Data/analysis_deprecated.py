import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt


deprecated_id = ['p03', 'p05', 'p06', 'p07', 'p08', 'p09', 'p10', 'p11']
win_four = []
win_knobby = []

def find_duplicate_params(folder_path):
    params_dict = defaultdict(list)

    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name == 'params.json':
                params_path = os.path.join(root, file_name)
                params_dict[file_name].append(params_path)

    # Filter out files with unique names
    duplicate_params = {k: v for k, v in params_dict.items() if len(v) > 1}

    return duplicate_params

def calculate_win_rate(params_path):
    global win_four, win_knobby
    with open(params_path, 'r') as file:
        params_data = json.load(file)

    if (params_data['participant_id'] in deprecated_id):
        return

    # Extract relevant information
    games_condition = params_data.get('condition', 0)
    games_rule = params_data.get('games_rule', [])
    games_results = params_data.get('games_results', [])
    games_count = params_data.get('games_count', 0)

    # Calculate overall win rate
    win_rate_overall = sum(result == 1 for result in games_results) / games_count

    # Calculate win rate for each rule
    # Initialize win counts for each rule
    win_count_four = 0
    win_count_knobby = 0
    game_count_four = 0
    game_count_knobby = 0

    # Calculate win counts for each rule
    for result, rule in zip(games_results, games_rule):
        if rule == 0:  # Four-in-a-row
            game_count_four += 1
            win_count_four += result == 1
        elif rule == 1:  # Knobby
            game_count_knobby += 1
            win_count_knobby += result == 1

    # Calculate win rates
    win_rate_four = win_count_four / game_count_four if game_count_four > 0 else 0
    win_rate_knobby = win_count_knobby / game_count_knobby if game_count_knobby > 0 else 0

    print(params_data['participant_id']," fouriar:", win_rate_four, "knobby:", win_rate_knobby)

    # Update the existing params_data with the calculated win rates
    params_data['win_rate_fouriar'] = win_rate_four
    params_data['win_rate_knobby'] = win_rate_knobby

    # still need to write the updated params_data to the file

    win_four.append(win_rate_four)
    win_knobby.append(win_rate_knobby)

    # Plotting
    # groupped_results = calculate_staircase_win_rate(games_results)
    # draw_plot(list(range(1, len(groupped_results) + 1)), groupped_results, "game count (group=5)", "win rate", params_data['participant_id'], "Overall")
    # draw_plot(game_count_knobby, win_count_knobby, "game count", "win count", params_data['participant_id'], "Knobby")

def calculate_staircase_win_rate(results):
    starcase_wins = []
    group_size = 5
    # calculate win rate for each group
    for i in range(0, len(results), group_size):
        if (i > len(results) - group_size):
            group = results[i:]
            group_win = sum(result == 1 for result in group) / len(group)
        else:
            group = results[i:i + group_size]
            group_win = sum(result == 1 for result in group) / group_size
        starcase_wins.append(group_win)
    return starcase_wins

def draw_plot(x, y, x_label, y_label, title, label):
    plt.scatter(x, y, label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.show()

path = os.path.dirname(os.path.realpath(__file__))
data_folder = path

# get win rates from params
duplicates = find_duplicate_params(data_folder)
for file_name, file_paths in duplicates.items():
    print(f"File '{file_name}' found in {file_paths}")
    for path in file_paths:
        calculate_win_rate(path)

    # calculate average win rate in fouriar and knobby conditions
    avg_win_four = 0
    avg_win_knobby = 0
    for win in win_four:
        avg_win_four += win
    avg_win_four = avg_win_four / len(win_four)
    for win in win_knobby:
        avg_win_knobby += win
    avg_win_knobby = avg_win_knobby / len(win_knobby)

    # print average win rate for each condition
    print("\noverall fouriar:", avg_win_four, "knobby:", avg_win_knobby)
