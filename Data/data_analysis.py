import os
import json
from collections import defaultdict

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
    with open(params_path, 'r') as file:
        params_data = json.load(file)

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

# Get the current script's directory (Data in your case)
path = os.path.dirname(os.path.realpath(__file__))
data_folder = path
#print("data_folder", data_folder)

# Example usage:
duplicates = find_duplicate_params(data_folder)
#print("duplicates", duplicates)

for file_name, file_paths in duplicates.items():
    print(f"File '{file_name}' found in multiple subfolders:")
    for path in file_paths:
        calculate_win_rate(path)
