#!/usr/bin/python

"""
Responsible for aggregating and managing the data to be fed into the
projection model.
"""

import sys, getopt
from lxml import html
import requests
import csv

VERBOSE = False

# Stats for the first 50 running backs, sorted by name alphabetically, for the 2015 season.
DATA_SOURCE_URLS = [
    "https://fantasydata.com/nfl-stats/nfl-fantasy-football-stats.aspx?fs=0&stype=0&sn=1&scope=1&w=0&ew=0&s=&t=0&p=3&st=Name&d=0&ls=Name&live=false&pid=true&minsnaps=4",
    "https://fantasydata.com/nfl-stats/nfl-fantasy-football-stats.aspx?fs=0&stype=0&sn=1&scope=1&w=1&ew=1&s=&t=0&p=3&st=Name&d=0&ls=Name&live=false&pid=true&minsnaps=4",
    "https://fantasydata.com/nfl-stats/nfl-fantasy-football-stats.aspx?fs=0&stype=0&sn=1&scope=1&w=2&ew=2&s=&t=0&p=3&st=Name&d=0&ls=Name&live=false&pid=true&minsnaps=4",
    "https://fantasydata.com/nfl-stats/nfl-fantasy-football-stats.aspx?fs=0&stype=0&sn=1&scope=1&w=3&ew=3&s=&t=0&p=3&st=Name&d=0&ls=Name&live=false&pid=true&minsnaps=4",
    "https://fantasydata.com/nfl-stats/nfl-fantasy-football-stats.aspx?fs=0&stype=0&sn=1&scope=1&w=4&ew=4&s=&t=0&p=3&st=Name&d=0&ls=Name&live=false&pid=true&minsnaps=4",
    "https://fantasydata.com/nfl-stats/nfl-fantasy-football-stats.aspx?fs=0&stype=0&sn=1&scope=1&w=5&ew=5&s=&t=0&p=3&st=Name&d=0&ls=Name&live=false&pid=true&minsnaps=4",
    "https://fantasydata.com/nfl-stats/nfl-fantasy-football-stats.aspx?fs=0&stype=0&sn=1&scope=1&w=6&ew=6&s=&t=0&p=3&st=Name&d=0&ls=Name&live=false&pid=true&minsnaps=4",
    "https://fantasydata.com/nfl-stats/nfl-fantasy-football-stats.aspx?fs=0&stype=0&sn=1&scope=1&w=7&ew=7&s=&t=0&p=3&st=Name&d=0&ls=Name&live=false&pid=true&minsnaps=4",
    "https://fantasydata.com/nfl-stats/nfl-fantasy-football-stats.aspx?fs=0&stype=0&sn=1&scope=1&w=8&ew=8&s=&t=0&p=3&st=Name&d=0&ls=Name&live=false&pid=true&minsnaps=4",
    "https://fantasydata.com/nfl-stats/nfl-fantasy-football-stats.aspx?fs=0&stype=0&sn=1&scope=1&w=9&ew=9&s=&t=0&p=3&st=Name&d=0&ls=Name&live=false&pid=true&minsnaps=4",
    "https://fantasydata.com/nfl-stats/nfl-fantasy-football-stats.aspx?fs=0&stype=0&sn=1&scope=1&w=10&ew=10&s=&t=0&p=3&st=Name&d=0&ls=Name&live=false&pid=true&minsnaps=4",
    "https://fantasydata.com/nfl-stats/nfl-fantasy-football-stats.aspx?fs=0&stype=0&sn=1&scope=1&w=11&ew=11&s=&t=0&p=3&st=Name&d=0&ls=Name&live=false&pid=true&minsnaps=4",
    "https://fantasydata.com/nfl-stats/nfl-fantasy-football-stats.aspx?fs=0&stype=0&sn=1&scope=1&w=12&ew=12&s=&t=0&p=3&st=Name&d=0&ls=Name&live=false&pid=true&minsnaps=4",
    "https://fantasydata.com/nfl-stats/nfl-fantasy-football-stats.aspx?fs=0&stype=0&sn=1&scope=1&w=13&ew=13&s=&t=0&p=3&st=Name&d=0&ls=Name&live=false&pid=true&minsnaps=4",
    "https://fantasydata.com/nfl-stats/nfl-fantasy-football-stats.aspx?fs=0&stype=0&sn=1&scope=1&w=14&ew=14&s=&t=0&p=3&st=Name&d=0&ls=Name&live=false&pid=true&minsnaps=4",
]
COLUMN_NAMES = ["Rank","ID","Name","Position","Week","Team","Opp","Att","RushYds","RushYds/Att","RushTD","Targets","Rec","PassYds","PassTD","Fum","Lost","FantasyPoints"]

RAW_DATA_CSV_PATH = "./raw_data.csv"
TRAINING_DATA_CSV_PATH = "./model_data.train.csv"
TESTING_DATA_CSV_PATH = "./model_data.test.csv"


def print_usage():
    print("""
usage: data_manager.py [-h] [-v] action
    action: Operation to be perfomed (e.g. load, process)
""")


def load_raw_data():
    global VERBOSE
    write_header_flag = True

    if VERBOSE: print("Writing raw data to: " + RAW_DATA_CSV_PATH)
    with open(RAW_DATA_CSV_PATH, "w") as csvfile:
        csvwriter = csv.writer(csvfile, dialect=csv.Dialect.delimiter)

        for url in DATA_SOURCE_URLS:
            if VERBOSE: print("Loading data from: " + url)

            # Download the page.
            page = requests.get(url)

            # Extract stats from the HTML content.
            tree = html.fromstring(page.content)
            stats_cols = tree.xpath('//table[@class="table"]//th//text()')
            stats_vals = tree.xpath('//table[@class="table"]//td//text()')

            if len(COLUMN_NAMES) != len(stats_cols):
                print("ERROR: Unexpected column names: [" + ", ".join(stats_cols) + "]")
                continue

            if write_header_flag:
                csvwriter.writerow(COLUMN_NAMES)
                write_header_flag = False

            stats_count = len(stats_cols)
            player_count = int(len(stats_vals) / stats_count)
            for x in range(0, player_count):
                start_index = x * stats_count
                end_index = start_index + stats_count
                csvwriter.writerow(stats_vals[start_index:end_index])


def process_data():
    raw_stats = []
    if VERBOSE: print("Loading raw data from: " + RAW_DATA_CSV_PATH)
    with open(RAW_DATA_CSV_PATH, "r") as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            raw_stats.append(row)
        if VERBOSE: print("Loaded " + str(len(raw_stats)) + " entries.")

    max_week = 0
    player_stats = dict()
    if VERBOSE: print("Grouping stats by player, per-week.")
    for stats in raw_stats:
        player_id = stats['ID']
        if player_id not in player_stats:
            player_stats[player_id] = dict()
        player_weekly_stats = player_stats[player_id]

        week = stats['Week']
        if week in player_weekly_stats:
            print("ERROR: Duplicate stats found for player " + player_id + " during week " + week)
            continue

        max_week = max(max_week, int(week))
        player_weekly_stats[week] = stats
    if VERBOSE: print("Stats grouped across " + str(len(player_stats)) + " players.")

    if max_week < 4:
        print("ERROR: Insufficient data to pull historical weekly data. Max Week: " + str(max_week))

    if VERBOSE: print("Constructing the input data for the prediction model.")
    model_training_data = []
    model_testing_data = []
    for player_id, player_weekly_stats in player_stats.items():
        for week in range(4, max_week + 1):
            current_week = str(week)
            one_week_ago = str(week - 1)
            two_week_ago = str(week - 2)
            three_week_ago = str(week - 3)

            if ((current_week not in player_weekly_stats) or
                (one_week_ago not in player_weekly_stats) or
                (two_week_ago not in player_weekly_stats) or
                (three_week_ago not in player_weekly_stats)):
                 if VERBOSE: print("Insufficient data found for player " + player_id + " to project a week " + str(week) + " score.")
                 continue
            current_week_stats = player_weekly_stats[current_week]
            one_week_ago_stats = player_weekly_stats[one_week_ago]
            two_week_ago_stats = player_weekly_stats[two_week_ago]
            three_week_ago_stats = player_weekly_stats[three_week_ago]

            avg_attempts = (int(one_week_ago_stats['Att']) + int(one_week_ago_stats['Targets']) +
                            int(two_week_ago_stats['Att']) + int(two_week_ago_stats['Targets']) +
                            int(three_week_ago_stats['Att']) + int(three_week_ago_stats['Targets'])) / 3;
            avg_yards = (int(one_week_ago_stats['RushYds']) + int(one_week_ago_stats['PassYds']) +
                         int(two_week_ago_stats['RushYds']) + int(two_week_ago_stats['PassYds']) +
                         int(three_week_ago_stats['RushYds']) + int(three_week_ago_stats['PassYds'])) / 3;
            avg_tds = (int(one_week_ago_stats['RushTD']) + int(one_week_ago_stats['PassTD']) +
                       int(two_week_ago_stats['RushTD']) + int(two_week_ago_stats['PassTD']) +
                       int(three_week_ago_stats['RushTD']) + int(three_week_ago_stats['PassTD'])) / 3;
            avg_points = (float(one_week_ago_stats['FantasyPoints']) +
                          float(two_week_ago_stats['FantasyPoints']) +
                          float(three_week_ago_stats['FantasyPoints'])) / 3;
            points_bracket = int(float(current_week_stats['FantasyPoints']) / 5)
            performance = min(points_bracket, 4);

            model_data = [
                avg_attempts,
                avg_yards,
                avg_tds,
                avg_points,
                one_week_ago_stats['Att'],
                one_week_ago_stats['RushYds'],
                one_week_ago_stats['RushTD'],
                one_week_ago_stats['Targets'],
                one_week_ago_stats['PassYds'],
                one_week_ago_stats['PassTD'],
                one_week_ago_stats['FantasyPoints'],
                performance
            ]

            if week == max_week:
                if VERBOSE: print("Adding week " + str(week) + " for player " + player_id + " to testing data")
                model_testing_data.append(model_data)
            else:
                if VERBOSE: print("Adding week " + str(week) + " for player " + player_id + " to training data")
                model_training_data.append(model_data)

    if VERBOSE: print("Writing " + str(len(model_training_data)) + " training records to " + TRAINING_DATA_CSV_PATH)
    with open(TRAINING_DATA_CSV_PATH, "w") as csvfile:
        csvwriter = csv.writer(csvfile, dialect=csv.Dialect.delimiter)
        csvwriter.writerow([len(model_training_data), 11, "0-5", "5-10", "10-15", "15-20", "20+"])
        for row in model_training_data:
            csvwriter.writerow(row)

    if VERBOSE: print("Writing " + str(len(model_testing_data)) + " testing records to " + TESTING_DATA_CSV_PATH)
    with open(TESTING_DATA_CSV_PATH, "w") as csvfile:
        csvwriter = csv.writer(csvfile, dialect=csv.Dialect.delimiter)
        csvwriter.writerow([len(model_testing_data), 11, "0-5", "5-10", "10-15", "15-20", "20+"])
        for row in model_testing_data:
            csvwriter.writerow(row)


def main(argv=None):
    global VERBOSE

    if len(argv) == 0:
        print_usage()
        sys.exit()

    options = "hv"
    long_options = [ "help", "verbose" ]
    try:
        opts, args = getopt.getopt(argv, options, long_options)
    except getopt.GetoptError:
        print_usage();
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print_usage()
            sys.exit()
        elif opt in [ "-v", "--verbose" ]:
            VERBOSE = True

    if len(args) == 0:
        print_usage()
        sys.exit(1)

    action = args[0]
    if action == "load":
        print("Loading data.")
        load_raw_data()
        if VERBOSE: print("Finished loading data.")
    elif action == "process":
        print("Processing data.")
        process_data()
        if VERBOSE: print("Finished processing data.")


if __name__ == "__main__":
   main(sys.argv[1:])
