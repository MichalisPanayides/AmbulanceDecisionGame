import csv
import itertools
import time

import numpy as np
import pandas as pd

import ambulance_game as abg


def run_single_experiment(tolerance, **problem_parameters):
    then = time.time()
    abg.markov.calculate_class_2_individuals_best_response(
        tolerance=tolerance, **problem_parameters
    )
    now = time.time()
    return now - then


def read_data(path="main.csv"):
    """
    Read the data file as a pandas data frame
    """
    return pd.read_csv(path)


def write_data(data, path="main.csv"):
    """
    Opens `path` in append mode and write the data
    """
    with open(path, "w") as out_file:
        csv_writer = csv.writer(out_file)
        csv_writer.writerow(data)


def main(
    path="main.csv",
    number_of_repetitions=200,
    tolerance_values=None,
    problem_parameters=None,
):
    """
    Main experiment file.

    Runs a timing experiment on a system with default parameters:

        "lambda_2": 4,
        "lambda_1_1": 3,
        "lambda_1_2": 3,
        "threshold_1": 4,
        "threshold_2": 5,
        "mu_1": 4,
        "mu_2": 3,
        "num_of_servers_1": 2,
        "num_of_servers_2": 3,
        "system_capacity_1": 8,
        "system_capacity_2": 8,
        "buffer_capacity_1": 8,
        "buffer_capacity_2": 8,

    and increasing system_capacity_1

    This reads in the data frame and only run new experiments.
    """
    if tolerance_values is None:
        tolerance_values = np.logspace(stop=-1, start=-10, num=10)

    if problem_parameters is None:
        problem_parameters = {
            "lambda_2": 4,
            "lambda_1_1": 3,
            "lambda_1_2": 3,
            "threshold_1": 4,
            "threshold_2": 5,
            "mu_1": 4,
            "mu_2": 3,
            "num_of_servers_1": 2,
            "num_of_servers_2": 3,
            "system_capacity_1": 8,
            "system_capacity_2": 8,
            "buffer_capacity_1": 8,
            "buffer_capacity_2": 8,
        }

    keys = sorted(problem_parameters.keys())

    try:
        df = read_data()
        cache = set(tuple(row) for _, row in df[keys].iterrows())
    except FileNotFoundError:
        header = ["repetition", "tolerance"] + keys + ["time_taken"]
        write_data(data=header, path=path)

    while True:

        parameter_values = tuple((problem_parameters[key] for key in keys))

        if parameter_values not in cache:
            for tolerance, repetition in itertools.product(
                tolerance_values, range(number_of_repetitions)
            ):
                time_taken = run_single_experiment(
                    tolerance=tolerance, **problem_parameters
                )

                data = [repetition, tolerance] + list(parameter_values) + [time_taken]
                write_data(data=header, path=path)

        problem_parameters["system_capacity_1"] += 1


if __name__ == "__main__":
    main()
