import csv
import random

import numpy as np
import pandas as pd

import ambulance_game as abg


def generate_data_for_current_parameters(**problem_parameters):
    routing_parameters = {
        key: value for key, value in problem_parameters.items() if key != "target"
    }
    routing_matrix = abg.game.get_routing_matrix(**routing_parameters)
    payoff_matrix_A, payoff_matrix_B = abg.game.get_payoff_matrices(
        routing_matrix=routing_matrix, **problem_parameters
    )
    return routing_matrix, payoff_matrix_A, payoff_matrix_B


def read_data(path="main.csv"):
    """
    Read the data file as a pandas data frame
    """
    return pd.read_csv(path)


def write_data(data, path="main.csv"):
    """
    Opens `path` in append mode and write the data
    """
    with open(path, "a", newline="") as out_file:
        csv_writer = csv.writer(out_file)
        csv_writer.writerow(data)


def main(
    path="main.csv",
    problem_parameters=None,
):
    """
    Main experiment file.

    Gets the routing matrix and the payoff matrices on a system with default
    parameters:

        "lambda_2": 1,
        "lambda_1_1": 1,
        "lambda_1_2": 1,
        "mu_1": 2,
        "mu_2": 2,
        "num_of_servers_1": 1,
        "num_of_servers_2": 1,
        "system_capacity_1": 2,
        "system_capacity_2": 2,
        "buffer_capacity_1": 2,
        "buffer_capacity_2": 2,
        "alpha" : 0,
        "target" : 1,

    and increasing system_capacity_1

    This reads in the data frame and only run new experiments.
    """
    if problem_parameters is None:
        problem_parameters = {
            "lambda_2": 1,
            "lambda_1_1": 1,
            "lambda_1_2": 1,
            "mu_1": 2,
            "mu_2": 2,
            "num_of_servers_1": 1,
            "num_of_servers_2": 1,
            "system_capacity_1": 2,
            "system_capacity_2": 2,
            "buffer_capacity_1": 2,
            "buffer_capacity_2": 2,
            "alpha": 0,
            "target": 1,
        }

    keys = sorted(problem_parameters.keys())

    try:
        df = read_data()
        cache = set(tuple(row) for _, row in df[keys].iterrows())
    except FileNotFoundError:
        header = keys + ["routing matrix", "payoff matrix A", "payoff matrix B"]
        write_data(data=header, path=path)
        cache = set()

    while True:

        parameter_values = tuple((problem_parameters[key] for key in keys))

        if parameter_values not in cache:

            (
                routing_matrix,
                payoff_matrix_A,
                payoff_matrix_B,
            ) = generate_data_for_current_parameters(**problem_parameters)
            data = list(parameter_values) + [
                routing_matrix.tostring().hex(),
                payoff_matrix_A.tostring().hex(),
                payoff_matrix_B.tostring().hex(),
            ]
            write_data(data=data, path=path)

        problem_parameters["lambda_2"] = round(random.uniform(0, 10), 1)
        problem_parameters["lambda_1_1"] = round(random.uniform(0, 10), 1)
        problem_parameters["lambda_1_2"] = round(random.uniform(0, 10), 1)
        problem_parameters["mu_1"] = round(random.uniform(0, 10), 1)
        problem_parameters["mu_2"] = round(random.uniform(0, 10), 1)
        problem_parameters["num_of_servers_1"] = random.randint(1, 10)
        problem_parameters["num_of_servers_2"] = random.randint(1, 10)
        problem_parameters["system_capacity_1"] = random.randint(
            problem_parameters["num_of_servers_1"], 10
        )
        problem_parameters["system_capacity_2"] = random.randint(
            problem_parameters["num_of_servers_2"], 10
        )
        problem_parameters["buffer_capacity_1"] = random.randint(1, 10)
        problem_parameters["buffer_capacity_2"] = random.randint(1, 10)
        problem_parameters["alpha"] = int(random.random() * 11) / 10


if __name__ == "__main__":
    main()
