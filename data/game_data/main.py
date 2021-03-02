import csv
import itertools
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
        header = keys + ["routing_matrix", "payoff_matrix_A", "payoff_matrix_B"]
        write_data(data=header, path=path)
        cache = set()

    while True:

        parameter_values = tuple((problem_parameters[key] for key in keys))

        lambda_2_values = np.linspace(
            start=0,
            stop=problem_parameters["mu_1"] * problem_parameters["system_capacity_1"]
            + problem_parameters["mu_2"] * problem_parameters["system_capacity_2"],
            num=10,
        )
        lambda_1_1_values = np.linspace(
            start=0,
            stop=problem_parameters["mu_1"] * problem_parameters["system_capacity_1"],
            num=10,
        )
        lambda_1_2_values = np.linspace(
            start=0,
            stop=problem_parameters["mu_2"] * problem_parameters["system_capacity_2"],
            num=10,
        )
        alpha_values = np.linspace(
            start=0,
            stop=1,
            num=11,
        )

        for lambda_2, lambda_1_1, lambda_1_2, alpha in itertools.product(
            lambda_2_values, lambda_1_1_values, lambda_1_2_values, alpha_values
        ):
            problem_parameters["lambda_2"] = lambda_2
            problem_parameters["lambda_1_1"] = lambda_1_1
            problem_parameters["lambda_1_2"] = lambda_1_2
            problem_parameters["alpha"] = alpha

            if parameter_values not in cache:
                cache.add(parameter_values)
                (
                    routing_matrix,
                    payoff_matrix_A,
                    payoff_matrix_B,
                ) = generate_data_for_current_parameters(**problem_parameters)
                data = list(parameter_values) + [
                    np.array2string(routing_matrix, separator=","),
                    np.array2string(payoff_matrix_A, separator=","),
                    np.array2string(payoff_matrix_B, separator=","),
                ]
                write_data(data=data, path=path)

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


if __name__ == "__main__":
    main()
