import csv
import itertools
import pathlib
import random
import sys

import numpy as np
import pandas as pd

import ambulance_game as abg


def read_data(path=pathlib.Path("data/parameters/main.csv")):
    """
    Read the data contents of the file as a pandas data frame
    """
    return pd.read_csv(path)


def write_data_to_csv(data, path=pathlib.Path("data/parameters/main.csv")):
    """
    Opens `path` in append mode and write the data
    """
    with path.open("a", newline="") as outfile:
        csv_writer = csv.writer(outfile)
        csv_writer.writerow(data)


def initialise_parameters_directory(**problem_parameters):
    """
    Creates the parameters directory along with the readme file and the empty
    main.csv file that will hold all investigated parameters
    """
    directory = pathlib.Path("data/parameters/")
    directory.mkdir(parents=True, exist_ok=True)
    readme_contents = (
        "# Parameters"
        "\n\nThis directory keeps track of all the parameters investigated so far. The"
        "\ncontents of `main.csv` correspond to parameter values of the model in the"
        "\nfollowing order:\n"
    )
    with (directory / "README.md").open("w") as file:
        file.write(readme_contents)
        file.write("".join("\n\t" + key + "," for key, _ in problem_parameters.items()))
    if not (directory / "main.csv").exists():
        write_data_to_csv(sorted(problem_parameters.keys()))


def generate_data_for_current_parameters(processes, **problem_parameters):
    """
    Generates the routing matrix, the payoff matrix for the row player (A) and
    the payoff matrix for the column player (B), given a set of parameters.

    Returns
    -------
    numpy array
    numpy array
    numpy array
    """
    payoff_matrix_A, payoff_matrix_B, routing_matrix = abg.game.get_payoff_matrices(
        processes=processes,
        **problem_parameters,
    )
    return routing_matrix, payoff_matrix_A, payoff_matrix_B


def write_README_for_current_parameters_directory(readme_path, **problem_parameters):
    """
    Writes the readme file (README.md) for the directory of the given set of
    problem parameters.

    Parameters
    ----------
    readme_path : pathlib.Path object
        the path where the readme file will be located
    """
    parameters_string = "".join(
        "\n\t" + key + " = " + str(value) for key, value in problem_parameters.items()
    )
    readme_contents = (
        "# Experiments for game"
        "\n\nThis directory consists of the data for the following set of parameters: \n"
        + "".join(parameters_string)
        + "\n\nThe directory is structured in the following way:\n\n"
        "\t|-- main.csv\n"
        "\t|-- README.md\n"
        "\t|-- routing\n"
        "\t|   |-- main.csv\n"
        "\t|   |-- README.md \n"
        "\t|-- A\n"
        "\t|   |-- main.csv\n"
        "\t|   |-- README.md\n"
        "\t|-- B\n"
        "\t|   |-- main.csv\n"
        "\t|   |-- README.md\n"
    )
    with readme_path.open("w") as file:
        file.write(readme_contents)


def write_README_for_current_parameters_sub_directories(readme_path, output_name):
    """
    Writes the readme files (README.md) for the sub-directories routing, A or B

    Parameters
    ----------
    readme_path : pathlib.Path object
        the path that the readme will go on
    output_name : string
        the numpy that we write the readme for, i.e. "routing matrix",
        "payoff matrix of A", "payoff matrix of B"
    """
    with readme_path.open("w") as file:
        file.write("# " + output_name[0].upper() + output_name[1:])
        file.write(
            "\n\nThis sub-directory contains the value of "
            + output_name
            + " in `main.csv`"
        )


def create_sub_directories_for_current_parameters(
    routing_matrix,
    payoff_matrix_A,
    payoff_matrix_B,
    path=pathlib.Path("data"),
    **problem_parameters,
):
    """
    Create the directory and all of the sub-directories for the current set of
    parameters.
    """
    directory_name = "_".join(
        str(round(value, 2)) for value in problem_parameters.values()
    )
    new_directory = path / directory_name
    new_directory.mkdir(parents=True, exist_ok=True)

    write_data_to_csv(data=problem_parameters.values(), path=new_directory / "main.csv")
    write_README_for_current_parameters_directory(
        readme_path=new_directory / "README.md", **problem_parameters
    )

    routing_subdirectory = new_directory / "Routing"
    A_subdirectory = new_directory / "A"
    B_subdirectory = new_directory / "B"
    routing_subdirectory.mkdir(parents=True, exist_ok=True)
    A_subdirectory.mkdir(parents=True, exist_ok=True)
    B_subdirectory.mkdir(parents=True, exist_ok=True)

    np.savetxt(routing_subdirectory / "main.csv", routing_matrix, delimiter=",")
    np.savetxt(A_subdirectory / "main.csv", payoff_matrix_A, delimiter=",")
    np.savetxt(B_subdirectory / "main.csv", payoff_matrix_B, delimiter=",")

    write_README_for_current_parameters_sub_directories(
        readme_path=routing_subdirectory / "README.md",
        output_name="routing matrix",
    )
    write_README_for_current_parameters_sub_directories(
        readme_path=A_subdirectory / "README.md",
        output_name="payoff matrix A",
    )
    write_README_for_current_parameters_sub_directories(
        readme_path=B_subdirectory / "README.md",
        output_name="payoff matrix B",
    )


def main(
    path=pathlib.Path(),
    problem_parameters=None,
    processes=None,
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
    problem_parameters = dict(sorted(problem_parameters.items()))

    try:
        df = read_data()
        cache = set(tuple(row) for _, row in df[problem_parameters.keys()].iterrows())
    except FileNotFoundError:
        initialise_parameters_directory(**problem_parameters)
        cache = set()

    while True:

        lambda_2_values = np.linspace(
            start=0.1,
            stop=2
            * (
                problem_parameters["mu_1"] * problem_parameters["num_of_servers_1"]
                + problem_parameters["mu_2"] * problem_parameters["num_of_servers_2"]
            ),
            num=30,
        )
        lambda_1_1_values = np.linspace(
            start=0,
            stop=problem_parameters["mu_1"] * problem_parameters["num_of_servers_1"],
            num=20,
        )
        lambda_1_2_values = np.linspace(
            start=0,
            stop=problem_parameters["mu_2"] * problem_parameters["num_of_servers_2"],
            num=20,
        )
        alpha_values = np.linspace(
            start=0,
            stop=1,
            num=21,
        )
        target_values = np.linspace(
            start=0,
            stop=10,
            num=20,
        )

        for lambda_2, lambda_1_1, lambda_1_2, alpha, target in itertools.product(
            lambda_2_values,
            lambda_1_1_values,
            lambda_1_2_values,
            alpha_values,
            target_values,
        ):
            problem_parameters["lambda_2"] = round(lambda_2, 2)
            problem_parameters["lambda_1_1"] = round(lambda_1_1, 2)
            problem_parameters["lambda_1_2"] = round(lambda_1_2, 2)
            problem_parameters["alpha"] = round(alpha, 2)
            problem_parameters["target"] = round(target, 2)

            if problem_parameters.values() not in cache:
                cache.add(problem_parameters.values())
                (
                    routing_matrix,
                    payoff_matrix_A,
                    payoff_matrix_B,
                ) = generate_data_for_current_parameters(
                    processes=processes, **problem_parameters
                )

                create_sub_directories_for_current_parameters(
                    routing_matrix=routing_matrix,
                    payoff_matrix_A=payoff_matrix_A,
                    payoff_matrix_B=payoff_matrix_B,
                    **problem_parameters,
                )
                write_data_to_csv(data=problem_parameters.values())

        problem_parameters["mu_1"] = round(random.uniform(0.1, 10), 1)
        problem_parameters["mu_2"] = round(random.uniform(0.1, 10), 1)
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
    arguments = sys.argv
    if len(arguments) == 2:
        try:
            processes = int(arguments[1])
        except ValueError:
            processes = None
    else:
        processes = None
    main(processes=processes)
