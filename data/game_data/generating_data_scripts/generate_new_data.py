import csv
import hashlib
import itertools
import pathlib
import random
import sys

import numpy as np
import pandas as pd

import ambulance_game as abg


def read_data(path=pathlib.Path("data/_parameters/main.csv")):
    """
    Read the data contents of the file as a pandas data frame
    """
    return pd.read_csv(path)


def write_data_to_csv(data, path=pathlib.Path("data/_parameters/main.csv")):
    """
    Opens `path` in append mode and write the data
    """
    with path.open("a", newline="") as outfile:
        csv_writer = csv.writer(outfile)
        csv_writer.writerow(data)


def initialise_parameters_directory(**problem_parameters):
    """
    Creates the parameters directory along with the readme file and the empty
    main.csv file that will hold all investigated parameters along with their
    hash value.
    """
    directory = pathlib.Path("data/_parameters/")
    directory.mkdir(parents=True, exist_ok=True)
    readme_contents = (
        "# Parameters"
        "\n\nThis directory keeps track of all the parameters investigated so far. The"
        "\ncontents of `main.csv` correspond to parameter values of the model in the"
        "\nfollowing order:\n"
        + "".join(
            "\n\t" + key + ","
            for key in tuple(problem_parameters.keys()) + ("hash_value",)
        )
        + "\n\nNote here that the `hash_value` parameter is a bit-array that "
        "holds a unique id \nfor each set of parameters and payoff matrices "
        "that also corresponds to the \ndirectory for that set of parameters"
    )
    with (directory / "README.md").open("w") as file:
        file.write(readme_contents)
    if not (directory / "main.csv").exists():
        write_data_to_csv(tuple(sorted(problem_parameters.keys())) + ("hash_value",))


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
        "\t|-- main.npz\n"
        "\t|-- README.md\n"
        "\nwhere `main.csv` holds the values of the parameters and "
        "`main.npz` holds the \nvalues of the generated data.\n\n"
        "To read the data from `main.npz` in python run:\n\n"
        "```python\n"
        ">>> import numpy as np\n"
        ">>> loaded = np.load('main.npz')\n"
        ">>> print(loaded['routing_matrix'])\n"
        ">>> print(loaded['payoff_matrix_A'])\n"
        ">>> print(loaded['payoff_matrix_B'])\n"
        "```"
    )
    with readme_path.open("w") as file:
        file.write(readme_contents)


def get_hash_value_of_parameters_and_matrices(
    problem_parameters, routing_matrix, payoff_matrix_A, payoff_matrix_B
):
    hash_object = hashlib.md5(
        "".join(str(value) for value in problem_parameters.values()).encode("utf-8")
    )
    hash_object.update(routing_matrix)
    hash_object.update(payoff_matrix_A)
    hash_object.update(payoff_matrix_B)
    return hash_object.hexdigest()


def create_and_update_directories_with_current_parameter_values(
    routing_matrix,
    payoff_matrix_A,
    payoff_matrix_B,
    path=pathlib.Path("data"),
    **problem_parameters,
):
    """
    Create the directory for the current set of parameters, under a unique name
    created by the hash value of the string of all the parameters and the three
    generated numpy array objects. After creating the directory, save the values
    of the matrices in a compressed main.npz file, write a README.md file with
    instructions on how the directory is structured and how to read the data and
    save the values of the parameters in a main.csv file. Lastly update the
    main.csv file located in parameters/ with the parameters values and the hash
    value (which is also the directory name).
    """

    hash_value = get_hash_value_of_parameters_and_matrices(
        problem_parameters=problem_parameters,
        routing_matrix=routing_matrix,
        payoff_matrix_A=payoff_matrix_A,
        payoff_matrix_B=payoff_matrix_B,
    )
    new_directory = path / hash_value
    new_directory.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        file=new_directory / "main",
        routing_matrix=routing_matrix,
        payoff_matrix_A=payoff_matrix_A,
        payoff_matrix_B=payoff_matrix_B,
    )

    write_README_for_current_parameters_directory(
        readme_path=new_directory / "README.md", **problem_parameters
    )
    write_data_to_csv(data=problem_parameters.values(), path=new_directory / "main.csv")
    write_data_to_csv(data=tuple(problem_parameters.values()) + (hash_value,))


def main(
    path=pathlib.Path(),
    problem_parameters=None,
    processes=None,
):
    """
    Main experiment function.

    Gets the routing matrix and the payoff matrices on a system with a default
    set of parameters and explores new sets as it progresses. This reads in the
    data frame and only run new experiments.
    """
    if problem_parameters is None:
        # problem_parameters = {
        #     "lambda_2": None,
        #     "lambda_1_1": 4.5,
        #     "lambda_1_2": 6,
        #     "mu_1": 2,
        #     "mu_2": 3,
        #     "num_of_servers_1": 3,
        #     "num_of_servers_2": 2,
        #     "system_capacity_1": 6,
        #     "system_capacity_2": 7,
        #     "buffer_capacity_1": 5,
        #     "buffer_capacity_2": 4,
        #     "alpha": 0.9,
        #     "target": 2,
        # }
        problem_parameters = {
            "lambda_2": 10.7,
            "lambda_1_1": 4.5,
            "lambda_1_2": 6,
            "mu_1": 2,
            "mu_2": 3,
            "num_of_servers_1": 3,
            "num_of_servers_2": 2,
            "system_capacity_1": 6,
            "system_capacity_2": 7,
            "buffer_capacity_1": 5,
            "buffer_capacity_2": 4,
            "alpha": 0.9,
            "target": 2,
        }

    problem_parameters = dict(sorted(problem_parameters.items()))

    try:
        df = read_data()
        cache = set(tuple(row) for _, row in df[problem_parameters.keys()].iterrows())
    except FileNotFoundError:
        initialise_parameters_directory(**problem_parameters)
        cache = set()

    system_capacity_1_values = np.array([12, 24])
    system_capacity_2_values = np.array([14, 28])

    buffer_capacity_1_values = np.array([5, 10, 20])
    buffer_capacity_2_values = np.array([4, 8, 16])

    for (
        system_capacity_1,
        system_capacity_2,
        buffer_capacity_1,
        buffer_capacity_2,
    ) in itertools.product(
        system_capacity_1_values,
        system_capacity_2_values,
        buffer_capacity_1_values,
        buffer_capacity_2_values,
    ):
        problem_parameters["system_capacity_1"] = system_capacity_1
        problem_parameters["system_capacity_2"] = system_capacity_2
        problem_parameters["buffer_capacity_1"] = buffer_capacity_1
        problem_parameters["buffer_capacity_2"] = buffer_capacity_2

        if tuple(problem_parameters.values()) not in cache:
            cache.add(tuple(problem_parameters.values()))
            (
                routing_matrix,
                payoff_matrix_A,
                payoff_matrix_B,
            ) = generate_data_for_current_parameters(
                processes=processes, **problem_parameters
            )

            create_and_update_directories_with_current_parameter_values(
                routing_matrix=routing_matrix,
                payoff_matrix_A=payoff_matrix_A,
                payoff_matrix_B=payoff_matrix_B,
                **problem_parameters,
            )


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
