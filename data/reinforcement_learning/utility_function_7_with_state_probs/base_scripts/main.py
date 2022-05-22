"""Reinforcement learning algorithm with e = 0"""
# pylint: disable=invalid-name
import copy
import sys

import numpy as np

from utils import (
    utility_function_7,
    run_simulation,
    pick_policy,
    accept_policy,
    are_within_limits,
    update_server_utilities,
    output_to_file,
)


def team_expertise_priority(srv, ind):  # pylint: disable=unused-argument
    """
    Servers priority based on expertise
    """
    if srv.id_number == 1:
        return 0
    if srv.id_number == 4:
        return 1
    return np.random.random()


def main(e_parameter, lambda_1=None, lambda_2=None, mu=None):
    """The main function"""
    # Model parameters
    if lambda_1 is None:
        lambda_1 = 0.5
    if lambda_2 is None:
        lambda_2 = 1
    if mu is None:
        mu = 0.7

    num_of_servers = 4
    threshold = 7
    system_capacity = 10
    buffer_capacity = 7

    rates = {}
    for server_id in range(1, num_of_servers + 1):
        rates[server_id] = {}
        for u in range(buffer_capacity + 1):
            for v in range(system_capacity + 1):
                if v >= threshold or u == 0:  # hope I don't regret adding this
                    rates[server_id][(u, v)] = mu

    # Model parameters in dictionary format
    parameters = {
        "lambda_1": lambda_1,
        "lambda_2": lambda_2,
        "num_of_servers": num_of_servers,
        "threshold": threshold,
        "system_capacity": system_capacity,
        "buffer_capacity": buffer_capacity,
        "runtime": 10000,
        "num_of_trials": 20,
        "seed_num": 0,
        "server_priority_function": team_expertise_priority,
    }

    # Reinforcement learning parameters
    initial_rates = copy.deepcopy(rates)
    server_utilities = [
        -float("inf") for srv in range(1, parameters["num_of_servers"] + 1)
    ]
    num_of_iterations = 5000
    current_utility_function = utility_function_7

    # Define the directory name that the results will be stored in
    key_params = [
        "lambda_1",
        "lambda_2",
        "num_of_servers",
        "threshold",
        "system_capacity",
        "buffer_capacity",
    ]
    filename_parameters = {
        key: value for key, value in parameters.items() if key in key_params
    }
    filepath = (
        "results/e="
        + str(e_parameter)
        + ","
        + str(filename_parameters)
        .replace(" ", "")
        .replace("'", "")
        .replace(":", "=")
        .replace("{", "")
        .replace("}", "")
        + ",mu="
        + str(mu)
    )

    # Reinforcement learning algorithm
    Qs, state_probs = run_simulation(parameters, initial_rates)
    for _ in range(num_of_iterations):
        new_rates, server_id = pick_policy(Qs, rates, num_of_servers)
        Qs, new_state_probs = run_simulation(parameters, new_rates)
        if are_within_limits(new_rates) and accept_policy(
            Qs=Qs,
            utility_function=current_utility_function,
            e_parameter=e_parameter,
            server_utilities=server_utilities,
            current_server_id=server_id,
        ):
            server_utilities = update_server_utilities(
                Qs=Qs,
                utility_function=current_utility_function,
                e_parameter=e_parameter,
                num_of_servers=parameters["num_of_servers"],
            )
            rates = copy.deepcopy(new_rates)
            state_probs = copy.deepcopy(new_state_probs)

        output_to_file(str(server_utilities), filepath + "/utilities.csv")

        str_rates = str([list(rates[server].values()) for server in rates])
        output_to_file(str_rates, filepath + "/rates.csv")

        str_state_probs = str(list(state_probs.flatten()))
        output_to_file(str_state_probs, filepath + "/state_probs.csv")


if __name__ == "__main__":
    arguments = sys.argv
    arguments = [
        float(arguments[i]) if i < len(arguments) and arguments[i] != "None" else None
        for i in range(1, 5)
    ]
    e_value = arguments[0]
    lambda_1_value = arguments[1]
    lambda_2_value = arguments[2]
    mu_value = arguments[3]

    if e_value < 0 or e_value > 1:
        raise ValueError("Please provide a value of e between 0 and 1")

    main(
        e_parameter=e_value,
        lambda_1=lambda_1_value,
        lambda_2=lambda_2_value,
        mu=mu_value,
    )
