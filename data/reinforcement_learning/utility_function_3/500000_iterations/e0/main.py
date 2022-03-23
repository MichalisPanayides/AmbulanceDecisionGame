"""Reinforcement learning algorithm with e = 0"""
import copy

import numpy as np

from utils import (
    utility_function_3,
    run_simulation,
    pick_policy,
    accept_policy,
    are_within_limits,
    update_server_utilities,
    output_to_file,
)


def team_expertise_priority(srv, ind):
    """
    Servers priority based on expertise
    """
    if srv.id_number == 1:
        return 0
    if srv.id_number == 4:
        return 1
    return np.random.random()


# Model parameters
lambda_2 = 1
lambda_1 = 0.5
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
            rates[server_id][(u, v)] = mu

# Model parameters in dictionary format
parameters = {
    "lambda_1": lambda_1,
    "lambda_2": lambda_2,
    "num_of_servers": num_of_servers,
    "threshold": threshold,
    "system_capacity": system_capacity,
    "buffer_capacity": buffer_capacity,
    "runtime": 1000,
    "num_of_trials": 5,
    "seed_num": 0,
    "server_priority_function": team_expertise_priority,
}

# Reinforcement learning parameters
initial_rates = copy.deepcopy(rates)
server_utilities = [-float("inf") for srv in range(1, parameters["num_of_servers"] + 1)]
num_of_iterations = 500000
current_utility_function = utility_function_3
e_parameter = 0
all_utilities, all_rates = [], []


# Reinforcement learning algorithm
Qs = run_simulation(parameters, initial_rates)
for _ in range(num_of_iterations):
    new_rates, server_id = pick_policy(Qs, rates, num_of_servers)
    Qs = run_simulation(parameters, new_rates)
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

    all_utilities.append(server_utilities.copy())
    all_rates.append(copy.deepcopy(rates))
    output_to_file(str(server_utilities), "utilities.csv")
    str_rates = str([list(rates[server].values()) for server in rates])
    output_to_file(str_rates, "rates.csv")
