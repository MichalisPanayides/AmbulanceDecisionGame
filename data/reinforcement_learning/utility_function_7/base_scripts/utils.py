"""Functions required for RL algorithm"""
# pylint: disable=invalid-name

import copy
import os
import itertools

import numpy as np

import ambulance_game as abg


def utility_function_3(Qs, server_id, e_parameter=0.5):
    """
    Utility function 3 is the weighted average of the following:
        - The mean service time
        - The proportion of time the server was idle
    Here we iterate over all simulations to get the average of both the service
    time and the idle time for all values of the server with id "server_id".
    """
    server_all_simulations = [Q.nodes[2].servers[server_id - 1] for Q in Qs]
    mean_service_time = np.mean(
        [np.mean(srv.service_times) for srv in server_all_simulations]
    )
    idle_time = np.mean(
        [
            (Qs[Q_id].current_time - srv.busy_time) / Qs[Q_id].current_time
            for Q_id, srv in enumerate(server_all_simulations)
        ]
    )
    return e_parameter * mean_service_time + (1 - e_parameter) * idle_time


def utility_function_7(Qs, server_id, e_parameter=0.5):
    """
    Utility function 7 is the weighted average of the following
        - The proportion of individuals not lost to the system
        - The proportion of time the server was idle
    """
    server_all_simulations = [Q.nodes[2].servers[server_id - 1] for Q in Qs]
    all_lost_individuals = [
        len(Q.rejection_dict[1][0]) + len(Q.rejection_dict[2][0]) for Q in Qs
    ]
    all_accepted_individuals = [len(Q.nodes[-1].all_individuals) for Q in Qs]

    mean_proportion_accepted = np.mean(
        [
            accepted_inds / (accepted_inds + lost_inds)
            for accepted_inds, lost_inds in zip(
                all_accepted_individuals, all_lost_individuals
            )
        ]
    )

    idle_proportion = np.mean(
        [
            (Qs[Q_id].current_time - srv.busy_time) / Qs[Q_id].current_time
            for Q_id, srv in enumerate(server_all_simulations)
        ]
    )
    return e_parameter * mean_proportion_accepted + (1 - e_parameter) * idle_proportion


def pick_a_state(Q):
    """
    Pick a state to update.
    """
    all_states = Q.statetracker.state_probabilities()
    all_visited_states = [state for state in all_states if all_states[state] > 0]
    index_choice = np.random.randint(len(all_visited_states))
    state = all_visited_states[index_choice]
    return state


def pick_policy(Qs, rates, num_of_servers):
    """
    Pick a new policy for a single server.
    """
    new_rates = copy.deepcopy(rates)
    Q = np.random.choice(Qs)
    srv = np.random.randint(1, num_of_servers + 1)
    state = pick_a_state(Q)
    scale = np.random.uniform(0, 2)
    new_rates[srv][state] *= scale
    return new_rates, srv


def accept_policy(
    Qs, utility_function, e_parameter, server_utilities, current_server_id
):
    """
    Accept or reject a policy.
    """
    condition = server_utilities[current_server_id - 1] <= utility_function(
        Qs, current_server_id, e_parameter
    )
    return condition


def are_within_limits(rates):
    """
    Check if the rates are within the limits.
    """
    for srv in rates:
        for state in rates[srv]:
            rate = rates[srv][state]
            if rate < 0 or rate > 5:
                print(f"Rate {rate} out of bounds: Server {srv}, State {state}")
                return False
    return True


def update_server_utilities(Qs, utility_function, e_parameter, num_of_servers):
    """
    Update the utilities of all servers.
    """
    new_utilities = [
        utility_function(Qs, srv_id, e_parameter)
        for srv_id in range(1, num_of_servers + 1)
    ]
    return new_utilities


def run_simulation(parameters, rates):
    """
    Run the simulation
    """
    simulations = abg.simulation.simulate_model(
        lambda_1=parameters["lambda_1"],
        lambda_2=parameters["lambda_2"],
        mu=rates,
        num_of_servers=parameters["num_of_servers"],
        threshold=parameters["threshold"],
        system_capacity=parameters["system_capacity"],
        buffer_capacity=parameters["buffer_capacity"],
        runtime=parameters["runtime"],
        num_of_trials=parameters["num_of_trials"],
        seed_num=parameters["seed_num"],
        server_priority_function=parameters["server_priority_function"],
    )
    if not isinstance(simulations, list):
        simulations = [simulations]
    return simulations


def output_to_file(utilist, filepath="demo.csv"):
    """
    Output the utilities to a file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "a") as f:
        f.write(utilist + "\n")


def reconstruct_rates(rates_from_file, system_capacity, buffer_capacity, threshold):
    """
    Reconstruct rates dictionary where it will be of the form:
        rates = dict{
            iteration : dict{
                server_id: dict{
                    state: rate
                    }
                }
            }

    I changed main.py after the first two experiments and the results are now
    saved in two different ways. That's why I needed to ude the two if
    statements. The two if statements are:
    - If num_of_states == len(all_states) means that there is one entry for
            every valid rate for each server
    - Elif num_of_states == (system_capacity + 1) * (buffer_capacity + 1) means
            that there is one entry for all possible combinations of (u,v)
            where some are not valid

    e.g. T=3, N=4, M=2 => state (1,1) does not exist in the first case
            while it is on the second (stupid Mike)
    """

    num_of_servers = len(rates_from_file)
    num_of_iterations = len(rates_from_file[0])
    num_of_states = len(rates_from_file[0][0])

    all_states = abg.markov.build_states(
        threshold=threshold,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
    )
    if num_of_states == len(all_states):
        raise NotImplementedError("This code block is not implemented yet")
    elif num_of_states == (system_capacity + 1) * (buffer_capacity + 1):
        rates = {}
        for iteration in range(num_of_iterations):
            rates[iteration] = {}
            for server_id in range(1, num_of_servers + 1):
                rates[iteration][server_id] = {}
                for index, (u, v) in enumerate(
                    itertools.product(
                        range(buffer_capacity + 1), range(system_capacity + 1)
                    )
                ):
                    if v >= threshold or u == 0:
                        rates[iteration][server_id][(u, v)] = rates_from_file[
                            server_id - 1
                        ][iteration][index]
        return rates
    else:
        raise Exception("Dunno what you on about mate")


def reconstruct_rates_matrix_from_dictionary(rates_dict):
    """
    Reconstruct rates matrix from dictionary.
    """
    buffer_capacity, system_capacity = max(list(rates_dict.keys()))
    rates_array = np.empty((buffer_capacity + 1, system_capacity + 1)) * np.nan
    for (u, v), rate in rates_dict.items():
        rates_array[(u, v)] = rate
    return rates_array
