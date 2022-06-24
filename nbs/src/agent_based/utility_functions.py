import numpy as np


def utility_function_1(Qs, server_id, e_parameter=0.5):
    """
    Utility function 1 is the weighted average of the following:
        - The number of individuals served
        - The amount of time the server was idle
    Here we iterate over all simulations to get the average of both the number
    of individuals served and the amount of idle time for all values of the
    server with id "server_id".
    """
    server_all_simulations = [Q.nodes[2].servers[server_id - 1] for Q in Qs]
    mean_served_inds = np.mean(
        [len(server.served_inds) for server in server_all_simulations]
    )
    mean_idle_time = np.mean(
        [
            (Qs[Q_id].current_time - srv.busy_time)
            for Q_id, srv in enumerate(server_all_simulations)
        ]
    )
    return e_parameter * mean_served_inds + (1 - e_parameter) * mean_idle_time


def utility_function_2(Qs, server_id, e_parameter=0.5):
    """
    Utility function 2 is the weighted average of the following:
        - The proportion of individuals served
        - The proportion of time the server was idle
    Here we iterate over all simulations to get the average of both the
    proportion of served individuals and the proportio of idle time for all
    values of the server with id "server_id".
    """
    server_all_simulations = [Q.nodes[2].servers[server_id - 1] for Q in Qs]
    mean_served_inds_prop = np.mean(
        [
            len(srv.served_inds) / len(Qs[Q_id].nodes[-1].all_individuals)
            for Q_id, srv in enumerate(server_all_simulations)
        ]
    )
    mean_idle_prop = np.mean(
        [
            (Qs[Q_id].current_time - srv.busy_time) / Qs[Q_id].current_time
            for Q_id, srv in enumerate(server_all_simulations)
        ]
    )
    return e_parameter * mean_served_inds_prop + (1 - e_parameter) * mean_idle_prop


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


def utility_function_4(Qs, server_id, e_parameter=0.5):
    """
    Utility function 4 is the weighted average of the following:
        - The mean service rate
        - The proportion of time the server was idle
    Here we iterate over all simulations to get the average of both the service
    rate and the idle time for all values of the server with id "server_id".
    """
    server_all_simulations = [Q.nodes[2].servers[server_id - 1] for Q in Qs]
    mean_service_rate = np.mean(
        [1 / np.mean(srv.service_times) for srv in server_all_simulations]
    )
    idle_time = np.mean(
        [
            (Qs[Q_id].current_time - srv.busy_time) / Qs[Q_id].current_time
            for Q_id, srv in enumerate(server_all_simulations)
        ]
    )
    return e_parameter * mean_service_rate + (1 - e_parameter) * idle_time


def utility_function_5(Qs, server_id, e_parameter=0.5):
    """
    Utility function 5 is the weighted average of the following
        - The proportion of individuals served
        - The mean service time
    Here we iterate over all simulations to get the average of both the
    proportion of served individuals and the service time for all values of the
    server with id "server_id".
    """
    server_all_simulations = [Q.nodes[2].servers[server_id - 1] for Q in Qs]
    mean_served_inds_prop = np.mean(
        [
            len(srv.served_inds) / len(Qs[Q_id].nodes[-1].all_individuals)
            for Q_id, srv in enumerate(server_all_simulations)
        ]
    )
    mean_service_time = np.mean(
        [np.mean(srv.service_times) for srv in server_all_simulations]
    )
    return e_parameter * mean_served_inds_prop + (1 - e_parameter) * mean_service_time


def utility_function_6(Qs, server_id, e_parameter=0.5):
    """
    Utility function 6 is the weighted average of the following
        - The proportion of individuals served
        - The mean service rate
    Here we iterate over all simulations to get the average of both the
    proportion of served individuals and the service rate for all values of the
    server with id "server_id".
    """
    server_all_simulations = [Q.nodes[2].servers[server_id - 1] for Q in Qs]
    mean_served_inds_prop = np.mean(
        [
            len(srv.served_inds) / len(Qs[Q_id].nodes[-1].all_individuals)
            for Q_id, srv in enumerate(server_all_simulations)
        ]
    )
    mean_service_rate = np.mean(
        [1 / np.mean(srv.service_times) for srv in server_all_simulations]
    )
    return e_parameter * mean_served_inds_prop + (1 - e_parameter) * mean_service_rate


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


def get_utility_values(utility_function, Q, e_parameter=0.5):
    """
    Returns a list of utility values for each server in the queue
    """
    all_servers = Q.nodes[2].servers
    all_utilities = [utility_function(Q, server, e_parameter) for server in all_servers]
    return all_utilities
