import numpy as np


def utility_function_1(Q, server, e_parameter=0.5):
    """
    Utility function 1 is the weighted average of the following:
        - The number of individuals served
        - The amount of time the server was idle
    """
    served_inds = len(server.served_inds)
    idle_times = Q.current_time - server.busy_time
    return e_parameter * served_inds + (1 - e_parameter) * idle_times


def utility_function_2(Q, server, e_parameter=0.5):
    """
    Utility function 2 is the weighted average of the following:
        - The proportion of individuals served
        - The proportion of time the server was idle
    """
    served_inds_prop = len(server.served_inds) / len(Q.nodes[-1].all_individuals)
    idle_proportion = (Q.current_time - server.busy_time) / Q.current_time
    return e_parameter * served_inds_prop + (1 - e_parameter) * idle_proportion


def utility_function_3(Q, server, e_parameter=0.5):
    """
    Utility function 3 is the weighted average of the following:
        - The mean service time
        - The proportion of time the server was idle
    """
    mean_service_time = np.mean(server.service_times)
    idle_time = (Q.current_time - server.busy_time) / Q.current_time
    return e_parameter * mean_service_time + (1 - e_parameter) * idle_time


def utility_function_4(Q, server, e_parameter=0.5):
    """
    Utility function 4 is the weighted average of the following:
        - The mean service rate
        - The proportion of time the server was idle
    """
    mean_service_rate = 1 / np.mean(server.service_times)
    idle_time = (Q.current_time - server.busy_time) / Q.current_time
    return e_parameter * mean_service_rate + (1 - e_parameter) * idle_time


def utility_function_5(Q, server, e_parameter=0.5):
    """
    Utility function 5 is the weighted average of the following
        - The proportion of individuals served
        - The mean service time
    """
    served_inds_prop = len(server.served_inds) / len(Q.nodes[-1].all_individuals)
    mean_service_time = np.mean(server.service_times)
    return e_parameter * served_inds_prop + (1 - e_parameter) * mean_service_time


def utility_function_6(Q, server, e_parameter=0.5):
    """
    Utility function 6 is the weighted average of the following
        - The proportion of individuals served
        - The mean service rate
    """
    served_inds_prop = len(server.served_inds) / len(Q.nodes[-1].all_individuals)
    mean_service_rate = 1 / np.mean(server.service_times)
    return e_parameter * served_inds_prop + (1 - e_parameter) * mean_service_rate


def get_utility_values(utility_function, Q, e_parameter=0.5):
    """
    Returns a list of utility values for each server in the queue
    """
    all_servers = Q.nodes[2].servers
    all_utilities = [utility_function(Q, server, e_parameter) for server in all_servers]
    return all_utilities
