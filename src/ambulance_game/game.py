import functools
import itertools

import dask as da
import nashpy as nash
import numpy as np
from numpy.core.fromnumeric import shape
import scipy.optimize

from .markov.blocking import get_mean_blocking_time_using_markov_state_probabilities
from .markov.markov import (
    build_states,
    get_markov_state_probabilities,
    get_steady_state_algebraically,
    get_transition_matrix,
)
from .markov.proportion import proportion_within_target_using_markov_state_probabilities
from .markov.utils import get_probability_of_accepting


@functools.lru_cache(maxsize=None)
def get_accepting_proportion_of_class_2_individuals(
    lambda_1, lambda_2, mu, num_of_servers, threshold, system_capacity, buffer_capacity
):
    """
    Get the proportion of class 2 individuals that are not lost to the system

    Parameters
    ----------
    lambda_1 : float
    lambda_2 : float
    mu : float
    num_of_servers : int
    threshold : int
    system_capacity : int
    buffer_capacity : int

    Returns
    -------
    float
        The probability that an individual entering will not be lost to the
        system
    """
    transition_matrix = get_transition_matrix(
        lambda_2=lambda_2,
        lambda_1=lambda_1,
        mu=mu,
        num_of_servers=num_of_servers,
        threshold=threshold,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
    )
    all_states = build_states(
        threshold=threshold,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
    )
    pi = get_steady_state_algebraically(
        Q=transition_matrix, algebraic_function=np.linalg.solve
    )
    pi = get_markov_state_probabilities(pi, all_states, output=np.ndarray)

    prob_accept = get_probability_of_accepting(
        all_states=all_states,
        pi=pi,
        threshold=threshold,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
    )
    return prob_accept[1]


@functools.lru_cache(maxsize=None)
def get_weighted_mean_blocking_difference_between_two_systems(
    prop_1,
    lambda_2,
    lambda_1_1,
    lambda_1_2,
    mu_1,
    mu_2,
    num_of_servers_1,
    num_of_servers_2,
    threshold_1,
    threshold_2,
    system_capacity_1,
    system_capacity_2,
    buffer_capacity_1,
    buffer_capacity_2,
    alpha=0,
):
    """
    Get a weighted mean blocking difference between two systems. This
    function is to be used as a routing function to find the point at
    which it is set to 0. This function calculates:
    a*(1 - P(A_1)) + (1 - a)*B_1 = a*(1 - P(A_2)) + (1 - a)*B_2

    Parameters
    ----------
    prop_1 : float
        The proportion of class 2 individuals to distribute to the first system
    lambda_2 : float
        The overall arrival rate of class 2 individuals for both systems
    lambda_1_1 : float
        The arrival rate of class 1 individuals in the first system
    lambda_1_2 : float
        The arrival rate of class 1 individuals in the second system
    mu_1 : float
    mu_2 : float
    num_of_servers_1 : int
    num_of_servers_2 : int
    threshold_1 : int
    threshold_2 : int
    system_capacity_1 : int
    system_capacity_2 : int
    buffer_capacity_1 : int
    buffer_capacity_2 : int

    Returns
    -------
    float
        The mean blocking difference B_1 - B_2
    """
    lambda_2_1 = prop_1 * lambda_2
    lambda_2_2 = (1 - prop_1) * lambda_2

    mean_blocking_time_1 = get_mean_blocking_time_using_markov_state_probabilities(
        lambda_2=lambda_2_1,
        lambda_1=lambda_1_1,
        mu=mu_1,
        num_of_servers=num_of_servers_1,
        threshold=threshold_1,
        system_capacity=system_capacity_1,
        buffer_capacity=buffer_capacity_1,
    )
    mean_blocking_time_2 = get_mean_blocking_time_using_markov_state_probabilities(
        lambda_2=lambda_2_2,
        lambda_1=lambda_1_2,
        mu=mu_2,
        num_of_servers=num_of_servers_2,
        threshold=threshold_2,
        system_capacity=system_capacity_2,
        buffer_capacity=buffer_capacity_2,
    )
    prob_accept_1 = get_accepting_proportion_of_class_2_individuals(
        lambda_1=lambda_1_1,
        lambda_2=lambda_2_1,
        mu=mu_1,
        num_of_servers=num_of_servers_1,
        threshold=threshold_1,
        system_capacity=system_capacity_1,
        buffer_capacity=buffer_capacity_1,
    )
    prob_accept_2 = get_accepting_proportion_of_class_2_individuals(
        lambda_1=lambda_1_2,
        lambda_2=lambda_2_2,
        mu=mu_2,
        num_of_servers=num_of_servers_2,
        threshold=threshold_2,
        system_capacity=system_capacity_2,
        buffer_capacity=buffer_capacity_2,
    )

    decision_value_1 = alpha * (1 - prob_accept_1) + (1 - alpha) * mean_blocking_time_1

    decision_value_2 = alpha * (1 - prob_accept_2) + (1 - alpha) * mean_blocking_time_2

    return decision_value_1 - decision_value_2


@functools.lru_cache(maxsize=None)
def calculate_class_2_individuals_best_response(
    lambda_2,
    lambda_1_1,
    lambda_1_2,
    mu_1,
    mu_2,
    num_of_servers_1,
    num_of_servers_2,
    threshold_1,
    threshold_2,
    system_capacity_1,
    system_capacity_2,
    buffer_capacity_1,
    buffer_capacity_2,
    lower_bound=0.01,
    upper_bound=0.99,
    routing_function=get_weighted_mean_blocking_difference_between_two_systems,
    alpha=0,
    xtol=1e-04,
    rtol=8.9e-16,
):
    """
    Get the best distribution of individuals (i.e. p_1, p_2) such that the
    the routing function given is 0.

    Parameters
    ----------
    lambda_2 : float
    lambda_1_1 : float
    lambda_1_2 : float
    mu_1 : float
    mu_2 : float
    num_of_servers_1 : float
    num_of_servers_2 : float
    threshold_1 : float
    threshold_2 : float
    system_capacity_1 : float
    system_capacity_2 : float
    buffer_capacity_1 : float
    buffer_capacity_2 : float
    lower_bound : float, optional
        The lower bound of p_1, by default 0.01
    upper_bound : float, optional
        The upper bound of p_1, by default 0.99
    routing_function : function, optional
        The function to find the root of

    Returns
    -------
    float
        The value of p_1 such that routing_function = 0
    """
    check_1 = routing_function(
        prop_1=lower_bound,
        lambda_2=lambda_2,
        lambda_1_1=lambda_1_1,
        lambda_1_2=lambda_1_2,
        mu_1=mu_1,
        mu_2=mu_2,
        num_of_servers_1=num_of_servers_1,
        num_of_servers_2=num_of_servers_2,
        threshold_1=threshold_1,
        threshold_2=threshold_2,
        system_capacity_1=system_capacity_1,
        system_capacity_2=system_capacity_2,
        buffer_capacity_1=buffer_capacity_1,
        buffer_capacity_2=buffer_capacity_2,
        alpha=alpha,
    )
    check_2 = routing_function(
        prop_1=upper_bound,
        lambda_2=lambda_2,
        lambda_1_1=lambda_1_1,
        lambda_1_2=lambda_1_2,
        mu_1=mu_1,
        mu_2=mu_2,
        num_of_servers_1=num_of_servers_1,
        num_of_servers_2=num_of_servers_2,
        threshold_1=threshold_1,
        threshold_2=threshold_2,
        system_capacity_1=system_capacity_1,
        system_capacity_2=system_capacity_2,
        buffer_capacity_1=buffer_capacity_1,
        buffer_capacity_2=buffer_capacity_2,
        alpha=alpha,
    )

    if check_1 >= 0 and check_2 >= 0:
        return 0
    if check_1 <= 0 and check_2 <= 0:
        return 1

    optimal_prop = scipy.optimize.brentq(
        routing_function,
        a=lower_bound,
        b=upper_bound,
        args=(
            lambda_2,
            lambda_1_1,
            lambda_1_2,
            mu_1,
            mu_2,
            num_of_servers_1,
            num_of_servers_2,
            threshold_1,
            threshold_2,
            system_capacity_1,
            system_capacity_2,
            buffer_capacity_1,
            buffer_capacity_2,
            alpha,
        ),
        xtol=xtol,
        rtol=rtol,
    )
    return optimal_prop


@functools.lru_cache(maxsize=None)
def get_routing_matrix(
    lambda_2,
    lambda_1_1,
    lambda_1_2,
    mu_1,
    mu_2,
    num_of_servers_1,
    num_of_servers_2,
    system_capacity_1,
    system_capacity_2,
    buffer_capacity_1,
    buffer_capacity_2,
    routing_function=get_weighted_mean_blocking_difference_between_two_systems,
    alpha=0,
):
    """
    Get the optimal distribution matrix that consists of the proportion of
    individuals to be distributed to each hospital for all possible
    combinations of thresholds of the two hospitals (T_1, T_2). For every set of
    thresholds, the function fills the entries of the matrix using the
    proportion of individuals to distribute to hospital 1.

    Parameters
    ----------
    lambda_2 : float
    lambda_1_1 : float
    lambda_1_2 : float
    mu_1 : float
    mu_2 : float
    num_of_servers_1 : int
    num_of_servers_2 : int
    system_capacity_1 : int
    system_capacity_2 : int
    buffer_capacity_1 : int
    buffer_capacity_2 : int
    routing_function : function, optional
        The function to use to get the optimal distribution of patients

    Returns
    -------
    numpy array
        The matrix with proportions of all possible combinations of threshold
    """
    routing_matrix = np.zeros((system_capacity_1, system_capacity_2))
    for threshold_1, threshold_2 in itertools.product(
        range(1, system_capacity_1 + 1), range(1, system_capacity_2 + 1)
    ):
        opt = calculate_class_2_individuals_best_response(
            lambda_2=lambda_2,
            lambda_1_1=lambda_1_1,
            lambda_1_2=lambda_1_2,
            mu_1=mu_1,
            mu_2=mu_2,
            num_of_servers_1=num_of_servers_1,
            num_of_servers_2=num_of_servers_2,
            system_capacity_1=system_capacity_1,
            system_capacity_2=system_capacity_2,
            buffer_capacity_1=buffer_capacity_1,
            buffer_capacity_2=buffer_capacity_2,
            threshold_1=threshold_1,
            threshold_2=threshold_2,
            routing_function=routing_function,
            alpha=alpha,
        )
        routing_matrix[threshold_1 - 1, threshold_2 - 1] = opt
    return routing_matrix


@da.delayed
def get_individual_entries_of_matrices(
    lambda_2,
    lambda_1_1,
    lambda_1_2,
    mu_1,
    mu_2,
    num_of_servers_1,
    num_of_servers_2,
    threshold_1,
    threshold_2,
    system_capacity_1,
    system_capacity_2,
    buffer_capacity_1,
    buffer_capacity_2,
    alpha,
    target,
    alternative_utility=False,
):
    """
    Gets the (i,j)th entry of the payoff matrices and the routing matrix where
    i=threshold_1 and j=threshold_2

    Parameters
    ----------
    lambda_2 : float
    lambda_1_1 : float
    lambda_1_2 : float
    mu_1 : float
    mu_2 : float
    num_of_servers_1 : int
    num_of_servers_2 : int
    threshold_1 : int
    threshold_2 : int
    system_capacity_1 : int
    system_capacity_2 : int
    buffer_capacity_1 : int
    buffer_capacity_2 : int
    alpha : float
    target : float
    alternative_utility : bool, optional

    Returns
    -------
    tuple
        A tuple of the form (i, j, R[i,j], A[i,j], B[i,j])
    """
    prop_to_hospital_1 = calculate_class_2_individuals_best_response(
        lambda_2=lambda_2,
        lambda_1_1=lambda_1_1,
        lambda_1_2=lambda_1_2,
        mu_1=mu_1,
        mu_2=mu_2,
        num_of_servers_1=num_of_servers_1,
        num_of_servers_2=num_of_servers_2,
        system_capacity_1=system_capacity_1,
        system_capacity_2=system_capacity_2,
        buffer_capacity_1=buffer_capacity_1,
        buffer_capacity_2=buffer_capacity_2,
        threshold_1=threshold_1,
        threshold_2=threshold_2,
        alpha=alpha,
    )
    prop_to_hospital_2 = 1 - prop_to_hospital_1

    proportion_within_target_1 = (
        proportion_within_target_using_markov_state_probabilities(
            lambda_2=lambda_2 * prop_to_hospital_1,
            lambda_1=lambda_1_1,
            mu=mu_1,
            num_of_servers=num_of_servers_1,
            threshold=threshold_1,
            system_capacity=system_capacity_1,
            buffer_capacity=buffer_capacity_1,
            class_type=None,
            target=target,
        )
    )
    proportion_within_target_2 = (
        proportion_within_target_using_markov_state_probabilities(
            lambda_2=lambda_2 * prop_to_hospital_2,
            lambda_1=lambda_1_2,
            mu=mu_2,
            num_of_servers=num_of_servers_2,
            threshold=threshold_2,
            system_capacity=system_capacity_2,
            buffer_capacity=buffer_capacity_2,
            class_type=None,
            target=target,
        )
    )
    if alternative_utility:
        utility_1 = proportion_within_target_1
        utility_2 = proportion_within_target_2
    else:
        utility_1 = -((proportion_within_target_1 - 0.95) ** 2)
        utility_2 = -((proportion_within_target_2 - 0.95) ** 2)

    return threshold_1, threshold_2, prop_to_hospital_1, utility_1, utility_2


def compute_tasks(tasks, processes):
    """
    Compute all dask tasks
    """
    if processes is None:
        out = da.compute(*tasks, scheduler="single-threaded")
    else:
        out = da.compute(*tasks, num_workers=processes)
    return out


def build_matrices_from_computed_tasks(computed_tasks, N_1, N_2):
    """
    Using the computed tasks builds the utility matrix of the row and the column
    players and the routing matrix.

    Parameters
    ----------
    computed_tasks : tuple
        A tuple of tuples of the form (i, j, R[i,j], A[i,j], B[i,j])
    N_1 : int
        The number of rows for all matrices
    N_2 : int
        The number of columns for all matrices

    Returns
    -------
    numpy array, numpy array, numpy array
        The routing matrix and the two payoff matrices
    """
    routing_matrix = np.zeros((N_1, N_2))
    utility_matrix_1 = np.zeros((N_1, N_2))
    utility_matrix_2 = np.zeros((N_1, N_2))

    for (
        threshold_1,
        threshold_2,
        routing_entry,
        utility_1_entry,
        utility_2_entry,
    ) in computed_tasks:
        row_index, col_index = threshold_1 - 1, threshold_2 - 1
        routing_matrix[row_index, col_index] = routing_entry
        utility_matrix_1[row_index, col_index] = utility_1_entry
        utility_matrix_2[row_index, col_index] = utility_2_entry

    return routing_matrix, utility_matrix_1, utility_matrix_2


def get_payoff_matrices(
    lambda_2,
    lambda_1_1,
    lambda_1_2,
    mu_1,
    mu_2,
    num_of_servers_1,
    num_of_servers_2,
    system_capacity_1,
    system_capacity_2,
    buffer_capacity_1,
    buffer_capacity_2,
    target,
    alternative_utility=False,
    alpha=0,
    processes=None,
):
    """
    The function uses the distribution array (that is the array that holds the
    optimal proportion of individuals to send to each hospital), to calculate
    the proportion of patients within time for every possible set of thresholds
    chosen by each system.

    Parameters
    ----------
    lambda_2 : float
    lambda_1_1 : float
    lambda_1_2 : float
    mu_1 : float
    mu_2 : float
    num_of_servers_1 : int
    num_of_servers_2 : int
    system_capacity_1 : int
    system_capacity_2 : int
    buffer_capacity_1 : int
    buffer_capacity_2 : int
    target : float
        The target time that individuals should be within
    alternative_utility : bool
        Use an alternative method to get the utilities by just using the
        probabilities that the target is less than 95%
    routing_matrix : numpy.array, optional
        The array that defines the class 2 distribution split. If None is given
        the function calculates it from start.
    routing_function : function, optional
        The function to use to get the optimal distribution of patients, if the
        value of routing_matrix is none

    Returns
    -------
    numpy.array, numpy.array
        The payoff matrices of the game
    """
    tasks = (
        get_individual_entries_of_matrices(
            lambda_2=lambda_2,
            lambda_1_1=lambda_1_1,
            lambda_1_2=lambda_1_2,
            mu_1=mu_1,
            mu_2=mu_2,
            num_of_servers_1=num_of_servers_1,
            num_of_servers_2=num_of_servers_2,
            threshold_1=threshold_1,
            threshold_2=threshold_2,
            system_capacity_1=system_capacity_1,
            system_capacity_2=system_capacity_2,
            buffer_capacity_1=buffer_capacity_1,
            buffer_capacity_2=buffer_capacity_2,
            alpha=alpha,
            target=target,
            alternative_utility=alternative_utility,
        )
        for threshold_1, threshold_2 in itertools.product(
            range(1, system_capacity_1 + 1), range(1, system_capacity_2 + 1)
        )
    )
    computed_tasks = compute_tasks(tasks=tasks, processes=processes)
    (
        routing_matrix,
        utility_matrix_1,
        utility_matrix_2,
    ) = build_matrices_from_computed_tasks(
        computed_tasks=computed_tasks, N_1=system_capacity_1, N_2=system_capacity_2
    )
    return utility_matrix_1, utility_matrix_2, routing_matrix


@functools.lru_cache(maxsize=None)
def build_game_using_payoff_matrices(
    lambda_2,
    lambda_1_1,
    lambda_1_2,
    mu_1,
    mu_2,
    num_of_servers_1,
    num_of_servers_2,
    system_capacity_1,
    system_capacity_2,
    buffer_capacity_1,
    buffer_capacity_2,
    target,
    payoff_matrix_A=None,
    payoff_matrix_B=None,
    alternative_utility=False,
    alpha=0,
):
    """
    Build the game theoretic model either by building the payoff matrices or by
    using the given ones by the user.

    Parameters
    ----------
    lambda_2 : float
    lambda_1_1 : float
    lambda_1_2 : float
    mu_1 : float
    mu_2 : float
    num_of_servers_1 : int
    num_of_servers_2 : int
    system_capacity_1 : int
    system_capacity_2 : int
    buffer_capacity_1 : int
    buffer_capacity_2 : int
    target : float
    payoff_matrix_A : numpy array, optional
    payoff_matrix_B : numpy array, optional

    Returns
    -------
    nashpy.Game
        the game with the constructed or given payoff matrices
    """
    if payoff_matrix_A == None or payoff_matrix_B == None:
        payoff_matrix_A, payoff_matrix_B, _ = get_payoff_matrices(
            lambda_2=lambda_2,
            lambda_1_1=lambda_1_1,
            lambda_1_2=lambda_1_2,
            mu_1=mu_1,
            mu_2=mu_2,
            num_of_servers_1=num_of_servers_1,
            num_of_servers_2=num_of_servers_2,
            system_capacity_1=system_capacity_1,
            system_capacity_2=system_capacity_2,
            buffer_capacity_1=buffer_capacity_1,
            buffer_capacity_2=buffer_capacity_2,
            target=target,
            alpha=alpha,
            alternative_utility=alternative_utility,
        )

    game = nash.Game(payoff_matrix_A, payoff_matrix_B)
    return game
