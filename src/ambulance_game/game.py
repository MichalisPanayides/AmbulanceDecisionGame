"""
Code for the game between two queueing systems and a distributor
"""

import functools
import itertools

import dask as da
import nashpy as nash
import numpy as np
import scipy.optimize

from .markov import get_mean_blocking_difference_using_markov
from .markov import proportion_within_target_using_markov_state_probabilities
from .simulation import (
    get_mean_blocking_difference_using_simulation,
    get_mean_proportion_of_individuals_within_target_for_multiple_runs,
)


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
    use_simulation=False,
    runtime=1440,
    num_of_trials=10,
    warm_up_time=100,
    seed_num_1=None,
    seed_num_2=None,
    lower_bound=0.01,
    upper_bound=0.99,
    alpha=0,
    xtol=1e-04,
    rtol=8.9e-16,
):
    """
    Obtains the optimal distribution of class 2 individuals such that the
    blocking times in the two systems are identical and thus optimal(minimised).

    The brentq function is used which is an algorithm created to find the root of
    a function that combines root bracketing, bisection, and inverse quadratic
    interpolation. In this specific example the root to be found is the difference
    between the blocking times of two systems. In essence the brentq algorithm
    attempts to find the value of "prop_1" where the "diff" is zero.

    Parameters
    ----------
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

    if use_simulation:
        routing_function = get_mean_blocking_difference_using_simulation
    else:
        routing_function = get_mean_blocking_difference_using_markov

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
        runtime=runtime,
        num_of_trials=num_of_trials,
        warm_up_time=warm_up_time,
        seed_num_1=seed_num_1,
        seed_num_2=seed_num_2,
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
        runtime=runtime,
        num_of_trials=num_of_trials,
        warm_up_time=warm_up_time,
        seed_num_1=seed_num_1,
        seed_num_2=seed_num_2,
    )

    if check_1 >= 0 and check_2 >= 0:
        return 0
    if check_1 <= 0 and check_2 <= 0:
        return 1

    brentq_arguments = (
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
    )
    if use_simulation:
        brentq_arguments += (
            seed_num_1,
            seed_num_2,
            num_of_trials,
            warm_up_time,
            runtime,
        )
    optimal_prop = scipy.optimize.brentq(
        routing_function,
        a=lower_bound,
        b=upper_bound,
        args=brentq_arguments,
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
    alpha=0,
    use_simulation=False,
    runtime=1440,
    num_of_trials=10,
    warm_up_time=100,
    seed_num_1=None,
    seed_num_2=None,
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
            alpha=alpha,
            use_simulation=use_simulation,
            runtime=runtime,
            num_of_trials=num_of_trials,
            warm_up_time=warm_up_time,
            seed_num_1=seed_num_1,
            seed_num_2=seed_num_2,
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
    p_hat=0.95,
    alternative_utility=False,
    use_simulation=False,
    runtime=1440,
    num_of_trials=10,
    warm_up_time=100,
    seed_num_1=None,
    seed_num_2=None,
    use_cache=True,
):
    """
    Gets the (i,j)th entry of the payoff matrices and the routing matrix where
    i=threshold_1 and j=threshold_2. The calculated utility function is given
    by: U_i = - (P(X < target) - p_hat) ** 2)
      where `P(X < target)` is the number of individuals within `target`
            `p_hat` is the proportion of individuals that must be within `target`
    This function is wrapped by the dask.delayed decorator and returns the
    output of the function as a dask task.

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
    p_hat : float, optional
    alternative_utility : bool, optional

    Returns
    -------
    tuple
        A tuple of the form (i, j, R[i,j], A[i,j], B[i,j])
    """
    if use_cache:
        best_response_function = calculate_class_2_individuals_best_response
    else:
        best_response_function = calculate_class_2_individuals_best_response.__wrapped__

    prop_to_hospital_1 = best_response_function(
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
        use_simulation=use_simulation,
        runtime=runtime,
        num_of_trials=num_of_trials,
        warm_up_time=warm_up_time,
        seed_num_1=seed_num_1,
        seed_num_2=seed_num_2,
    )
    prop_to_hospital_2 = 1 - prop_to_hospital_1

    if use_simulation:
        proportion_within_target_1 = (
            get_mean_proportion_of_individuals_within_target_for_multiple_runs(
                lambda_2=lambda_2 * prop_to_hospital_1,
                lambda_1=lambda_1_1,
                mu=mu_1,
                num_of_servers=num_of_servers_1,
                threshold=threshold_1,
                system_capacity=system_capacity_1,
                buffer_capacity=buffer_capacity_1,
                target=target,
                runtime=runtime,
                num_of_trials=num_of_trials,
                seed_num=seed_num_1,
            )[0]
        )
        proportion_within_target_2 = (
            get_mean_proportion_of_individuals_within_target_for_multiple_runs(
                lambda_2=lambda_2 * prop_to_hospital_2,
                lambda_1=lambda_1_2,
                mu=mu_2,
                num_of_servers=num_of_servers_2,
                threshold=threshold_2,
                system_capacity=system_capacity_2,
                buffer_capacity=buffer_capacity_2,
                target=target,
                runtime=runtime,
                num_of_trials=num_of_trials,
                seed_num=seed_num_2,
            )[0]
        )
    else:
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
        utility_1 = 1 - ((np.nanmean(proportion_within_target_1) - p_hat) ** 2)
        utility_2 = 1 - ((np.nanmean(proportion_within_target_2) - p_hat) ** 2)

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


def build_matrices_from_computed_tasks(computed_tasks, n_1, n_2):
    """
    Using the computed tasks builds the utility matrix of the row and the column
    players and the routing matrix.

    Parameters
    ----------
    computed_tasks : tuple
        A tuple of tuples of the form (i, j, R[i,j], A[i,j], B[i,j])
    n_1 : int
        The number of rows for all matrices
    n_2 : int
        The number of columns for all matrices

    Returns
    -------
    numpy array, numpy array, numpy array
        The routing matrix and the two payoff matrices
    """
    routing_matrix = np.zeros((n_1, n_2))
    utility_matrix_1 = np.zeros((n_1, n_2))
    utility_matrix_2 = np.zeros((n_1, n_2))

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
    p_hat=0.95,
    processes=None,
    use_simulation=False,
    runtime=1440,
    num_of_trials=10,
    warm_up_time=100,
    seed_num_1=None,
    seed_num_2=None,
    use_cache=True,
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
            p_hat=p_hat,
            alternative_utility=alternative_utility,
            use_simulation=use_simulation,
            runtime=runtime,
            num_of_trials=num_of_trials,
            warm_up_time=warm_up_time,
            seed_num_1=seed_num_1,
            seed_num_2=seed_num_2,
            use_cache=use_cache,
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
        computed_tasks=computed_tasks, n_1=system_capacity_1, n_2=system_capacity_2
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
    alpha=0,
    p_hat=0.95,
    payoff_matrix_A=None,
    payoff_matrix_B=None,
    alternative_utility=False,
    use_simulation=False,
    runtime=1440,
    num_of_trials=10,
    warm_up_time=100,
    seed_num_1=None,
    seed_num_2=None,
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
    if payoff_matrix_A is None or payoff_matrix_B is None:
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
            p_hat=p_hat,
            alternative_utility=alternative_utility,
            use_simulation=use_simulation,
            runtime=runtime,
            num_of_trials=num_of_trials,
            warm_up_time=warm_up_time,
            seed_num_1=seed_num_1,
            seed_num_2=seed_num_2,
        )

    game = nash.Game(payoff_matrix_A, payoff_matrix_B)
    return game
