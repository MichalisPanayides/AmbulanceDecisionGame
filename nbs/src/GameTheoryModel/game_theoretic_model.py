import itertools
import functools
import random

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nashpy as nash
import numpy as np
import pandas as pd
import scipy.optimize

import ambulance_game as abg


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
    transition_matrix = abg.markov.get_transition_matrix(
        lambda_2=lambda_2,
        lambda_1=lambda_1,
        mu=mu,
        num_of_servers=num_of_servers,
        threshold=threshold,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
    )
    all_states = abg.markov.build_states(
        threshold=threshold,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
    )
    pi = abg.markov.get_steady_state_algebraically(
        Q=transition_matrix, algebraic_function=np.linalg.solve
    )
    pi = abg.markov.get_markov_state_probabilities(pi, all_states, output=np.ndarray)

    prob_accept = abg.markov.get_probability_of_accepting(
        all_states=all_states,
        pi=pi,
        threshold=threshold,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
    )
    return prob_accept[1]


def make_plot_of_distribution_among_two_systems(
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
    accuracy=10,
    alpha=0,
):
    """
    Given two distinct systems and a joint value for lambda_2, plot the blocking
    times of the two systems by altering the value of the proportion of people
    that go to each hospital.

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
    accuracy : int, optional
    alpha : float, optional

    Returns
    -------
    plot
        The plot of blocking times of 2 systems over different arrival
        distributions of individuals
    """

    system_times_1 = []
    system_times_2 = []
    all_arrival_rates = np.linspace(0, lambda_2, accuracy + 1)
    for lambda_2_1 in all_arrival_rates[1:-1]:
        lambda_2_2 = lambda_2 - lambda_2_1
        blocking_times_1 = (
            abg.markov.get_mean_blocking_time_using_markov_state_probabilities(
                lambda_2=lambda_2_1,
                lambda_1=lambda_1_1,
                mu=mu_1,
                num_of_servers=num_of_servers_1,
                threshold=threshold_1,
                system_capacity=system_capacity_1,
                buffer_capacity=buffer_capacity_1,
            )
        )
        blocking_times_2 = (
            abg.markov.get_mean_blocking_time_using_markov_state_probabilities(
                lambda_2=lambda_2_2,
                lambda_1=lambda_1_2,
                mu=mu_2,
                num_of_servers=num_of_servers_2,
                threshold=threshold_2,
                system_capacity=system_capacity_2,
                buffer_capacity=buffer_capacity_2,
            )
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

        system_times_1.append(
            alpha * (1 - prob_accept_1) + (1 - alpha) * blocking_times_1
        )
        system_times_2.append(
            alpha * (1 - prob_accept_2) + (1 - alpha) * blocking_times_2
        )

    x_labels = all_arrival_rates[1:-1] / all_arrival_rates[-1]
    plt.figure(figsize=(23, 10))
    distribution_plot = plt.plot(x_labels, system_times_1, ls="solid", lw=1.5)
    plt.plot(x_labels, system_times_2, ls="solid", lw=1.5)
    plt.legend(["System 1", "System 2"], fontsize="x-large")

    title = "Individuals distribution between two systems"
    y_axis_label = "$\\alpha P(L_i) + (1 - \\alpha) B_i $"

    plt.title(
        title
        + "($T_1$="
        + str(threshold_1)
        + ", $T_2$="
        + str(threshold_2)
        + ", $\\alpha$="
        + str(alpha)
        + ")",
        fontsize=18,
    )
    plt.ylabel(y_axis_label, fontsize=15, fontweight="bold")
    plt.xlabel(f"$p_1$", fontsize=15, fontweight="bold")

    return distribution_plot


@functools.lru_cache(maxsize=None)
def get_weighted_mean_blocking_difference_between_two_markov_systems(
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
    Get a weighted mean blocking difference between two Markov systems. This
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

    mean_blocking_time_1 = (
        abg.markov.get_mean_blocking_time_using_markov_state_probabilities(
            lambda_2=lambda_2_1,
            lambda_1=lambda_1_1,
            mu=mu_1,
            num_of_servers=num_of_servers_1,
            threshold=threshold_1,
            system_capacity=system_capacity_1,
            buffer_capacity=buffer_capacity_1,
        )
    )
    mean_blocking_time_2 = (
        abg.markov.get_mean_blocking_time_using_markov_state_probabilities(
            lambda_2=lambda_2_2,
            lambda_1=lambda_1_2,
            mu=mu_2,
            num_of_servers=num_of_servers_2,
            threshold=threshold_2,
            system_capacity=system_capacity_2,
            buffer_capacity=buffer_capacity_2,
        )
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
def calculate_class_2_individuals_best_response_markov(
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
    routing_function=get_weighted_mean_blocking_difference_between_two_markov_systems,
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
    routing_function=get_weighted_mean_blocking_difference_between_two_markov_systems,
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
        opt = calculate_class_2_individuals_best_response_markov(
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


@functools.lru_cache(maxsize=None)
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
    routing_matrix=None,
    routing_function=get_weighted_mean_blocking_difference_between_two_markov_systems,
    alpha=0,
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
    if routing_matrix is None:
        routing_matrix = get_routing_matrix(
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
            routing_function=routing_function,
            alpha=alpha,
        )
    utility_matrix_1 = np.zeros((system_capacity_1, system_capacity_2))
    utility_matrix_2 = np.zeros((system_capacity_1, system_capacity_2))
    for threshold_1, threshold_2 in itertools.product(
        range(1, system_capacity_1 + 1), range(1, system_capacity_2 + 1)
    ):
        p1 = routing_matrix[threshold_1 - 1, threshold_2 - 1]
        p2 = 1 - p1
        prop_1 = abg.markov.proportion_within_target_using_markov_state_probabilities(
            lambda_2=lambda_2 * p1,
            lambda_1=lambda_1_1,
            mu=mu_1,
            num_of_servers=num_of_servers_1,
            threshold=threshold_1,
            system_capacity=system_capacity_1,
            buffer_capacity=buffer_capacity_1,
            class_type=None,
            target=target,
        )
        prop_2 = abg.markov.proportion_within_target_using_markov_state_probabilities(
            lambda_2=lambda_2 * p2,
            lambda_1=lambda_1_2,
            mu=mu_2,
            num_of_servers=num_of_servers_2,
            threshold=threshold_2,
            system_capacity=system_capacity_2,
            buffer_capacity=buffer_capacity_2,
            class_type=None,
            target=target,
        )
        if alternative_utility:
            u_1 = prop_1
            u_2 = prop_2
        else:
            u_1 = -((prop_1 - 0.95) ** 2)
            u_2 = -((prop_2 - 0.95) ** 2)

        utility_matrix_1[threshold_1 - 1, threshold_2 - 1] = u_1
        utility_matrix_2[threshold_1 - 1, threshold_2 - 1] = u_2

    return (utility_matrix_1, utility_matrix_2)


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
        payoff_matrix_A, payoff_matrix_B = get_payoff_matrices(
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
            alternative_utility=alternative_utility,
            routing_matrix=None,
            routing_function=get_weighted_mean_blocking_difference_between_two_markov_systems,
            alpha=alpha,
        )

    game = nash.Game(payoff_matrix_A, payoff_matrix_B)
    return game


def make_fictitious_play_plot(game, iterations=20, seed=None, play_counts_start=None):
    np.random.seed(seed)
    all_play_counts = tuple(
        game.fictitious_play(iterations=iterations, play_counts=play_counts_start)
    )
    plt.figure(figsize=(14, 10))
    probabilities = [
        row_play_counts / np.sum(row_play_counts)
        for row_play_counts, _ in all_play_counts
    ]
    for number, strategy in enumerate(zip(*probabilities)):
        plt.plot(strategy, label=f"$s_{number}$")
    plt.xlabel("Iteration")
    plt.ylabel("Probability")
    plt.title("Actions taken by row player")
    plt.legend()
    return all_play_counts[-1]


def make_ficititious_play_plot_over_different_values_of_alpha(
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
    iterations=1000,
    width=0.1,
    seed=random.randint(1, 100),
):
    player_strategies_1 = np.array([], dtype=np.int64)
    player_strategies_2 = np.array([], dtype=np.int64)
    for alpha in np.linspace(start=0, stop=1, num=6):
        game = build_game_using_payoff_matrices(
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
        )
        np.random.seed(seed)
        play_counts = tuple(game.fictitious_play(iterations=iterations))
        probabilities = [
            player_strategies / np.sum(player_strategies)
            for player_strategies in play_counts[-1]
        ]

        player_strategies_1 = (
            np.vstack((player_strategies_1, probabilities[0]))
            if player_strategies_1.size
            else probabilities[0]
        )

        player_strategies_2 = (
            np.vstack((player_strategies_2, probabilities[1]))
            if player_strategies_2.size
            else probabilities[1]
        )

    ind = np.linspace(start=0, stop=1, num=6)

    plt.figure(figsize=(14, 10))
    bottom_bar = np.array(
        [0 for _ in range(len(player_strategies_1[:, 0]))], dtype=np.float64
    )
    for row_player_strategy in range(len(player_strategies_1[0])):
        plt.bar(
            ind,
            player_strategies_1[:, row_player_strategy],
            width,
            bottom=bottom_bar,
            label=f"$s_{{{row_player_strategy + 1}}}$",
        )
        bottom_bar += player_strategies_1[:, row_player_strategy]
    plt.title(
        "Fictitious play over different values of $\\alpha$ for the strategies of the row player"
    )
    plt.xlabel("$\\alpha$")
    plt.ylabel("Proportion of each strategy being played")
    plt.legend(loc=4, fontsize="x-large")

    plt.figure(figsize=(14, 10))
    bottom_bar = np.array(
        [0 for _ in range(len(player_strategies_2[:, 0]))], dtype=np.float64
    )
    for column_player_strategy in range(len(player_strategies_2[0])):
        plt.bar(
            ind,
            player_strategies_2[:, column_player_strategy],
            width,
            bottom=bottom_bar,
            label=f"$s_{{{column_player_strategy + 1}}}$",
        )
        bottom_bar += player_strategies_2[:, column_player_strategy]
    plt.title(
        "Fictitious play over different values of $\\alpha$ for the strategies of the row player"
    )
    plt.xlabel("$\\alpha$")
    plt.ylabel("Proportion of each strategy being played")
    plt.legend(loc=4, fontsize="x-large")

    return player_strategies_1, player_strategies_2


def get_brentq_tolerance_heatmaps(
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
    xtol_values,
    rtol_values,
):
    calculated_roots = np.zeros((len(rtol_values), len(xtol_values)))
    for (r_index, rtol), (x_index, xtol) in itertools.product(
        enumerate(rtol_values),
        enumerate(xtol_values),
    ):
        root = calculate_class_2_individuals_best_response_markov(
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
            xtol=xtol,
            rtol=rtol,
        )
        calculated_roots[r_index, x_index] = root
    plt.figure(figsize=(20, 10))
    plt.imshow(calculated_roots)
    plt.title("Heatmap of rtol VS xtol VS calculated root")
    plt.ylabel("rtol values")
    plt.xlabel("xtol values")
    plt.colorbar()

    return calculated_roots


def make_calculated_roots_over_alpha_plot(
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
    alpha_points=11,
):
    calculated_roots = np.zeros(alpha_points)
    alpha_range = np.linspace(0, 1, alpha_points)
    for ind, alpha in enumerate(alpha_range):
        root = calculate_class_2_individuals_best_response_markov(
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
        calculated_roots[ind] = root
    plt.figure(figsize=(15, 10))
    plt.plot(alpha_range, calculated_roots)
    plt.title("Calculated root over different values of $\\alpha$")
    plt.ylabel("Calculated root")
    plt.xlabel("$\\alpha$")

    return calculated_roots


def make_brentq_heatmap_of_time_vs_xtol_vs_capacity(file_path="main.csv"):
    df = pd.read_csv(file_path)
    mean_time_df = pd.DataFrame(
        np.zeros((19, 10)), columns=list(np.logspace(-10, -1, 10))
    )
    for index, row in df.iterrows():
        mean_time_df[row["tolerance"]][int(row["system_capacity_1"] - 8)] += row[
            "time_taken"
        ]
    mean_time_df /= 200
    plt.figure(figsize=(23, 10))
    plt.title("Heatmap of xtol VS $C_1$ VS time")
    plt.xlabel("xtol values")
    plt.ylabel("$C_1$")
    plt.imshow(mean_time_df[:-1])
    plt.colorbar()


def make_violinplots_of_fictitious_play(
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
    alpha=0.5,
    iterations=100,
    seed_start=0,
    seed_reps=30,
    num_of_violiplots=8,
    violin_width=10,
):
    seed_range = np.linspace(seed_start, seed_start + 10000, seed_reps, dtype=int)
    violinplots_data_pos = np.linspace(1, iterations, num_of_violiplots, dtype=int)

    game = build_game_using_payoff_matrices(
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
    )

    all_violinplots_data_row = None
    for seed in seed_range:
        np.random.seed(seed)
        play_counts = tuple(game.fictitious_play(iterations=iterations))
        current_violinplot_data_row_player = None
        for pos in violinplots_data_pos:
            row_plays, col_plays = play_counts[pos]
            if current_violinplot_data_row_player is None:
                current_violinplot_data_row_player = np.array([row_plays])
            else:
                current_violinplot_data_row_player = np.concatenate(
                    (current_violinplot_data_row_player, np.array([row_plays]))
                )

        if all_violinplots_data_row is None:
            all_violinplots_data_row = [current_violinplot_data_row_player]
        else:
            all_violinplots_data_row = np.concatenate(
                (
                    all_violinplots_data_row,
                    [current_violinplot_data_row_player],
                )
            )
    row_player_strategies = all_violinplots_data_row.shape[2]
    plt.figure(figsize=(20, 10))
    row_labels = []
    for row_strategy in range(row_player_strategies):
        violin = plt.violinplot(
            all_violinplots_data_row[:, :, row_strategy],
            violinplots_data_pos,
            widths=violin_width,
        )
        color = violin["bodies"][0].get_facecolor().flatten()
        row_labels.append((mpatches.Patch(color=color), f"$s_{{{row_strategy + 1}}}$"))

    plt.xlabel("Iteration")
    plt.ylabel("Times played")
    plt.legend(*zip(*row_labels), fontsize="x-large")
    return all_violinplots_data_row
