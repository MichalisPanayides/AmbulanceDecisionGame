import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import nashpy as nash

import ambulance_game as abg


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
    plot_variable=0,
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
    plot_variable : int, optional
        to determine what performance measure to be plotted:
            (0): blocking times
            (1): acceptance probability
            (2): lost probability
            (3): (0) * (1)
            (4): (0) * (2)
            (5): (0) * [ 1 + (1)]
            (6): (0) * [ 1 + (2)]

    Returns
    -------
    plot
        The plot of blocking times of 2 systems over different arrival
        distributions of individuals
    """

    hospital_times_1 = []
    hospital_times_2 = []
    all_arrival_rates = np.linspace(0, lambda_2, accuracy + 1)
    for lambda_2_1 in all_arrival_rates[1:-1]:
        lambda_2_2 = lambda_2 - lambda_2_1
        times_1 = abg.markov.get_mean_blocking_time_using_markov_state_probabilities(
            lambda_2=lambda_2_1,
            lambda_1=lambda_1_1,
            mu=mu_1,
            num_of_servers=num_of_servers_1,
            threshold=threshold_1,
            system_capacity=system_capacity_1,
            buffer_capacity=buffer_capacity_1,
        )
        times_2 = abg.markov.get_mean_blocking_time_using_markov_state_probabilities(
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

        if plot_variable == 0:
            pass
        elif plot_variable == 1:
            times_1 = prob_accept_1
            times_2 = prob_accept_2
        elif plot_variable == 2:
            times_1 = 1 - prob_accept_1
            times_2 = 1 - prob_accept_2
        elif plot_variable == 3:
            times_1 = times_1 * prob_accept_1
            times_2 = times_2 * prob_accept_2
        elif plot_variable == 4:
            times_1 = times_1 * (1 - prob_accept_1)
            times_2 = times_2 * (1 - prob_accept_2)
        elif plot_variable == 5:
            times_1 = times_1 * (1 + prob_accept_1)
            times_2 = times_2 * (1 + prob_accept_2)
        elif plot_variable == 6:
            times_1 = times_1 * (2 - prob_accept_1)
            times_2 = times_2 * (2 - prob_accept_2)

        hospital_times_1.append(times_1)
        hospital_times_2.append(times_2)

    x_labels = all_arrival_rates[1:-1] / all_arrival_rates[-1]
    plt.figure(figsize=(23, 10))
    distribution_plot = plt.plot(x_labels, hospital_times_1, ls="solid", lw=1.5)
    plt.plot(x_labels, hospital_times_2, ls="solid", lw=1.5)
    plt.legend(["Hospital 1", "Hospital 2"], fontsize="x-large")

    title = "Individuals distribution between two systems"
    if plot_variable == 1:
        y_axis_label = "Acceptance Proportion"
    elif plot_variable == 2:
        y_axis_label = "Lost proportion"
    elif plot_variable == 3:
        y_axis_label = "$B_i \\times P(A_i)$"
    elif plot_variable == 4:
        y_axis_label = "$B_i \\times P(L_i)$"
    elif plot_variable == 5:
        y_axis_label = "$B_i \\times (1 + P(L_i))$"
    elif plot_variable == 6:
        y_axis_label = "$B_i \\times (1 + P(L_i))+$"
    else:
        y_axis_label = "Blocking times"

    plt.title(
        title + "($T_1$=" + str(threshold_1) + ", $T_2$=" + str(threshold_2) + ")",
        fontsize=18,
    )
    plt.ylabel(y_axis_label, fontsize=15, fontweight="bold")
    plt.xlabel(f"$p_1$", fontsize=15, fontweight="bold")

    return distribution_plot


def get_mean_blocking_difference_between_two_markov_systems(
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
):
    """Get the mean blocking difference between two Markov systems. This function
    is to be used as a decision function to find the point at which it reaches 0.

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

    mean_blocking_1 = (
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
    mean_blocking_2 = (
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
    return mean_blocking_1 - mean_blocking_2


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
    alpha=0.6,
):
    """
    Get a weighted mean blocking difference between two Markov systems. This
    function is to be used as an optimisation function to find the point at
    which it is set to 0. This function calculates c_1 * B_1 - c_2 * B_2, where
    c_1, c_2 are the coefficients that represent the proportion of lost
    individuals

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
    optimisation_function=get_weighted_mean_blocking_difference_between_two_markov_systems,
):
    """
    Get the best distribution of individuals (i.e. p_1, p_2) such that the
    the optimisation function given is 0.

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
    optimisation_function : function, optional
        The function to find the root of

    Returns
    -------
    float
        The value of p_1 such that optimisation_function = 0
    """
    check_1 = optimisation_function(
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
    )
    check_2 = optimisation_function(
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
    )

    if check_1 >= 0 and check_2 >= 0:
        return 0
    if check_1 <= 0 and check_2 <= 0:
        return 1

    optimal_prop = scipy.optimize.brentq(
        optimisation_function,
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
        ),
    )
    return optimal_prop


def get_optimal_distribution_matrix(
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
    optimisation_function=get_weighted_mean_blocking_difference_between_two_markov_systems,
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
    optimisation_function : function, optional
        The function to use to get the optimal distribution of patients

    Returns
    -------
    numpy array
        The matrix with proportions of all possible combinations of threshold
    """
    distribution_array = np.zeros((system_capacity_1, system_capacity_2))
    for threshold_1 in range(1, system_capacity_1 + 1):
        for threshold_2 in range(1, system_capacity_2 + 1):
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
                optimisation_function=optimisation_function,
            )
            distribution_array[threshold_1 - 1, threshold_2 - 1] = opt
    return distribution_array


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
    distribution_array=None,
    optimisation_function=get_weighted_mean_blocking_difference_between_two_markov_systems,
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
    distribution_array : numpy.array, optional
        The array that defines the class 2 distribution split. If None is given
        the function calculates it from start.
    optimisation_function : function, optional
        The function to use to get the optimal distribution of patients, if the
        value of distribution_array is none

    Returns
    -------
    numpy.array, numpy.array
        The payoff matrices of the game
    """
    if distribution_array is None:
        distribution_array = get_optimal_distribution_matrix(
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
            optimisation_function=optimisation_function,
        )
    utility_matrix_1 = np.zeros((system_capacity_1, system_capacity_2))
    utility_matrix_2 = np.zeros((system_capacity_1, system_capacity_2))
    for threshold_1 in range(1, system_capacity_1 + 1):
        for threshold_2 in range(1, system_capacity_2 + 1):
            p1 = distribution_array[threshold_1 - 1, threshold_2 - 1]
            p2 = 1 - p1
            prop_1 = (
                abg.markov.proportion_within_target_using_markov_state_probabilities(
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
            )
            prop_2 = (
                abg.markov.proportion_within_target_using_markov_state_probabilities(
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
            )
            utility_matrix_1[threshold_1 - 1, threshold_2 - 1] = -((prop_1 - 0.95) ** 2)
            utility_matrix_2[threshold_1 - 1, threshold_2 - 1] = -((prop_2 - 0.95) ** 2)

    return (utility_matrix_1, utility_matrix_2)


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
            distribution_array=None,
            optimisation_function=get_weighted_mean_blocking_difference_between_two_markov_systems,
        )

    game = nash.Game(payoff_matrix_A, payoff_matrix_B)
    return game
