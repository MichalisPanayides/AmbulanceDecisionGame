import random

import dask as da
import numpy as np

from ambulance_game.game import (
    build_game_using_payoff_matrices,
    build_matrices_from_computed_tasks,
    calculate_class_2_individuals_best_response,
    compute_tasks,
    get_accepting_proportion_of_class_2_individuals,
    get_individual_entries_of_matrices,
    get_payoff_matrices,
    get_routing_matrix,
    get_weighted_mean_blocking_difference_between_two_systems,
)

number_of_digits_to_round = 8


def test_get_accepting_proportion_of_class_2_individuals_examples():
    """
    Test for getting the proportion of class 2 individuals
    """
    assert (
        get_accepting_proportion_of_class_2_individuals(
            lambda_1=1,
            lambda_2=1,
            mu=1,
            num_of_servers=1,
            threshold=1,
            system_capacity=1,
            buffer_capacity=1,
        )
        == 0.6
    )

    assert (
        round(
            get_accepting_proportion_of_class_2_individuals(
                lambda_1=2,
                lambda_2=2,
                mu=2,
                num_of_servers=2,
                threshold=2,
                system_capacity=2,
                buffer_capacity=2,
            ),
            number_of_digits_to_round,
        )
        == round(0.9230769230769231, number_of_digits_to_round)
    )

    assert (
        round(
            get_accepting_proportion_of_class_2_individuals(
                lambda_1=10,
                lambda_2=10,
                mu=2,
                num_of_servers=5,
                threshold=5,
                system_capacity=5,
                buffer_capacity=5,
            ),
            number_of_digits_to_round,
        )
        == round(0.8523592984113859, number_of_digits_to_round)
    )


def test_get_weighted_mean_blocking_difference_between_two_systems_example_1():
    """
    Test for getting the weighted mean blocking difference between two systems
    for different values of alpha
    """
    assert (
        round(
            get_weighted_mean_blocking_difference_between_two_systems(
                prop_1=0.5,
                lambda_2=2,
                lambda_1_1=2,
                lambda_1_2=2,
                mu_1=2,
                mu_2=2,
                num_of_servers_1=2,
                num_of_servers_2=2,
                threshold_1=3,
                threshold_2=5,
                system_capacity_1=5,
                system_capacity_2=5,
                buffer_capacity_1=5,
                buffer_capacity_2=4,
                alpha=0,
            ),
            number_of_digits_to_round,
        )
        == round(0.3786808388674136, number_of_digits_to_round)
    )

    assert (
        round(
            get_weighted_mean_blocking_difference_between_two_systems(
                prop_1=0.5,
                lambda_2=2,
                lambda_1_1=2,
                lambda_1_2=2,
                mu_1=2,
                mu_2=2,
                num_of_servers_1=2,
                num_of_servers_2=2,
                threshold_1=3,
                threshold_2=5,
                system_capacity_1=5,
                system_capacity_2=5,
                buffer_capacity_1=5,
                buffer_capacity_2=4,
                alpha=0.5,
            ),
            number_of_digits_to_round,
        )
        == round(0.1946865334809922, number_of_digits_to_round)
    )

    assert (
        round(
            get_weighted_mean_blocking_difference_between_two_systems(
                prop_1=0.5,
                lambda_2=2,
                lambda_1_1=2,
                lambda_1_2=2,
                mu_1=2,
                mu_2=2,
                num_of_servers_1=2,
                num_of_servers_2=2,
                threshold_1=3,
                threshold_2=5,
                system_capacity_1=5,
                system_capacity_2=5,
                buffer_capacity_1=5,
                buffer_capacity_2=4,
                alpha=1,
            ),
            number_of_digits_to_round,
        )
        == round(0.010692228094570821, number_of_digits_to_round)
    )


def test_get_weighted_mean_blocking_difference_between_two_systems_example_2():
    """
    Test for getting the weighted mean blocking difference between two systems
    for different values of alpha
    """
    assert (
        round(
            get_weighted_mean_blocking_difference_between_two_systems(
                prop_1=0.8,
                lambda_2=4,
                lambda_1_1=3,
                lambda_1_2=2,
                mu_1=3,
                mu_2=2,
                num_of_servers_1=3,
                num_of_servers_2=4,
                threshold_1=6,
                threshold_2=5,
                system_capacity_1=8,
                system_capacity_2=9,
                buffer_capacity_1=4,
                buffer_capacity_2=3,
                alpha=0,
            ),
            number_of_digits_to_round,
        )
        == round(0.039826434411056905, number_of_digits_to_round)
    )

    assert (
        round(
            get_weighted_mean_blocking_difference_between_two_systems(
                prop_1=0.8,
                lambda_2=4,
                lambda_1_1=3,
                lambda_1_2=2,
                mu_1=3,
                mu_2=2,
                num_of_servers_1=3,
                num_of_servers_2=4,
                threshold_1=6,
                threshold_2=5,
                system_capacity_1=8,
                system_capacity_2=9,
                buffer_capacity_1=4,
                buffer_capacity_2=3,
                alpha=0.5,
            ),
            number_of_digits_to_round,
        )
        == round(0.024201250377136538, number_of_digits_to_round)
    )

    assert (
        round(
            get_weighted_mean_blocking_difference_between_two_systems(
                prop_1=0.8,
                lambda_2=4,
                lambda_1_1=3,
                lambda_1_2=2,
                mu_1=3,
                mu_2=2,
                num_of_servers_1=3,
                num_of_servers_2=4,
                threshold_1=6,
                threshold_2=5,
                system_capacity_1=8,
                system_capacity_2=9,
                buffer_capacity_1=4,
                buffer_capacity_2=3,
                alpha=1,
            ),
            number_of_digits_to_round,
        )
        == round(0.008576066343216171, number_of_digits_to_round)
    )


def test_calculate_class_2_individuals_best_response_example_1():
    """
    Test for calculating the best response of distributing class 2 individuals
    with same parameter systems
    """
    assert (
        calculate_class_2_individuals_best_response(
            lambda_2=2,
            lambda_1_1=1,
            lambda_1_2=1,
            mu_1=2,
            mu_2=2,
            num_of_servers_1=3,
            num_of_servers_2=3,
            threshold_1=3,
            threshold_2=3,
            system_capacity_1=5,
            system_capacity_2=5,
            buffer_capacity_1=4,
            buffer_capacity_2=4,
        )
        == 0.5
    )


def test_calculate_class_2_individuals_best_response_example_2():
    """
    Test for calculating the best response of distributing class 2 individuals
    for slightly larger model.
    """
    assert (
        round(
            calculate_class_2_individuals_best_response(
                lambda_2=6,
                lambda_1_1=2,
                lambda_1_2=3,
                mu_1=5,
                mu_2=2,
                num_of_servers_1=3,
                num_of_servers_2=4,
                threshold_1=7,
                threshold_2=9,
                system_capacity_1=10,
                system_capacity_2=10,
                buffer_capacity_1=10,
                buffer_capacity_2=10,
            ),
            number_of_digits_to_round,
        )
        == round(0.8224704160104401, number_of_digits_to_round)
    )


def test_calculate_class_2_individuals_best_response_upper_and_lower_bounds():
    """
    Tests that when both the lower and upper bound of the routing function have
    the same sign, then the output is either 0 (when positive) and 1 (when
    negative).
    """
    assert (
        calculate_class_2_individuals_best_response(
            lambda_2=2,
            lambda_1_1=1,
            lambda_1_2=1,
            mu_1=2,
            mu_2=2,
            num_of_servers_1=3,
            num_of_servers_2=3,
            threshold_1=3,
            threshold_2=3,
            system_capacity_1=5,
            system_capacity_2=5,
            buffer_capacity_1=4,
            buffer_capacity_2=4,
            lower_bound=0.1,
            upper_bound=0.2,
        )
        == 1
    )

    assert (
        calculate_class_2_individuals_best_response(
            lambda_2=2,
            lambda_1_1=1,
            lambda_1_2=1,
            mu_1=2,
            mu_2=2,
            num_of_servers_1=3,
            num_of_servers_2=3,
            threshold_1=3,
            threshold_2=3,
            system_capacity_1=5,
            system_capacity_2=5,
            buffer_capacity_1=4,
            buffer_capacity_2=4,
            lower_bound=0.8,
            upper_bound=0.9,
        )
        == 0
    )


def test_get_routing_matrix_example_1():
    """
    Test for the routing matrix of the game
    """
    assert np.allclose(
        get_routing_matrix(
            lambda_2=1,
            lambda_1_1=1,
            lambda_1_2=1,
            mu_1=1,
            mu_2=1,
            num_of_servers_1=3,
            num_of_servers_2=3,
            system_capacity_1=3,
            system_capacity_2=3,
            buffer_capacity_1=2,
            buffer_capacity_2=2,
            alpha=0.5,
        ),
        np.array([[0.5, 0.0, 0.0], [1.0, 0.5, 0.0], [1.0, 1.0, 0.5]]),
    )


def test_get_routing_matrix_example_2():
    """
    Test for the routing matrix of the game
    """
    assert np.allclose(
        get_routing_matrix(
            lambda_2=10,
            lambda_1_1=0,
            lambda_1_2=5,
            mu_1=1,
            mu_2=1,
            num_of_servers_1=4,
            num_of_servers_2=4,
            system_capacity_1=3,
            system_capacity_2=3,
            buffer_capacity_1=2,
            buffer_capacity_2=4,
            alpha=0.5,
        ),
        np.array(
            [
                [1.0, 0.95206422, 0.16897752],
                [1.0, 0.98501658, 0.51821881],
                [1.0, 1.0, 0.66397863],
            ]
        ),
    )


def test_get_routing_matrix_example_3():
    """
    Test for the routing matrix of the game
    """
    assert np.allclose(
        get_routing_matrix(
            lambda_2=7,
            lambda_1_1=3,
            lambda_1_2=4,
            mu_1=2,
            mu_2=2,
            num_of_servers_1=3,
            num_of_servers_2=3,
            system_capacity_1=5,
            system_capacity_2=5,
            buffer_capacity_1=4,
            buffer_capacity_2=4,
            alpha=0.5,
        ),
        np.array(
            [
                [0.88685659, 0.09056231, 0.03000287, 0.0, 0.0],
                [1.0, 0.67730979, 0.34483691, 0.24191106, 0.08278824],
                [1.0, 0.8569611, 0.60484286, 0.46578746, 0.2444959],
                [1.0, 0.89931756, 0.68934163, 0.55754486, 0.34566953],
                [1.0, 1.0, 0.85033968, 0.72747955, 0.5226364],
            ]
        ),
    )


def test_get_individual_entries_of_matrices_example():
    """
    Tests that the function returns a dask task and that the computed task
    returns the expected tuple
    """
    task = get_individual_entries_of_matrices(
        lambda_2=2,
        lambda_1_1=2,
        lambda_1_2=2,
        mu_1=2,
        mu_2=2,
        num_of_servers_1=2,
        num_of_servers_2=2,
        threshold_1=2,
        threshold_2=2,
        system_capacity_1=4,
        system_capacity_2=4,
        buffer_capacity_1=2,
        buffer_capacity_2=2,
        alpha=0.5,
        target=2,
    )

    assert da.is_dask_collection(task)
    values = da.compute(task)
    assert values == ((2, 2, 0.5, -0.00046944342133137197, -0.00046944342133137197),)


def test_compute_tasks():
    """
    Tests that dask tasks are computed as expected
    """

    @da.delayed
    def inc(x):
        return x + 1

    @da.delayed
    def double(x):
        return x * 2

    x = random.random()
    y = random.random()

    tasks = tuple((inc(x), double(y)))
    assert compute_tasks(tasks, processes=None) == (x + 1, 2 * y)


def test_build_matrices_from_computed_tasks():
    """
    Tests the matrices that are generated from computed tasks (tuple)
    """
    computed_tasks = tuple(
        (
            (1, 1, 1, 2, 3),
            (1, 2, 10, 20, 30),
            (2, 1, 100, 200, 300),
            (2, 2, 1000, 2000, 3000),
        )
    )
    routing, utility_1, utility_2 = build_matrices_from_computed_tasks(
        computed_tasks=computed_tasks, N_1=2, N_2=2
    )
    assert np.allclose(routing, np.array([[1, 10], [100, 1000]]))
    assert np.allclose(utility_1, np.array([[2, 20], [200, 2000]]))
    assert np.allclose(utility_2, np.array([[3, 30], [300, 3000]]))


def test_get_payoff_matrices_example_1():
    """
    Test for payoff matrices of the game
    """
    A, B, _ = get_payoff_matrices(
        lambda_2=1,
        lambda_1_1=1,
        lambda_1_2=1,
        mu_1=1,
        mu_2=1,
        num_of_servers_1=1,
        num_of_servers_2=1,
        system_capacity_1=2,
        system_capacity_2=2,
        buffer_capacity_1=2,
        buffer_capacity_2=2,
        target=1,
    )
    assert np.allclose(
        A, np.array([[-0.25182247, -0.25182247], [-0.40094816, -0.34137716]])
    )

    assert np.allclose(
        B, np.array([[-0.25182247, -0.40094816], [-0.25182247, -0.34137716]])
    )


def test_get_payoff_matrices_example_2():
    """
    Test for payoff matrices of the game using 2 processes
    """
    A, B, _ = get_payoff_matrices(
        lambda_2=2,
        lambda_1_1=2,
        lambda_1_2=2,
        mu_1=2,
        mu_2=2,
        num_of_servers_1=2,
        num_of_servers_2=2,
        system_capacity_1=4,
        system_capacity_2=4,
        buffer_capacity_1=4,
        buffer_capacity_2=4,
        target=2,
        processes=2,
    )

    assert np.allclose(
        A,
        np.array(
            [
                [-5.64325041e-04, -5.64325041e-04, -5.64325041e-04, -5.64325041e-04],
                [-4.11252209e-04, -4.61900039e-04, -5.01311925e-04, -5.64325041e-04],
                [-1.02850193e-04, -1.82421878e-04, -2.78276595e-04, -4.50963918e-04],
                [-2.75913690e-05, -2.75913690e-05, -8.23151544e-05, -2.33912176e-04],
            ]
        ),
    )

    assert np.allclose(
        B,
        np.array(
            [
                [-5.64325041e-04, -4.11252209e-04, -1.02850193e-04, -2.75913690e-05],
                [-5.64325041e-04, -4.61900039e-04, -1.82421878e-04, -2.75913690e-05],
                [-5.64325041e-04, -5.01311925e-04, -2.78276595e-04, -8.23151544e-05],
                [-5.64325041e-04, -5.64325041e-04, -4.50963918e-04, -2.33912176e-04],
            ]
        ),
    )


def test_get_payoff_matrices_example_3():
    """
    Test for payoff matrices of the game when the alternative utility is used
    """
    A, B, _ = get_payoff_matrices(
        lambda_2=1,
        lambda_1_1=1,
        lambda_1_2=1,
        mu_1=1,
        mu_2=1,
        num_of_servers_1=1,
        num_of_servers_2=1,
        system_capacity_1=2,
        system_capacity_2=2,
        buffer_capacity_1=2,
        buffer_capacity_2=2,
        target=1,
        alternative_utility=True,
    )
    assert np.allclose(A, np.array([[0.44818084, 0.44818084], [0.31679532, 0.3657251]]))

    assert np.allclose(B, np.array([[0.44818084, 0.31679532], [0.44818084, 0.3657251]]))


def test_build_game_using_payoff_matrices_example_1():
    """
    Test representation of the game
    """
    game = build_game_using_payoff_matrices(
        lambda_2=1,
        lambda_1_1=1,
        lambda_1_2=1,
        mu_1=1,
        mu_2=1,
        num_of_servers_1=1,
        num_of_servers_2=1,
        system_capacity_1=2,
        system_capacity_2=2,
        buffer_capacity_1=2,
        buffer_capacity_2=2,
        target=1,
    )

    assert len(game.payoff_matrices) == 2
    assert (
        game.__repr__()
        == """Bi matrix game with payoff matrices:

Row player:
[[-0.25182247 -0.25182247]
 [-0.40094816 -0.34137716]]

Column player:
[[-0.25182247 -0.40094816]
 [-0.25182247 -0.34137716]]"""
    )


def test_build_game_using_payoff_matrices_example_2():
    """
    Test the game's payoff matrices
    """
    game = build_game_using_payoff_matrices(
        lambda_2=5,
        lambda_1_1=1,
        lambda_1_2=1,
        mu_1=3,
        mu_2=3,
        num_of_servers_1=2,
        num_of_servers_2=2,
        system_capacity_1=4,
        system_capacity_2=5,
        buffer_capacity_1=2,
        buffer_capacity_2=2,
        target=2,
    )

    assert np.allclose(
        game.payoff_matrices[0],
        np.array(
            [
                [-0.00224433, -0.00224433, -0.00224433, -0.00224433, -0.00224433],
                [-0.00221647, -0.00222381, -0.00222728, -0.00223013, -0.00223415],
                [-0.00205908, -0.00211616, -0.00214196, -0.00216115, -0.00218337],
                [-0.00187811, -0.00197168, -0.00202778, -0.00206889, -0.00211227],
            ]
        ),
    )

    assert np.allclose(
        game.payoff_matrices[1],
        np.array(
            [
                [-0.00224261, -0.00221144, -0.00203882, -0.00178084, -0.00151419],
                [-0.00224261, -0.00221978, -0.00210315, -0.00192509, -0.00169457],
                [-0.00224261, -0.00222403, -0.00213345, -0.0019975, -0.00182025],
                [-0.00224261, -0.00222935, -0.00216478, -0.0020671, -0.00193602],
            ]
        ),
    )
