"""
Tests for the game.py module
"""

import dask as da
import numpy as np
from hypothesis import given, settings
from hypothesis.strategies import floats

from ambulance_game.game import (
    build_game_using_payoff_matrices,
    build_matrices_from_computed_tasks,
    calculate_class_2_individuals_best_response,
    compute_tasks,
    get_individual_entries_of_matrices,
    get_payoff_matrices,
    get_routing_matrix,
)

NUMBER_OF_DIGITS_TO_ROUND = 8


def test_calculate_class_2_individuals_best_response_markov_example_1():
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


def test_calculate_class_2_individuals_best_response_markov_example_2():
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
            NUMBER_OF_DIGITS_TO_ROUND,
        )
        == round(0.8224704160104401, NUMBER_OF_DIGITS_TO_ROUND)
    )


def test_calculate_class_2_individuals_best_response_markov_upper_and_lower_bounds():
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


#  TODO Investigate making it a property based test
def test_calculate_class_2_individuals_best_response_simulation_equal_split():
    """Make sure that the brentq() function that is used suggests that when two
    identical systems are considered the individuals will be split equally between
    them (50% - 50%)

    Note here that due to the ciw.seed() function it was possible to eliminate any
    randomness and make both systems identical, in terms of arrivals, services
    and any other stochasticity that the simulation models incorporates.
    """
    lambda_2 = 0.3
    equal_split = calculate_class_2_individuals_best_response(
        lambda_2=lambda_2,
        lambda_1_1=0.3,
        lambda_1_2=0.3,
        mu_1=0.2,
        mu_2=0.2,
        num_of_servers_1=4,
        num_of_servers_2=4,
        threshold_1=3,
        threshold_2=3,
        system_capacity_1=float("inf"),
        system_capacity_2=float("inf"),
        buffer_capacity_1=float("inf"),
        buffer_capacity_2=float("inf"),
        use_simulation=True,
        runtime=500,
        num_of_trials=5,
        warm_up_time=100,
        seed_num_1=0,
        seed_num_2=0,
    )

    assert np.isclose(equal_split, 0.5)


def test_calculate_class_2_individuals_best_response_simulation_all_inds_in_one():
    """
    Ensuring that the function is sends 100% of individuals to the first system
    when the second system is very busy and vise versa.
    """
    all_individuals_to_first = calculate_class_2_individuals_best_response(
        lambda_2=0.3,
        lambda_1_1=0.1,
        lambda_1_2=3,
        mu_1=10,
        mu_2=2,
        num_of_servers_1=8,
        num_of_servers_2=4,
        threshold_1=6,
        threshold_2=3,
        system_capacity_1=float("inf"),
        system_capacity_2=float("inf"),
        buffer_capacity_1=float("inf"),
        buffer_capacity_2=float("inf"),
        use_simulation=True,
        runtime=500,
        num_of_trials=5,
        warm_up_time=100,
        seed_num_1=10,
        seed_num_2=10,
    )
    assert all_individuals_to_first == 1

    all_individuals_to_second = calculate_class_2_individuals_best_response(
        lambda_2=0.3,
        lambda_1_1=3,
        lambda_1_2=0.1,
        mu_1=2,
        mu_2=10,
        num_of_servers_1=4,
        num_of_servers_2=8,
        threshold_1=3,
        threshold_2=6,
        system_capacity_1=float("inf"),
        system_capacity_2=float("inf"),
        buffer_capacity_1=float("inf"),
        buffer_capacity_2=float("inf"),
        use_simulation=True,
        runtime=500,
        num_of_trials=5,
        warm_up_time=100,
        seed_num_1=10,
        seed_num_2=10,
    )
    assert all_individuals_to_second == 0


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


def test_get_individual_entries_of_matrices_markov_example():
    """
    Tests that the function returns a dask task and that the computed task
    returns the expected tuple while using the markov model
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
        use_cache=False,
    )

    assert da.is_dask_collection(task)
    values = da.compute(task)
    assert np.allclose(
        values, ((2, 2, 0.5, -0.00046944342133137197, -0.00046944342133137197),)
    )


def test_get_individual_entries_of_matrices_simulation_example():
    """
    Tests that the function returns a dask task and that the computed task
    returns the expected tuple while using the simuation
    """
    task = get_individual_entries_of_matrices(
        lambda_2=2,
        lambda_1_1=0.1,
        lambda_1_2=0.5,
        mu_1=2,
        mu_2=1,
        num_of_servers_1=2,
        num_of_servers_2=2,
        threshold_1=3,
        threshold_2=5,
        system_capacity_1=4,
        system_capacity_2=6,
        buffer_capacity_1=2,
        buffer_capacity_2=2,
        target=2,
        alpha=0.5,
        use_simulation=True,
        runtime=300,
        num_of_trials=3,
        warm_up_time=5,
        seed_num_1=0,
        seed_num_2=0,
    )

    assert da.is_dask_collection(task)
    values = da.compute(task)
    assert np.allclose(
        values,
        ((3, 5, 0.7613063676529543, -0.0006520260736895711, -0.027937014444158834),),
    )


@settings(max_examples=20)
@given(
    float_1=floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
    float_2=floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
)
def test_compute_tasks(float_1, float_2):
    """
    Tests that dask tasks are computed as expected
    """

    @da.delayed
    def inc(num):
        return num + 1

    @da.delayed
    def double(num):
        return num * 2

    tasks = tuple((inc(float_1), double(float_2)))
    assert compute_tasks(tasks, processes=None) == (float_1 + 1, 2 * float_2)


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
        computed_tasks=computed_tasks, n_1=2, n_2=2
    )
    assert np.allclose(routing, np.array([[1, 10], [100, 1000]]))
    assert np.allclose(utility_1, np.array([[2, 20], [200, 2000]]))
    assert np.allclose(utility_2, np.array([[3, 30], [300, 3000]]))


def test_get_payoff_matrices_example_1():
    """
    Test for payoff matrices of the game
    """
    payoff_matrix_A, payoff_matrix_B, _ = get_payoff_matrices(
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
        payoff_matrix_A,
        np.array([[-0.25182247, -0.25182247], [-0.40094816, -0.34137716]]),
    )

    assert np.allclose(
        payoff_matrix_B,
        np.array([[-0.25182247, -0.40094816], [-0.25182247, -0.34137716]]),
    )


def test_get_payoff_matrices_example_2():
    """
    Test for payoff matrices of the game using 2 processes
    """
    payoff_matrix_A, payoff_matrix_B, _ = get_payoff_matrices(
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
        payoff_matrix_A,
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
        payoff_matrix_B,
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
    payoff_matrix_A, payoff_matrix_B, _ = get_payoff_matrices(
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
    assert np.allclose(
        payoff_matrix_A, np.array([[0.44818084, 0.44818084], [0.31679532, 0.3657251]])
    )

    assert np.allclose(
        payoff_matrix_B, np.array([[0.44818084, 0.31679532], [0.44818084, 0.3657251]])
    )


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
