import numpy as np
import pytest

from ambulance_game.markov.blocking import (
    get_coefficients_row_of_array_associated_with_state,
    get_blocking_time_linear_system,
    convert_solution_to_correct_array_format,
    get_blocking_times_of_all_states_using_direct_approach,
    mean_blocking_time_formula_using_direct_approach,
    mean_blocking_time_formula_using_closed_form_approach,
    get_mean_blocking_time_using_markov_state_probabilities,
)

number_of_digits_to_round = 8


def test_get_coefficients_row_of_array_associated_with_state_example_1():
    M_row, b_element = get_coefficients_row_of_array_associated_with_state(
        state=(2, 1),
        lambda_1=0.3,
        mu=0.5,
        num_of_servers=1,
        threshold=1,
        system_capacity=5,
        buffer_capacity=3,
    )
    assert np.allclose(
        M_row, np.array([0.625, 0, 0, 0, 0, -1, 0.375, 0, 0, 0, 0, 0, 0, 0, 0])
    )
    assert b_element == -1.25


def test_get_coefficients_row_of_array_associated_with_state_example_2():
    M_row, b_element = get_coefficients_row_of_array_associated_with_state(
        state=(4, 7),
        lambda_1=2,
        mu=1,
        num_of_servers=3,
        threshold=4,
        system_capacity=10,
        buffer_capacity=5,
    )
    assert np.allclose(
        M_row,
        np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.6,
                -1.0,
                0.4,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        ),
    )
    assert b_element == -0.2


def test_get_coefficients_row_of_array_associated_with_state_example_3():
    with pytest.raises(IndexError):
        get_coefficients_row_of_array_associated_with_state(
            state=(4, 7),
            lambda_1=2,
            mu=1,
            num_of_servers=3,
            threshold=10,
            system_capacity=10,
            buffer_capacity=5,
        )


def test_get_coefficients_row_of_array_associated_with_state_example_4():
    assert (
        get_coefficients_row_of_array_associated_with_state(
            state=(0, 4),
            lambda_1=2,
            mu=1,
            num_of_servers=3,
            threshold=10,
            system_capacity=10,
            buffer_capacity=5,
        )
        == 0
    )


def test_get_blocking_time_linear_system_example_1():
    M, b = get_blocking_time_linear_system(
        lambda_1=2,
        mu=3,
        num_of_servers=1,
        threshold=3,
        system_capacity=4,
        buffer_capacity=2,
    )
    assert np.alltrue(
        M
        == np.array(
            [
                [-1.0, 0.4, 0.0, 0.0],
                [0.6, 0.0, -1.0, 0.4],
                [1.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, -1.0],
            ]
        ),
    )
    assert np.alltrue(b == [-0.2, -0.2, -0.3333333333333333, -0.3333333333333333])


def test_get_blocking_time_linear_system_example_2():
    M, b = get_blocking_time_linear_system(
        lambda_1=2,
        mu=1,
        num_of_servers=3,
        threshold=3,
        system_capacity=5,
        buffer_capacity=2,
    )
    assert np.alltrue(
        M
        == np.array(
            [
                [-1.0, 0.4, 0.0, 0.0, 0.0, 0.0],
                [0.6, 0.0, 0.0, -1.0, 0.4, 0.0],
                [0.6, -1.0, 0.4, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.6, -1.0, 0.4],
                [0.0, 1.0, -1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, -1.0],
            ]
        ),
    )
    assert np.alltrue(
        b == [-0.2, -0.2, -0.2, -0.2, -0.3333333333333333, -0.3333333333333333]
    )


def test_get_blocking_time_linear_system_example_3():
    M, b = get_blocking_time_linear_system(
        lambda_1=0.4,
        mu=0.1,
        num_of_servers=6,
        threshold=4,
        system_capacity=4,
        buffer_capacity=7,
    )
    assert np.alltrue(
        M
        == np.array(
            [
                [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0],
            ]
        ),
    )
    assert np.alltrue(b == [-2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5])


def test_convert_solution_to_correct_array_format_examples():
    converted_1 = convert_solution_to_correct_array_format(
        np.array([1, 2, 3, 4, 5, 6]), 2, 4, 2
    )
    assert np.alltrue(
        converted_1 == np.array([[0, 0, 0, 0, 0], [0, 0, 1, 2, 3], [0, 0, 4, 5, 6]])
    )

    converted_2 = convert_solution_to_correct_array_format(
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]), 3, 5, 3
    )
    assert np.alltrue(
        converted_2
        == np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 2, 3],
                [0, 0, 0, 4, 5, 6],
                [0, 0, 0, 7, 8, 9],
            ]
        )
    )


def test_get_blocking_times_of_all_states_using_direct_approach_example_1():
    """Example of blocking times of all states when the threshold is the same as
    the system capacity (T = N)
    """
    blocking_times = get_blocking_times_of_all_states_using_direct_approach(
        lambda_1=2,
        mu=3,
        num_of_servers=1,
        threshold=3,
        system_capacity=3,
        buffer_capacity=4,
    )
    assert np.allclose(
        blocking_times,
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.33333333],
                [0.0, 0.0, 0.0, 0.66666667],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.33333333],
            ]
        ),
    )


def test_get_blocking_times_of_all_states_using_direct_approach_example_2():
    blocking_times = get_blocking_times_of_all_states_using_direct_approach(
        lambda_1=2,
        mu=1,
        num_of_servers=3,
        threshold=1,
        system_capacity=4,
        buffer_capacity=5,
    )
    assert np.allclose(
        blocking_times,
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 3.11111111, 4.16666667, 4.72222222, 5.05555556],
                [0.0, 6.22222222, 7.27777778, 7.83333333, 8.16666667],
                [0.0, 9.33333333, 10.38888889, 10.94444444, 11.27777778],
                [0.0, 12.44444444, 13.5, 14.05555556, 14.38888889],
                [0.0, 15.55555556, 16.61111111, 17.16666667, 17.5],
            ]
        ),
    )


def test_get_blocking_times_of_all_states_using_direct_approach_example_3():
    blocking_times = get_blocking_times_of_all_states_using_direct_approach(
        lambda_1=4,
        mu=1,
        num_of_servers=5,
        threshold=3,
        system_capacity=6,
        buffer_capacity=8,
    )
    assert np.allclose(
        blocking_times,
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.14666667, 1.75666667, 2.11666667, 2.31666667],
                [0.0, 0.0, 0.0, 2.29333333, 2.90333333, 3.26333333, 3.46333333],
                [0.0, 0.0, 0.0, 3.44, 4.05, 4.41, 4.61],
                [0.0, 0.0, 0.0, 4.58666667, 5.19666667, 5.55666667, 5.75666667],
                [0.0, 0.0, 0.0, 5.73333333, 6.34333333, 6.70333333, 6.90333333],
                [0.0, 0.0, 0.0, 6.88, 7.49, 7.85, 8.05],
                [0.0, 0.0, 0.0, 8.02666667, 8.63666667, 8.99666667, 9.19666667],
                [0.0, 0.0, 0.0, 9.17333333, 9.78333333, 10.14333333, 10.34333333],
            ]
        ),
    )


def test_mean_blocking_time_formula_using_direct_approach():
    all_states = [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 4),
        (2, 4),
        (0, 5),
        (1, 5),
        (2, 5),
        (0, 6),
        (1, 6),
        (2, 6),
        (0, 7),
        (1, 7),
        (2, 7),
        (0, 8),
        (1, 8),
        (2, 8),
    ]
    state_probabilities = np.array(
        [
            [
                0.05924777,
                0.14811941,
                0.18514927,
                0.15429106,
                0.12857588,
                0.04291957,
                0.01439794,
                0.00493644,
                0.00185116,
            ],
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                0.064227,
                0.03378794,
                0.01552454,
                0.00676837,
                0.00300093,
            ],
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                0.04110293,
                0.04024539,
                0.02855397,
                0.01753342,
                0.00976702,
            ],
        ]
    )
    blocking_time = mean_blocking_time_formula_using_direct_approach(
        all_states=all_states,
        pi=state_probabilities,
        lambda_1=3,
        mu=2,
        num_of_servers=3,
        threshold=4,
        system_capacity=8,
        buffer_capacity=2,
    )
    assert round(blocking_time, number_of_digits_to_round) == 0.23047954


def test_get_mean_blocking_time_using_markov_state_probabilities_example_1():
    assert (
        round(
            get_mean_blocking_time_using_markov_state_probabilities(
                lambda_2=2,
                lambda_1=3,
                mu=2,
                num_of_servers=3,
                threshold=4,
                system_capacity=8,
                buffer_capacity=2,
                blocking_formula=mean_blocking_time_formula_using_direct_approach,
            ),
            number_of_digits_to_round,
        )
        == 0.23047954
    )


def test_get_mean_blocking_time_using_markov_state_probabilities_example_2():
    assert (
        round(
            get_mean_blocking_time_using_markov_state_probabilities(
                lambda_2=5,
                lambda_1=6,
                mu=2,
                num_of_servers=7,
                threshold=5,
                system_capacity=15,
                buffer_capacity=7,
                blocking_formula=mean_blocking_time_formula_using_direct_approach,
            ),
            number_of_digits_to_round,
        )
        == 0.62492091
    )


def test_mean_blocking_time_formula_using_direct_approach_closed_form():
    # TODO: Make test once closed form formula is found
    with pytest.raises(NotImplementedError):
        mean_blocking_time_formula_using_closed_form_approach(
            all_states=None,
            pi=None,
            lambda_1=None,
            mu=None,
            num_of_servers=None,
            threshold=None,
            system_capacity=None,
            buffer_capacity=None,
        )
