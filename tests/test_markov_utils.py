"""
Tests for the utils.py module
"""

import random

import numpy as np

from ambulance_game.markov.utils import (
    expected_time_in_markov_state_ignoring_arrivals,
    expected_time_in_markov_state_ignoring_class_2_arrivals,
    is_accepting_state,
    is_blocking_state,
    is_waiting_state,
    prob_class_1_arrival,
    prob_service,
    get_probability_of_accepting,
    get_proportion_of_individuals_not_lost,
    get_accepting_proportion_of_class_2_individuals,
    get_accepting_proportion_of_individuals,
)

NUMBER_OF_DIGITS_TO_ROUND = 8


def test_is_waiting_state():
    """
    Ensure that function no matter the value of u of a state (u,v) returns:
        - True when u > num_of_servers
        - False when u <= num_of_servers
    """
    for v in range(11, 100):
        u = random.randint(1, 100)
        assert is_waiting_state(state=(u, v), num_of_servers=10)

    for v in range(1, 11):
        u = random.randint(1, 100)
        assert not is_waiting_state(state=(u, v), num_of_servers=10)


def test_is_blocking_state():
    """
    Ensure that function returns:
        - True when state is of the form (u,v) and u > 0
        - False when state is of the form (u,v) and u = 0
    """
    for v in range(1, 100):
        u = random.randint(1, 100)
        assert is_blocking_state(state=(u, v))

    for v in range(1, 100):
        assert not is_blocking_state(state=(0, v))


def test_is_accepting_state_when_false():
    """
    Test to ensure that function returns False for class 1 patients when on the
    final column of the Markov chain and for class 2 patient when on the final row
    of the Markov chain
    """
    for max_v in range(1, 20):
        assert not is_accepting_state(
            state=(0, max_v),
            class_type=0,
            threshold=2,
            system_capacity=max_v,
            buffer_capacity=10,
        )
    for max_v in range(1, 20):
        assert not is_accepting_state(
            state=(max_v, 6),
            class_type=1,
            threshold=2,
            system_capacity=6,
            buffer_capacity=max_v,
        )


def test_is_accepting_state_when_true():
    """
    Test to ensure that function returns True for class 1 patients when not on the
    final column of the Markov chain and for class 2 patient when not on the final
    row of the Markov chain
    """
    for u in range(1, 11):
        for v in range(1, 10):
            assert is_accepting_state(
                state=(u, v),
                class_type=0,
                threshold=2,
                system_capacity=10,
                buffer_capacity=10,
            )

    for u in range(1, 10):
        for v in range(1, 11):
            assert is_accepting_state(
                state=(u, v),
                class_type=1,
                threshold=2,
                system_capacity=10,
                buffer_capacity=10,
            )


def test_expected_time_in_markov_state_ignoring_arrivals_example_1():
    """
    Test for class 1 patient when on threshold column
    """
    for u in range(1, 10):
        assert (
            expected_time_in_markov_state_ignoring_arrivals(
                state=(u, 5),
                class_type=0,
                num_of_servers=4,
                mu=3,
                threshold=5,
            )
            == 0
        )


def test_expected_time_in_markov_state_ignoring_arrivals_example_2():
    """
    Test for class 2 patient when on a state (u,v) with u > 0
    """
    for u in range(1, 10):
        for v in range(1, 10):
            assert (
                expected_time_in_markov_state_ignoring_arrivals(
                    state=(u, v),
                    class_type=1,
                    num_of_servers=4,
                    mu=3,
                    threshold=1,
                )
                == 0
            )


def test_expected_time_in_markov_state_ignoring_arrivals_example_3():
    """
    Example test for class 1 patients when on a state (u,v) with v > C
    """
    assert (
        expected_time_in_markov_state_ignoring_arrivals(
            state=(1, 15),
            class_type=0,
            num_of_servers=4,
            mu=3,
            threshold=1,
        )
        == 1 / 12
    )


def test_expected_time_in_markov_state_ignoring_arrivals_example_4():
    """
    Example test for class 1 patients when on a state (u,v) with v < C
    """
    assert (
        expected_time_in_markov_state_ignoring_arrivals(
            state=(1, 7),
            class_type=0,
            num_of_servers=8,
            mu=3,
            threshold=1,
        )
        == 1 / 21
    )


def test_expected_time_in_markov_state_ignoring_class_2_arrivals():
    """
    Ensure that the expected time spent in state (u,v) does not depends on the value
    of u (with the exception of u=0).
    """
    assert (
        round(
            expected_time_in_markov_state_ignoring_class_2_arrivals(
                state=(1, 3), lambda_1=0.4, mu=1.2, num_of_servers=4, system_capacity=5
            ),
            NUMBER_OF_DIGITS_TO_ROUND,
        )
        == round(
            expected_time_in_markov_state_ignoring_class_2_arrivals(
                state=(100, 3),
                lambda_1=0.4,
                mu=1.2,
                num_of_servers=4,
                system_capacity=5,
            ),
            NUMBER_OF_DIGITS_TO_ROUND,
        )
        == 0.25
    )

    assert (
        round(
            expected_time_in_markov_state_ignoring_class_2_arrivals(
                state=(1, 5), lambda_1=0.4, mu=1.2, num_of_servers=4, system_capacity=5
            ),
            NUMBER_OF_DIGITS_TO_ROUND,
        )
        == round(
            expected_time_in_markov_state_ignoring_class_2_arrivals(
                state=(100, 5),
                lambda_1=0.4,
                mu=1.2,
                num_of_servers=4,
                system_capacity=5,
            ),
            NUMBER_OF_DIGITS_TO_ROUND,
        )
        == 0.20833333
    )


def test_prob_service():
    """
    Ensure that probability of service remains fixed for all states when C=1
    """
    for v in range(1, 100):
        u = random.randint(1, 100)
        mu = random.randint(1, 100)
        lambda_1 = random.random()
        prob = prob_service(state=(u, v), lambda_1=lambda_1, mu=mu, num_of_servers=1)
        assert prob == mu / (lambda_1 + mu)


def test_prob_class_1_arrival():
    """
    Ensure that probability of class 1 arrivals remains fixed for all states when C=1
    """
    for v in range(1, 100):
        u = random.randint(1, 100)
        mu = random.randint(1, 100)
        lambda_1 = random.random()
        prob = prob_class_1_arrival(
            state=(u, v), lambda_1=lambda_1, mu=mu, num_of_servers=1
        )
        assert prob == lambda_1 / (lambda_1 + mu)


def test_get_probability_of_accepting_example():
    """
    Test that the probability of accepting an individual is as expected
    """
    all_states = [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 3),
        (2, 3),
        (0, 4),
        (1, 4),
        (2, 4),
        (0, 5),
        (1, 5),
        (2, 5),
    ]
    pi = np.array(
        [
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [np.nan, np.nan, np.nan, 0.1, 0.1, 0.05],
            [np.nan, np.nan, np.nan, 0.05, 0.05, 0.05],
        ]
    )

    assert np.allclose(
        get_probability_of_accepting(
            all_states=all_states,
            pi=pi,
            threshold=2,
            system_capacity=5,
            buffer_capacity=2,
        ),
        [0.8, 0.85],
    )


def test_get_proportion_of_individuals_not_lost_example():
    """
    Tests that the proportion of non-lost individuals is as expected
    """
    all_states = [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 3),
        (2, 3),
        (0, 4),
        (1, 4),
        (2, 4),
        (0, 5),
        (1, 5),
        (2, 5),
    ]
    pi = np.array(
        [
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [np.nan, np.nan, np.nan, 0.1, 0.1, 0.05],
            [np.nan, np.nan, np.nan, 0.05, 0.05, 0.05],
        ]
    )

    assert np.allclose(
        get_proportion_of_individuals_not_lost(
            all_states=all_states,
            pi=pi,
            lambda_1=3,
            lambda_2=2,
            threshold=2,
            system_capacity=5,
            buffer_capacity=2,
        ),
        [0.5853658536585366, 0.4146341463414634],
    )


def test_get_accepting_proportion_of_class_2_individuals():
    """
    Tests that the proportion of class 2 individuals that are accepted is as expected
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


def test_get_accepting_proportion_of_individuals():
    """
    Tests that the proportion of individuals that are accepted is as expected
    """
    assert np.allclose(
        get_accepting_proportion_of_individuals(
            lambda_1=1,
            lambda_2=1,
            mu=1,
            num_of_servers=1,
            threshold=1,
            system_capacity=1,
            buffer_capacity=1,
        ),
        [0.2, 0.6],
    )
