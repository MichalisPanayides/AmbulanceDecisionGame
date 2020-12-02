import random

from ambulance_game.markov.utils import (
    is_waiting_state,
    is_blocking_state,
    is_accepting_state,
    expected_time_in_markov_state_ignoring_arrivals,
    expected_time_in_markov_state_ignoring_class_2_arrivals,
    prob_service,
    prob_class_1_arrival,
)

number_of_digits_to_round = 8


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
            number_of_digits_to_round,
        )
        == round(
            expected_time_in_markov_state_ignoring_class_2_arrivals(
                state=(100, 3),
                lambda_1=0.4,
                mu=1.2,
                num_of_servers=4,
                system_capacity=5,
            ),
            number_of_digits_to_round,
        )
        == 0.25
    )

    assert (
        round(
            expected_time_in_markov_state_ignoring_class_2_arrivals(
                state=(1, 5), lambda_1=0.4, mu=1.2, num_of_servers=4, system_capacity=5
            ),
            number_of_digits_to_round,
        )
        == round(
            expected_time_in_markov_state_ignoring_class_2_arrivals(
                state=(100, 5),
                lambda_1=0.4,
                mu=1.2,
                num_of_servers=4,
                system_capacity=5,
            ),
            number_of_digits_to_round,
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
