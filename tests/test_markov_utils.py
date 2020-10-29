import random

from ambulance_game.markov.utils import (
    is_accepting_state,
    is_waiting_state,
    is_blocking_state,
    expected_time_in_markov_state_ignoring_arrivals,
    expected_time_in_markov_state_ignoring_class_2_arrivals,
    prob_service,
    prob_class_1_arrival,
)

number_of_digits_to_round = 8


def test_is_blocking_state():
    """
    Ensure that function returns:
        - True when state is of the form (u,v) and u > 0
        - False when state is of the form (u,v) and u = 0
    """
    for v in range(1, 100):
        u = random.randint(1, 100)
        assert is_blocking_state((u, v))

    for v in range(1, 100):
        assert not is_blocking_state((0, v))


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
