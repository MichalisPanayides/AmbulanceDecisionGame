"""
Tests for the state and server dependent part of the simulation
"""
import pytest
from hypothesis import given, settings
from hypothesis.strategies import floats, integers

from ambulance_game.simulation import simulate_model

NUMBER_OF_DIGITS_TO_ROUND = 8


@given(
    lambda_2=floats(min_value=0.1, max_value=1.0),
    lambda_1=floats(min_value=0.1, max_value=1.0),
    mu=floats(min_value=0.5, max_value=2.0),
    num_of_servers=integers(min_value=1, max_value=10),
)
@settings(max_examples=10)
def test_simulate_state_dependent_model_with_non_state_dependent_property_based(
    lambda_2, lambda_1, mu, num_of_servers
):
    """
    Property based test with state dependent service rate. Ensures that for
    different values of lambda_1, lambda_2, mu and num_of_servers, the results
    of the state dependent and non-state dependent simulation are the same when
    the rates of the state-depndednt one are all set to `mu`
    """
    simulation = simulate_model(
        lambda_2=lambda_2,
        lambda_1=lambda_1,
        mu=mu,
        num_of_servers=num_of_servers,
        threshold=4,
        seed_num=0,
        runtime=100,
        system_capacity=10,
        buffer_capacity=10,
    )

    rates = {(i, j): mu for i in range(11) for j in range(11)}
    simulation_extension = simulate_model(
        lambda_2=lambda_2,
        lambda_1=lambda_1,
        mu=rates,
        num_of_servers=num_of_servers,
        threshold=4,
        seed_num=0,
        runtime=100,
        system_capacity=10,
        buffer_capacity=10,
    )
    assert sum([w.waiting_time for w in simulation.get_all_records()]) == sum(
        [w.waiting_time for w in simulation_extension.get_all_records()]
    )
    assert sum([b.time_blocked for b in simulation.get_all_records()]) == sum(
        [b.time_blocked for b in simulation_extension.get_all_records()]
    )
    assert sum([s.service_time for s in simulation.get_all_records()]) == sum(
        [s.service_time for s in simulation_extension.get_all_records()]
    )


def test_simulate_state_dependent_model_example_1():
    """
    Example 1 for the simulation with state dependent rates
    """
    rates = {
        (0, 0): 0.2,
        (0, 1): 0.5,
        (0, 2): 0.3,
        (0, 3): 0.2,
        (1, 3): 0.2,
        (0, 4): 0.2,
        (1, 4): 0.4,
    }
    simulation = simulate_model(
        lambda_2=0.15,
        lambda_1=0.2,
        mu=rates,
        num_of_servers=2,
        threshold=4,
        seed_num=0,
        runtime=100,
    )

    assert (
        round(
            sum([w.waiting_time for w in simulation.get_all_records()]),
            NUMBER_OF_DIGITS_TO_ROUND,
        )
        == round(69.05359560579672, NUMBER_OF_DIGITS_TO_ROUND)
    )
    assert (
        round(
            sum([b.time_blocked for b in simulation.get_all_records()]),
            NUMBER_OF_DIGITS_TO_ROUND,
        )
        == round(1.8837534828730575, NUMBER_OF_DIGITS_TO_ROUND)
    )
    assert (
        round(
            sum([s.service_time for s in simulation.get_all_records()]),
            NUMBER_OF_DIGITS_TO_ROUND,
        )
        == round(130.39705479506074, NUMBER_OF_DIGITS_TO_ROUND)
    )


def test_simulate_state_dependent_model_example_2():
    """
    Example 2 for the simulation with state dependent rates
    """
    rates = {(i, j): 0.05 if i < 4 else 1 for i in range(10) for j in range(10)}
    simulation = simulate_model(
        lambda_2=0.1,
        lambda_1=0.5,
        mu=rates,
        num_of_servers=8,
        threshold=4,
        seed_num=0,
        runtime=100,
    )

    assert (
        round(
            sum([w.waiting_time for w in simulation.get_all_records()]),
            NUMBER_OF_DIGITS_TO_ROUND,
        )
        == round(20.192189452374485, NUMBER_OF_DIGITS_TO_ROUND)
    )
    assert (
        round(
            sum([b.time_blocked for b in simulation.get_all_records()]),
            NUMBER_OF_DIGITS_TO_ROUND,
        )
        == round(229.48684030917272, NUMBER_OF_DIGITS_TO_ROUND)
    )
    assert (
        round(
            sum([s.service_time for s in simulation.get_all_records()]),
            NUMBER_OF_DIGITS_TO_ROUND,
        )
        == round(497.47902606711347, NUMBER_OF_DIGITS_TO_ROUND)
    )


def test_simulate_state_dependent_model_when_threshold_more_than_system_capacity():
    """
    Tests the following scenarios where specific cases occur:
        - when buffer_capacity is less than 1 -> an error is raised
        - when threshold is greater than system capacity the
          model forces threshold=system_capacity and buffer_capacity=1
    """
    sim_results_normal = []
    sim_results_forced = []
    for seed in range(5):
        simulation = simulate_model(
            lambda_2=0.15,
            lambda_1=0.2,
            mu={(i, j): 0.05 for i in range(10) for j in range(10)},
            num_of_servers=8,
            threshold=10,
            seed_num=seed,
            system_capacity=10,
            buffer_capacity=1,
            runtime=100,
        )
        rec = simulation.get_all_records()
        sim_results_normal.append(rec)

    for seed in range(5):
        simulation = simulate_model(
            lambda_2=0.15,
            lambda_1=0.2,
            mu={(i, j): 0.05 for i in range(10) for j in range(10)},
            num_of_servers=8,
            threshold=12,
            seed_num=seed,
            system_capacity=10,
            buffer_capacity=5,
            runtime=100,
        )
        rec = simulation.get_all_records()
        sim_results_forced.append(rec)
    assert sim_results_normal == sim_results_forced


def test_simulate_state_dependent_model_when_buffer_capacity_less_than_1():
    """
    Test that an error is raised when buffer_capacity is less than 1
    """
    with pytest.raises(ValueError):
        simulate_model(
            lambda_2=0.15,
            lambda_1=0.2,
            mu=None,
            num_of_servers=8,
            threshold=4,
            seed_num=0,
            system_capacity=10,
            buffer_capacity=0,
        )


def test_simulate_state_dependent_model_for_negative_and_0_rates():
    """
    Test that an error is raised when rates are negative or 0
    """
    with pytest.raises(ValueError):
        simulate_model(
            lambda_2=0.15,
            lambda_1=0.2,
            mu={(i, j): -0.05 for i in range(10) for j in range(10)},
            num_of_servers=8,
            threshold=4,
            system_capacity=10,
            buffer_capacity=3,
        )

    with pytest.raises(ValueError):
        simulate_model(
            lambda_2=0.15,
            lambda_1=0.2,
            mu={(i, j): 0 for i in range(10) for j in range(10)},
            num_of_servers=8,
            threshold=4,
            system_capacity=10,
            buffer_capacity=3,
        )


@given(
    lambda_2=floats(min_value=0.1, max_value=1.0),
    lambda_1=floats(min_value=0.1, max_value=1.0),
    mu=floats(min_value=0.5, max_value=2.0),
    num_of_servers=integers(min_value=1, max_value=10),
)
def test_server_dependent_model_property_based(lambda_2, lambda_1, mu, num_of_servers):
    """
    Example 1 for the simulation with server dependent rates
    """
    simulation = simulate_model(
        lambda_2=lambda_2,
        lambda_1=lambda_1,
        mu=mu,
        num_of_servers=num_of_servers,
        threshold=4,
        seed_num=0,
        runtime=100,
    )

    server_dependent_simulation = simulate_model(
        lambda_2=0.1,
        lambda_1=0.5,
        mu={k: 0.3 for k in range(1, 4)},
        num_of_servers=3,
        threshold=4,
        seed_num=0,
        runtime=100,
    )

    assert round(
        sum([w.waiting_time for w in simulation.get_all_records()]),
        NUMBER_OF_DIGITS_TO_ROUND,
    ) == round(
        sum([w.waiting_time for w in server_dependent_simulation.get_all_records()]),
        NUMBER_OF_DIGITS_TO_ROUND,
    )
    assert round(
        sum([b.time_blocked for b in simulation.get_all_records()]),
        NUMBER_OF_DIGITS_TO_ROUND,
    ) == round(
        sum([b.time_blocked for b in server_dependent_simulation.get_all_records()]),
        NUMBER_OF_DIGITS_TO_ROUND,
    )
    assert round(
        sum([s.service_time for s in simulation.get_all_records()]),
        NUMBER_OF_DIGITS_TO_ROUND,
    ) == round(
        sum([s.service_time for s in server_dependent_simulation.get_all_records()]),
        NUMBER_OF_DIGITS_TO_ROUND,
    )
