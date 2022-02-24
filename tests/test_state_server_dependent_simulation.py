"""
Tests for the state and server dependent part of the simulation
"""
import numpy as np
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
@settings(max_examples=10, deadline=None)
def test_compare_state_dependent_model_with_non_state_dependent_property_based(
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

    assert round(
        sum([w.waiting_time for w in simulation.get_all_records()]),
        NUMBER_OF_DIGITS_TO_ROUND,
    ) == round(69.05359560579672, NUMBER_OF_DIGITS_TO_ROUND)
    assert round(
        sum([b.time_blocked for b in simulation.get_all_records()]),
        NUMBER_OF_DIGITS_TO_ROUND,
    ) == round(1.8837534828730575, NUMBER_OF_DIGITS_TO_ROUND)
    assert round(
        sum([s.service_time for s in simulation.get_all_records()]),
        NUMBER_OF_DIGITS_TO_ROUND,
    ) == round(130.39705479506074, NUMBER_OF_DIGITS_TO_ROUND)


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

    assert round(
        sum([w.waiting_time for w in simulation.get_all_records()]),
        NUMBER_OF_DIGITS_TO_ROUND,
    ) == round(20.192189452374485, NUMBER_OF_DIGITS_TO_ROUND)
    assert round(
        sum([b.time_blocked for b in simulation.get_all_records()]),
        NUMBER_OF_DIGITS_TO_ROUND,
    ) == round(229.48684030917272, NUMBER_OF_DIGITS_TO_ROUND)
    assert round(
        sum([s.service_time for s in simulation.get_all_records()]),
        NUMBER_OF_DIGITS_TO_ROUND,
    ) == round(497.47902606711347, NUMBER_OF_DIGITS_TO_ROUND)


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
@settings(max_examples=10, deadline=None)
def test_compare_server_dependent_model_with_non_state_dependent_property_based(
    lambda_2, lambda_1, mu, num_of_servers
):
    """
    Property based test for the simulation with server dependent rates. For
    different values of lambda_1, lambda_2, mu and num_of_servers checks
    that the simulation outputs the same results when:
        - mu = mu
        - mu = {k: mu for k in range(1, num_of_servers + 1)}
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
        lambda_2=lambda_2,
        lambda_1=lambda_1,
        mu={k: mu for k in range(1, num_of_servers + 1)},
        num_of_servers=num_of_servers,
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


def test_server_dependent_simulation_example_1():
    """
    Example 1 for server dependent simulation
    """
    server_dependent_simulation = simulate_model(
        lambda_1=4,
        lambda_2=2,
        mu={1: 1, 2: 1.5, 3: 2, 4: 2.5},
        num_of_servers=4,
        threshold=4,
        seed_num=0,
        runtime=100,
    )

    mean_wait = np.mean(
        [w.waiting_time for w in server_dependent_simulation.get_all_records()]
    )
    mean_block = np.mean(
        [b.time_blocked for b in server_dependent_simulation.get_all_records()]
    )
    mean_service = np.mean(
        [s.service_time for s in server_dependent_simulation.get_all_records()]
    )

    assert round(mean_wait, NUMBER_OF_DIGITS_TO_ROUND) == round(
        0.11021530720155796, NUMBER_OF_DIGITS_TO_ROUND
    )
    assert round(mean_block, NUMBER_OF_DIGITS_TO_ROUND) == round(
        0.49569480054255816, NUMBER_OF_DIGITS_TO_ROUND
    )
    assert round(mean_service, NUMBER_OF_DIGITS_TO_ROUND) == round(
        0.4326597402401585, NUMBER_OF_DIGITS_TO_ROUND
    )


def test_server_dependent_simulation_example_2():
    """
    Example 2 for server dependent simulation
    """
    server_dependent_simulation = simulate_model(
        lambda_1=10,
        lambda_2=20,
        mu={1: 1, 2: 1.5, 3: 2, 4: 2.5, 5: 3, 6: 3.5, 7: 4, 8: 4.5, 9: 5, 10: 5.5},
        num_of_servers=10,
        threshold=15,
        system_capacity=20,
        buffer_capacity=5,
        seed_num=0,
        runtime=100,
    )

    mean_wait = np.mean(
        [w.waiting_time for w in server_dependent_simulation.get_all_records()]
    )
    mean_block = np.mean(
        [b.time_blocked for b in server_dependent_simulation.get_all_records()]
    )
    mean_service = np.mean(
        [s.service_time for s in server_dependent_simulation.get_all_records()]
    )

    assert round(mean_wait, NUMBER_OF_DIGITS_TO_ROUND) == round(
        0.0504065681590924, NUMBER_OF_DIGITS_TO_ROUND
    )
    assert round(mean_block, NUMBER_OF_DIGITS_TO_ROUND) == round(
        0.016470674877055978, NUMBER_OF_DIGITS_TO_ROUND
    )
    assert round(mean_service, NUMBER_OF_DIGITS_TO_ROUND) == round(
        0.1931311883483223, NUMBER_OF_DIGITS_TO_ROUND
    )


@given(
    lambda_2=floats(min_value=0.1, max_value=1.0),
    lambda_1=floats(min_value=0.1, max_value=1.0),
    mu=floats(min_value=2.0, max_value=3.0),
    num_of_servers=integers(min_value=1, max_value=10),
)
@settings(max_examples=5, deadline=None)
def test_compare_state_server_dependent_model_with_normal_property_based(
    lambda_2, lambda_1, mu, num_of_servers
):
    """
    Property based test for the simulation when using both state and server
    dependent rates. For different values of lambda_1, lambda_2, mu and
    num_of_servers checks that the simulation outputs the same results when:
        - mu = mu
        - mu = {k: {(i,j): mu}} -> dictionary of dictionaries
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

    rates = {}
    for server in range(1, num_of_servers + 1):
        rates[server] = {(u, v): mu for u in range(10) for v in range(10)}
    server_dependent_simulation = simulate_model(
        lambda_2=lambda_2,
        lambda_1=lambda_1,
        mu=rates,
        num_of_servers=num_of_servers,
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


def test_state_server_dependent_model_server_attributes():
    """
    Test that server objects in the simulation object get two new attributes
    when using the the StateServerDependentExponential class:
        - served_inds
        - service_times
    """
    num_of_servers = 4
    system_capacity = 10
    buffer_capacity = 10

    rates = {}
    for server in range(1, num_of_servers + 1):
        rates[server] = {
            (u, v): 1
            for u in range(buffer_capacity + 1)
            for v in range(system_capacity + 1)
        }

    simulation = simulate_model(
        lambda_2=2,
        lambda_1=2,
        mu=rates,
        num_of_servers=num_of_servers,
        threshold=8,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
        seed_num=0,
        runtime=100,
    )

    for server in simulation.nodes[2].servers:
        assert hasattr(server, "served_inds")
        assert hasattr(server, "service_times")
