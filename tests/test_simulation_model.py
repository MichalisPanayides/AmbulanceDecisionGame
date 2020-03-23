import numpy as np
import ciw

from hypothesis import given
from hypothesis.strategies import floats
from hypothesis.strategies import integers

from ambulance_game.models.simulation import (
    build_model,
    build_custom_node,
    simulate_model,
    get_multiple_runs_results,
)


@given(
    lambda_a=floats(min_value=0.1, max_value=10),
    lambda_o=floats(min_value=0.1, max_value=10),
    mu=floats(min_value=0.1, max_value=10),
    c=integers(min_value=1, max_value=20),
)
def test_build_model(lambda_a, lambda_o, mu, c):
    """
    Test to ensure consistent outcome type
    """
    result = build_model(lambda_a, lambda_o, mu, c)
    
    assert type(result) == ciw.network.Network


def test_example_model():
    """
    Test to ensure correct results to specific problem
    """
    ciw.seed(5)
    Q = ciw.Simulation(build_model(lambda_a=1, lambda_o=1, mu=2, total_capacity=1))
    Q.simulate_until_max_time(max_simulation_time=100)
    records = Q.get_all_records()
    wait = [r.waiting_time for r in records]
    blocks = [r.time_blocked for r in records]

    assert len(records) == 290
    assert sum(wait) == 1089.854729732795
    assert sum(blocks) == 0


@given(
    total_capacity=integers(min_value=1, max_value=20),
)
def test_build_custom_node(total_capacity):
    """
    Test to ensure blocking works as expected for extreme cases where the threshold is set to infinity and -1
    """
    ciw.seed(5)
    model_1 = ciw.Simulation(build_model(lambda_a=0.2, lambda_o=0.15, mu=0.05, total_capacity=total_capacity), node_class=build_custom_node(np.inf))
    model_1.simulate_until_max_time(max_simulation_time=100)
    records_1 = model_1.get_all_records()
    model_1_blocks = [r.time_blocked for r in records_1]
    model_1_waits = [r.waiting_time for r in records_1 if r.node == 1]

    model_2 = ciw.Simulation(build_model(lambda_a=0.2, lambda_o=0.15, mu=0.05, total_capacity=total_capacity), node_class=build_custom_node(-1))
    model_2.simulate_until_max_time(max_simulation_time=100)
    records_2 = model_2.get_all_records()
    model_2_blocks = [r.time_blocked for r in records_2 if r.node == 1]

    assert all(model_1_blocks) == 0
    assert all(model_1_waits) == 0
    assert all(model_2_blocks) != 0


def test_example_build_custom_node():
    """
    Test to ensure blocking occurs for specific case
    """
    ciw.seed(5)
    Q = ciw.Simulation(build_model(lambda_a=1, lambda_o=1, mu=2, total_capacity=1), node_class=build_custom_node(7))
    Q.simulate_until_max_time(max_simulation_time=100)
    records = Q.get_all_records()
    wait = [r.waiting_time for r in records]
    blocks = [r.time_blocked for r in records]

    assert len(records) == 274
    assert sum(wait) == 521.0071454616575
    assert sum(blocks) == 546.9988970370749


def test_simulate_model():
    """
    Test that correct amount of patients flow through the system given specific values
    """
    sim_results = []
    blocks = 0
    waits = 0
    services = 0
    for seed in range(5):
        simulation = simulate_model(
            lambda_a=0.15,
            lambda_o=0.2,
            mu=0.05,
            total_capacity=8,
            threshold=4,
            seed_num=seed,
        )
        rec = simulation.get_all_records()
        sim_results.append(rec)
        blocks = blocks + sum([b.time_blocked for b in rec])
        waits = waits + sum([w.waiting_time for w in rec])
        services = services + sum([s.service_time for s in rec])
        

    assert type(simulation) == ciw.simulation.Simulation
    assert len(sim_results[0]) == 474
    assert len(sim_results[1]) == 490
    assert len(sim_results[2]) == 491
    assert len(sim_results[3]) == 486
    assert len(sim_results[4]) == 458
    assert blocks == 171712.5200250419
    assert waits == 580.0884411214596
    assert services == 37134.74895651618


def test_get_multiple_results():
    mult_results_1 = get_multiple_runs_results(
        lambda_a=0.15,
        lambda_o=0.2,
        mu=0.05,
        total_capacity=8,
        threshold=4,
        num_of_trials=5,
        seed_num=1,
    )
    mult_results_2 = get_multiple_runs_results(
        lambda_a=0.15,
        lambda_o=0.2,
        mu=0.05,
        total_capacity=8,
        threshold=4,
        num_of_trials=5,
        seed_num=1,
        output_type="list"
    )
    assert type(mult_results_1) == list
    for trial in range(5):
        assert type(mult_results_1[trial]) != list
        assert type(mult_results_1[trial].waiting_times) == list
        assert type(mult_results_1[trial].service_times) == list
        assert type(mult_results_1[trial].blocking_times) == list

    assert type(mult_results_2) == list
    for times in range(3): 
        for trial in range(5):
            assert type(mult_results_2[times][trial]) == list


def test_example_get_multiple_results():
    """
    Test that multiple results function works with specific values
    """
    mult_results = get_multiple_runs_results(
        lambda_a=0.15,
        lambda_o=0.2,
        mu=0.05,
        total_capacity=8,
        threshold=4,
        num_of_trials=10,
        seed_num=1,
    )
    all_waits = [np.mean(w.waiting_times) for w in mult_results]
    all_servs = [np.mean(s.service_times) for s in mult_results]
    all_blocks = [np.mean(b.blocking_times) for b in mult_results]

    assert np.mean(all_waits) == 0.3333489337233605
    assert np.mean(all_servs) == 15.94133405666149
    assert np.mean(all_blocks) == 77.92554258000573
