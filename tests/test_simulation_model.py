import numpy as np
import ciw

from hypothesis import given
from hypothesis.strategies import floats
from hypothesis.strategies import integers

from ambulance_game.models import (
    simulate_model,
    get_multiple_runs_results,
)


# @given(
#     lambda_a=floats(min_value=0.1, max_value=10),
#     lambda_o=floats(min_value=0.1, max_value=10),
#     mu=floats(min_value=0.1, max_value=10),
#     c=integers(min_value=1, max_value=20),
# )
# def test_build_model(lambda_a, lambda_o, mu, c):
#     """
#     Test to ensure consistent outcome type
#     """
#     result = build_model(lambda_a, lambda_o, mu, c)
    
#     assert type(result) == ciw.network.Network


# def test_specific_model():
#     """
#     Test to ensure correct results to specific problem
#     """
#     ciw.seed(5)
#     Q = ciw.Simulation(build_model(lambda_a=1, lambda_o=1, mu=2, total_capacity=1))
#     Q.simulate_until_max_time(max_simulation_time=100)
#     records = Q.get_all_records()
#     wait = [r.waiting_time for r in records]
#     blocks = [r.time_blocked for r in records]

#     assert len(records) == 290
#     assert sum(wait) == 1089.854729732795
#     assert sum(blocks) == 0


# def test_build_custom_node():
#     """
#     Test to ensure blocking occurs for specific case
#     """
#     ciw.seed(5)
#     Q = ciw.Simulation(build_model(lambda_a=1, lambda_o=1, mu=2, total_capacity=1), node_class=build_custom_node(7))
#     Q.simulate_until_max_time(max_simulation_time=100)
#     records = Q.get_all_records()
#     wait = [r.waiting_time for r in records]
#     blocks = [r.time_blocked for r in records]

#     assert len(records) == 275
#     assert sum(wait) == 560.0768614256764
#     assert sum(blocks) == 522.1466827678948


def test_simulate_model():
    """
    Test that correct amount of patients flow through the system given specific values
    """
    sim_results = []
    for i in range(10):
        simulation = simulate_model(
            lambda_a=0.15,
            lambda_o=0.2,
            mu=0.05,
            total_capacity=8,
            threshold=4,
            seed_num=i,
        )
        sim_results.append(len(simulation.get_all_records()))

    assert sim_results[0] == 474
    assert sim_results[1] == 490
    assert sim_results[2] == 491
    assert sim_results[3] == 486
    assert sim_results[4] == 458
    assert sim_results[5] == 539
    assert sim_results[6] == 453
    assert sim_results[7] == 470
    assert sim_results[8] == 492
    assert sim_results[9] == 483


def test_get_multiple_results():
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

    assert type(mult_results) == list
    assert np.mean(all_waits) == 0.3333489337233605
    assert np.mean(all_servs) == 15.94133405666149
    assert np.mean(all_blocks) == 77.92554258000573
