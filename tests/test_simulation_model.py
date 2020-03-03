import ciw

from hypothesis import given
from hypothesis.strategies import floats
from hypothesis.strategies import integers

from ambulance_game import (
    build_model,
    build_custom_node,
    simulate_model,
)


@given(
    l_a=floats(min_value=0.1, max_value=10),
    l_o=floats(min_value=0.1, max_value=10),
    mu=floats(min_value=0.1, max_value=10),
    c=integers(min_value=1, max_value=20),
)
def test_build_model(l_a, l_o, mu, c):
    """
    Test to ensure consistent outcome type
    """
    result = build_model(l_a, l_o, mu, c)
    assert type(result) == ciw.network.Network
    # more specific examples
    # run black


def test_specific_model():
    """
    Test to ensure correct results to specific problem
    """
    ciw.seed(5)
    Q = ciw.Simulation(build_model(lambda_a=1, lambda_o=1, mu=2, total_capacity=1))
    Q.simulate_until_max_time(100)
    records = Q.get_all_records()
    wait = [r.waiting_time for r in records]
    blocks = [r.time_blocked for r in records]

    assert len(records) == 290
    assert sum(wait) == 1089.854729732795
    assert sum(blocks) == 0


def test_build_custom_node():
    """
    Test to ensure blocking occurs for specific case
    """
    ciw.seed(5)
    Q = ciw.Simulation(build_model(1, 1, 2, 1), node_class=build_custom_node(7))

    Q.simulate_until_max_time(100)
    records = Q.get_all_records()
    wait = [r.waiting_time for r in records]
    blocks = [r.time_blocked for r in records]

    assert len(records) == 290
    assert sum(wait) == 1026.2910789050652
    assert sum(blocks) == 66.03415121579033


def test_simulate_model():
    """
    Test that correct amount of patients flow through the system given specific values
    """
    sim_results = []
    for i in range(10):
        simulation = simulate_model(0.15, 0.2, 0.05, 8, 4, i)
        sim_results.append(len(simulation.get_all_records()))

    assert sim_results[0] == 705
    assert sim_results[1] == 647
    assert sim_results[2] == 712
    assert sim_results[3] == 747
    assert sim_results[4] == 681
    assert sim_results[5] == 738
    assert sim_results[6] == 699
    assert sim_results[7] == 724
    assert sim_results[8] == 744
    assert sim_results[9] == 700
