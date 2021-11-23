import ambulance_game as abg


def test_simulate_model_with_non_state_dependent_property_based():
    simulation = abg.simulation.simulate_model(
        lambda_2=0.15,
        lambda_1=0.2,
        mu=0.05,
        num_of_servers=8,
        threshold=4,
        seed_num=0,
    )
    simulation_extension = abg.server_simulation.simulate_model(
        lambda_2=0.15,
        lambda_1=0.2,
        mu=0.05,
        num_of_servers=8,
        threshold=4,
        seed_num=0,
        rates=None,
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


def test_simulate_model_example_1():
    """
    Example 1 for the simulation with state dependent rates
    """
    rates = {(0, 0): 0.2, (0, 1): 0.15, (1, 0): 0.15, (1, 1): 0.2}
    simulation = abg.server_simulation.simulate_model(
        lambda_2=0.15,
        lambda_1=0.2,
        mu=0.05,
        num_of_servers=8,
        threshold=4,
        seed_num=0,
        rates=rates,
    )

    assert sum([w.waiting_time for w in simulation.get_all_records()]) == 0.2
    assert sum([b.time_blocked for b in simulation.get_all_records()]) == 0.1
    assert sum([s.service_time for s in simulation.get_all_records()]) == 0.4


def test_simulate_model_example_2():
    """
    Example 2 for the simulation with state dependent rates
    """
    rates = {(0, 0): 0.2, (0, 1): 0.15, (1, 0): 0.15, (1, 1): 0.2}
    simulation = abg.server_simulation.simulate_model(
        lambda_2=0.15,
        lambda_1=0.2,
        mu=0.05,
        num_of_servers=8,
        threshold=4,
        seed_num=0,
        rates=rates,
    )

    assert sum([w.waiting_time for w in simulation.get_all_records()]) == 0.2
    assert sum([b.time_blocked for b in simulation.get_all_records()]) == 0.1
    assert sum([s.service_time for s in simulation.get_all_records()]) == 0.4
