import ambulance_game as abg

NUMBER_OF_DIGITS_TO_ROUND = 8


def test_simulate_model_with_non_state_dependent_property_based():
    """
    Propert based test with state dependent service rate. Ensures that for
    different values of lambda_1, lambda_2, and threshold, the simulation
    """
    simulation = abg.simulation.simulate_model(
        lambda_2=0.15,
        lambda_1=0.2,
        mu=0.05,
        num_of_servers=8,
        threshold=4,
        seed_num=0,
        runtime=100,
        system_capacity=10,
        buffer_capacity=10,
    )

    rates = {(i, j): 0.05 for i in range(10) for j in range(10)}
    simulation_extension = abg.simulation.simulate_state_dependent_model(
        lambda_2=0.15,
        lambda_1=0.2,
        rates=rates,
        num_of_servers=8,
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


def test_simulate_model_example_1():
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
    simulation = abg.simulation.simulate_state_dependent_model(
        lambda_2=0.15,
        lambda_1=0.2,
        rates=rates,
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


def test_simulate_model_example_2():
    """
    Example 2 for the simulation with state dependent rates
    """
    rates = {(i, j): 0.05 if i < 4 else 1 for i in range(10) for j in range(10)}
    simulation = abg.simulation.simulate_state_dependent_model(
        lambda_2=0.1,
        lambda_1=0.5,
        rates=rates,
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
