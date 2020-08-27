import pytest
import numpy as np
import ciw

from hypothesis import (
    given,
    settings,
)
from hypothesis.strategies import (
    floats,
    integers,
)


from ambulance_game.simulation.simulation import (
    build_model,
    build_custom_node,
    simulate_model,
    get_simulated_state_probabilities,
    get_average_simulated_state_probabilities,
    get_multiple_runs_results,
    get_mean_blocking_difference_between_two_hospitals,
    calculate_ambulance_best_response,
)

number_of_digits_to_round = 8


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
    Test to ensure that the correct results are output to a specific problem
    """
    ciw.seed(5)
    Q = ciw.Simulation(build_model(lambda_a=1, lambda_o=1, mu=2, num_of_servers=1))
    Q.simulate_until_max_time(max_simulation_time=100)
    records = Q.get_all_records()
    wait = [r.waiting_time for r in records]
    blocks = [r.time_blocked for r in records]

    assert len(records) == 290
    assert round(sum(wait), number_of_digits_to_round) == round(
        1089.854729732795, number_of_digits_to_round
    )
    assert sum(blocks) == 0


@given(
    num_of_servers=integers(min_value=1, max_value=20),
)
def test_build_custom_node(num_of_servers):
    """
    Test to ensure blocking works as expected for extreme cases where the threshold is set to infinity and -1
    """
    ciw.seed(5)
    model_1 = ciw.Simulation(
        build_model(
            lambda_a=0.2, lambda_o=0.15, mu=0.05, num_of_servers=num_of_servers
        ),
        node_class=build_custom_node(np.inf),
    )
    model_1.simulate_until_max_time(max_simulation_time=100)
    records_1 = model_1.get_all_records()
    model_1_blocks = [r.time_blocked for r in records_1]
    model_1_waits = [r.waiting_time for r in records_1 if r.node == 1]

    model_2 = ciw.Simulation(
        build_model(
            lambda_a=0.2, lambda_o=0.15, mu=0.05, num_of_servers=num_of_servers
        ),
        node_class=build_custom_node(-1),
    )
    model_2.simulate_until_max_time(max_simulation_time=100)
    records_2 = model_2.get_all_records()
    model_2_blocks = [r.time_blocked for r in records_2 if r.node == 1]

    assert all(b == 0 for b in model_1_blocks)
    assert all(w == 0 for w in model_1_waits)
    assert len(model_2_blocks) == 0


def test_example_build_custom_node():
    """
    Test to ensure blocking occurs for specific case
    """
    ciw.seed(5)
    Q = ciw.Simulation(
        build_model(lambda_a=1, lambda_o=1, mu=2, num_of_servers=1),
        node_class=build_custom_node(7),
    )
    Q.simulate_until_max_time(max_simulation_time=100)
    records = Q.get_all_records()
    wait = [r.waiting_time for r in records]
    blocks = [r.time_blocked for r in records]

    assert len(records) == 274
    assert round(sum(wait), number_of_digits_to_round) == round(
        521.0071454616575, number_of_digits_to_round
    )
    assert round(sum(blocks), number_of_digits_to_round) == round(
        546.9988970370749, number_of_digits_to_round
    )


def test_simulate_model_unconstrained():
    """
    Test that the crorect values are output given specific values and whhen the system capacity and the parking capacity are infinite
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
            num_of_servers=8,
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
    assert round(blocks, number_of_digits_to_round) == round(
        171712.5200250419, number_of_digits_to_round
    )
    assert round(waits, number_of_digits_to_round) == round(
        580.0884411214596, number_of_digits_to_round
    )
    assert round(services, number_of_digits_to_round) == round(
        37134.74895651618, number_of_digits_to_round
    )


def test_simulate_model_constrained():
    """
    Test that correct amount of patients flow through the system given specific values with a specified capacity of the hospital and the parking space
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
            num_of_servers=8,
            threshold=4,
            seed_num=seed,
            system_capacity=10,
            parking_capacity=5,
        )
        rec = simulation.get_all_records()
        sim_results.append(rec)
        blocks = blocks + sum([b.time_blocked for b in rec])
        waits = waits + sum([w.waiting_time for w in rec])
        services = services + sum([s.service_time for s in rec])

    assert type(simulation) == ciw.simulation.Simulation
    assert len(sim_results[0]) == 504
    assert len(sim_results[1]) == 449
    assert len(sim_results[2]) == 466
    assert len(sim_results[3]) == 453
    assert len(sim_results[4]) == 437
    assert round(blocks, number_of_digits_to_round) == round(
        27926.12213659, number_of_digits_to_round
    )
    assert round(waits, number_of_digits_to_round) == round(
        253.16063529519664, number_of_digits_to_round
    )
    assert round(services, number_of_digits_to_round) == round(
        36826.38173021053, number_of_digits_to_round
    )


def test_simulate_model_invalid_arguements():
    """
    Tests the following scenarios where specific cases occus:
        - when parking_capacity is less than 1 -> an error is raised
        - when threshold is greater than system capacity the
          model forces threshold=system_capacity and parking_capacity=1
    """
    sim_results_normal = []
    sim_results_forced = []
    for seed in range(5):
        simulation = simulate_model(
            lambda_a=0.15,
            lambda_o=0.2,
            mu=0.05,
            num_of_servers=8,
            threshold=10,
            seed_num=seed,
            system_capacity=10,
            parking_capacity=1,
        )
        rec = simulation.get_all_records()
        sim_results_normal.append(rec)

    for seed in range(5):
        simulation = simulate_model(
            lambda_a=0.15,
            lambda_o=0.2,
            mu=0.05,
            num_of_servers=8,
            threshold=12,
            seed_num=seed,
            system_capacity=10,
            parking_capacity=5,
        )
        rec = simulation.get_all_records()
        sim_results_forced.append(rec)

    assert sim_results_normal == sim_results_forced
    with pytest.raises(ValueError):
        simulate_model(
            lambda_a=0.15,
            lambda_o=0.2,
            mu=0.05,
            num_of_servers=8,
            threshold=4,
            seed_num=0,
            system_capacity=10,
            parking_capacity=0,
        )


def test_get_state_probabilities_dict():
    """
    Test to ensure that sum of the values of the pi dictionary equate to 1
    """
    lambda_a = 0.1
    lambda_o = 0.2
    mu = 0.2
    num_of_servers = 3
    threshold = 3
    system_capacity = 5
    parking_capacity = 4
    seed_num = None
    runtime = 2000
    tracker = ciw.trackers.NodePopulation()

    Q = simulate_model(
        lambda_a,
        lambda_o,
        mu,
        num_of_servers,
        threshold,
        seed_num,
        runtime,
        system_capacity,
        parking_capacity,
        tracker=tracker,
    )
    pi_dictionary = get_simulated_state_probabilities(simulation_object=Q, output=dict)
    assert round(sum(pi_dictionary.values()), number_of_digits_to_round) == 1


def test_get_state_probabilities_array():
    """
    Test to ensure that the sum of elements of the pi array equate to 1
    """
    lambda_a = 0.1
    lambda_o = 0.2
    mu = 0.2
    num_of_servers = 3
    threshold = 3
    system_capacity = 5
    parking_capacity = 4
    seed_num = None
    runtime = 2000
    tracker = ciw.trackers.NodePopulation()

    Q = simulate_model(
        lambda_a,
        lambda_o,
        mu,
        num_of_servers,
        threshold,
        seed_num,
        runtime,
        system_capacity,
        parking_capacity,
        tracker=tracker,
    )
    pi_array = get_simulated_state_probabilities(simulation_object=Q, output=np.ndarray)
    assert round(np.nansum(pi_array), number_of_digits_to_round) == 1


def test_get_average_state_probabilities_array():
    """
    Test to ensure that the sum of elements of the average pi array equate to 1
    """
    lambda_a = 0.1
    lambda_o = 0.2
    mu = 0.2
    num_of_servers = 3
    threshold = 3
    system_capacity = 5
    parking_capacity = 4
    seed_num = None
    runtime = 2000
    num_of_trials = 5

    pi_array = get_average_simulated_state_probabilities(
        lambda_a,
        lambda_o,
        mu,
        num_of_servers,
        threshold,
        system_capacity,
        parking_capacity,
        seed_num,
        runtime,
        num_of_trials,
        # output=np.ndarray,
    )

    assert round(np.nansum(pi_array), number_of_digits_to_round) == 1


def test_get_multiple_results():
    mult_results_1 = get_multiple_runs_results(
        lambda_a=0.15,
        lambda_o=0.2,
        mu=0.05,
        num_of_servers=8,
        threshold=4,
        num_of_trials=5,
        seed_num=1,
    )
    mult_results_2 = get_multiple_runs_results(
        lambda_a=0.15,
        lambda_o=0.2,
        mu=0.05,
        num_of_servers=8,
        threshold=4,
        num_of_trials=5,
        seed_num=1,
        output_type="list",
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
        num_of_servers=8,
        threshold=4,
        num_of_trials=10,
        seed_num=1,
    )
    all_waits = [np.mean(w.waiting_times) for w in mult_results]
    all_servs = [np.mean(s.service_times) for s in mult_results]
    all_blocks = [np.mean(b.blocking_times) for b in mult_results]

    assert round(np.mean(all_waits), number_of_digits_to_round) == round(
        0.40499090339103355, number_of_digits_to_round
    )
    assert round(np.mean(all_servs), number_of_digits_to_round) == round(
        19.47582689268173, number_of_digits_to_round
    )
    assert round(np.mean(all_blocks), number_of_digits_to_round) == round(
        432.68444649763916, number_of_digits_to_round
    )


def test_example_get_multiple_results_for_different_patient_types():
    """
    Test that multiple results function works as expected for different patient types
    """

    mult_results = get_multiple_runs_results(
        lambda_a=0.15,
        lambda_o=0.2,
        mu=0.05,
        num_of_servers=8,
        threshold=4,
        num_of_trials=10,
        seed_num=1,
        patient_type="both",
    )

    all_servs = [np.mean(s.service_times) for s in mult_results]
    all_blocks = [np.mean(b.blocking_times) for b in mult_results]

    mult_results = get_multiple_runs_results(
        lambda_a=0.15,
        lambda_o=0.2,
        mu=0.05,
        num_of_servers=8,
        threshold=4,
        num_of_trials=10,
        seed_num=1,
        patient_type="ambulance",
    )

    all_waits_ambulance = [np.mean(w.waiting_times) for w in mult_results]
    all_servs_ambulance = [np.mean(s.service_times) for s in mult_results]
    all_blocks_ambulance = [np.mean(b.blocking_times) for b in mult_results]

    mult_results = get_multiple_runs_results(
        lambda_a=0.15,
        lambda_o=0.2,
        mu=0.05,
        num_of_servers=8,
        threshold=4,
        num_of_trials=10,
        seed_num=1,
        patient_type="others",
    )
    all_waits_other = [np.mean(w.waiting_times) for w in mult_results]
    all_servs_other = [np.mean(s.service_times) for s in mult_results]
    all_blocks_other = [np.mean(b.blocking_times) for b in mult_results]

    assert all(w == 0 for w in all_waits_ambulance)
    assert int(np.mean(all_servs_ambulance)) == int(np.mean(all_servs))
    assert all_blocks_ambulance == all_blocks

    assert round(np.mean(all_waits_other), number_of_digits_to_round) == 0.53027998
    assert int(np.mean(all_servs_other)) == int(np.mean(all_servs))
    assert all(np.isnan(b) for b in all_blocks_other)


@given(
    lambda_a=floats(min_value=0.1, max_value=0.4),
    lambda_o=floats(min_value=0.1, max_value=0.4),
)
@settings(deadline=None)
def test_get_mean_blocking_difference_between_two_hospitals_equal_split(
    lambda_a, lambda_o
):
    """
    Test that ensures that the function that finds the optimal distribution of ambulance patients in two identical hospitals returns a solution that corresponds to 50% of patients going to one hospital and 50% going to another. That means that the difference in the number of patients must be 0, and that is precisely what the function checks. This test runs the function with a proportion variable of 0.5 (meaning equally distributing the ambulances between the two hospitals) and ensures that the difference is 0, given any values of λ^α and λ^ο_1 = λ^ο_2 = λ^ο

    Note here that due to the ciw.seed() function it was possible to eliminate any randomness and make both hospitals identical, in terms of arrivals, services and any other stochasticity that the simulation models incorporates.
    """
    diff = get_mean_blocking_difference_between_two_hospitals(
        prop_1=0.5,
        lambda_a=lambda_a,
        lambda_o_1=lambda_o,
        lambda_o_2=lambda_o,
        mu_1=0.2,
        mu_2=0.2,
        num_of_servers_1=4,
        num_of_servers_2=4,
        threshold_1=3,
        threshold_2=3,
        seed_num_1=2,
        seed_num_2=2,
        num_of_trials=5,
        warm_up_time=100,
        runtime=500,
    )
    assert diff == 0


# TODO Investigate making this a property based test
def test_get_mean_blocking_difference_between_two_hospitals_increasing():
    """Ensuring that the function is increasing for specific inputs"""
    diff_list = []
    proportions = np.linspace(0.1, 0.9, 9)
    for prop in proportions:
        diff_list.append(
            get_mean_blocking_difference_between_two_hospitals(
                prop_1=prop,
                lambda_a=0.15,
                lambda_o_1=0.08,
                lambda_o_2=0.08,
                mu_1=0.05,
                mu_2=0.05,
                num_of_servers_1=6,
                num_of_servers_2=6,
                threshold_1=5,
                threshold_2=5,
                seed_num_1=2,
                seed_num_2=2,
                num_of_trials=100,
                warm_up_time=100,
                runtime=500,
            )
        )
    print(diff_list)
    is_increasing = all(x <= y for x, y in zip(diff_list, diff_list[1:]))
    assert is_increasing


#  TODO Investigate making it a property based test
def test_calculate_ambulance_best_response_equal_split():
    """Make sure that the brenq() function that is used suggests that when two identical hospitals are considered the patients will be split equally between them (50% - 50%)

    Note here that due to the ciw.seed() function it was possible to eliminate any randomness and make both hospitals identical, in terms of arrivals, services and any other stochasticity that the simulation models incorporates.
    """
    lambda_a = 0.3
    equal_split = calculate_ambulance_best_response(
        lambda_a=lambda_a,
        lambda_o_1=0.3,
        lambda_o_2=0.3,
        mu_1=0.2,
        mu_2=0.2,
        num_of_servers_1=4,
        num_of_servers_2=4,
        threshold_1=3,
        threshold_2=3,
        seed_num_1=10,
        seed_num_2=10,
        num_of_trials=5,
        warm_up_time=100,
        runtime=500,
    )

    assert np.isclose(equal_split, 0.5)
