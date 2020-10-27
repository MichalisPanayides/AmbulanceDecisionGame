from ambulance_game.markov.waiting import (
    get_mean_waiting_time_using_markov_state_probabilities,
)

number_of_digits_to_round = 8


def test_get_mean_waiting_time_recursively_markov_example_1():
    """
    Example on getting the mean waiting time recursively from the Markov chain
    """
    mean_waiting_time = get_mean_waiting_time_using_markov_state_probabilities(
        lambda_2=0.2,
        lambda_1=0.2,
        mu=0.2,
        num_of_servers=3,
        threshold=4,
        system_capacity=10,
        buffer_capacity=10,
        patient_type="others",
        formula="recursive",
    )
    assert round(mean_waiting_time, number_of_digits_to_round) == 1.47207167


def test_get_mean_waiting_time_recursively_markov_example_2():
    """
    Example on getting the mean waiting time recursively from the Markov chain
    """
    mean_waiting_time = get_mean_waiting_time_using_markov_state_probabilities(
        lambda_2=0.2,
        lambda_1=0.2,
        mu=0.2,
        num_of_servers=3,
        threshold=4,
        system_capacity=10,
        buffer_capacity=10,
        patient_type="ambulance",
        formula="recursive",
    )
    assert round(mean_waiting_time, number_of_digits_to_round) == 0.73779145


def test_get_mean_waiting_time_recursively_markov_example_3():
    """
    Example on getting the mean waiting time recursively from the Markov chain
    """
    mean_waiting_time = get_mean_waiting_time_using_markov_state_probabilities(
        lambda_2=0.2,
        lambda_1=0.2,
        mu=0.2,
        num_of_servers=3,
        threshold=3,
        system_capacity=10,
        buffer_capacity=10,
        patient_type="ambulance",
        formula="recursive",
    )
    assert mean_waiting_time == 0


def test_get_mean_waiting_time_recursively_markov_example_4():
    """
    Example on getting the mean waiting time recursively from the Markov chain
    """
    mean_waiting_time = get_mean_waiting_time_using_markov_state_probabilities(
        lambda_2=0.2,
        lambda_1=0.2,
        mu=0.2,
        num_of_servers=3,
        threshold=4,
        system_capacity=10,
        buffer_capacity=10,
        patient_type="both",
        formula="recursive",
    )
    assert round(mean_waiting_time, number_of_digits_to_round) == round(
        1.1051493390764142, number_of_digits_to_round
    )


def test_get_mean_waiting_time_from_closed_form_markov_example_1():
    """
    Example on getting the mean waiting time from a closed form formula
    """
    mean_waiting_time = get_mean_waiting_time_using_markov_state_probabilities(
        lambda_2=0.2,
        lambda_1=0.2,
        mu=0.2,
        num_of_servers=3,
        threshold=4,
        system_capacity=10,
        buffer_capacity=10,
        patient_type="others",
        formula="closed_form",
    )
    assert round(mean_waiting_time, number_of_digits_to_round) == 1.47207167


def test_get_mean_waiting_time_from_closed_form_markov_example_2():
    """
    Example on getting the mean waiting time from a closed form formula
    """
    mean_waiting_time = get_mean_waiting_time_using_markov_state_probabilities(
        lambda_2=0.2,
        lambda_1=0.2,
        mu=0.2,
        num_of_servers=3,
        threshold=4,
        system_capacity=10,
        buffer_capacity=10,
        patient_type="ambulance",
        formula="closed_form",
    )
    assert round(mean_waiting_time, number_of_digits_to_round) == 0.73779145


def test_get_mean_waiting_time_from_closed_form_markov_example_3():
    """
    Example on getting the mean waiting time from a closed form formula
    """
    mean_waiting_time = get_mean_waiting_time_using_markov_state_probabilities(
        lambda_2=0.2,
        lambda_1=0.2,
        mu=0.2,
        num_of_servers=3,
        threshold=3,
        system_capacity=10,
        buffer_capacity=10,
        patient_type="ambulance",
        formula="closed_form",
    )
    assert mean_waiting_time == 0


def test_get_mean_waiting_time_from_closed_form_markov_example_4():
    """
    Example on getting the mean waiting time from a closed form formula
    """
    mean_waiting_time = get_mean_waiting_time_using_markov_state_probabilities(
        lambda_2=0.2,
        lambda_1=0.2,
        mu=0.2,
        num_of_servers=3,
        threshold=4,
        system_capacity=10,
        buffer_capacity=10,
        patient_type="both",
        formula="closed_form",
    )
    assert round(mean_waiting_time, number_of_digits_to_round) == round(
        1.1051493390764142, number_of_digits_to_round
    )
