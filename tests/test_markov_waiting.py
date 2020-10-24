from ambulance_game.markov.waiting import (
    get_mean_waiting_time_markov,
)

number_of_digits_to_round = 8


def test_get_mean_waiting_time_recursively_markov():
    """
    Examples on getting the mean waiting time recursively from the Markov chain
    """
    mean_waiting_time = get_mean_waiting_time_markov(
        lambda_a=0.2,
        lambda_o=0.2,
        mu=0.2,
        num_of_servers=3,
        threshold=4,
        system_capacity=10,
        parking_capacity=10,
        output="others",
        formula="recursive",
    )
    assert round(mean_waiting_time, number_of_digits_to_round) == 1.47207167

    mean_waiting_time = get_mean_waiting_time_markov(
        lambda_a=0.2,
        lambda_o=0.2,
        mu=0.2,
        num_of_servers=3,
        threshold=4,
        system_capacity=10,
        parking_capacity=10,
        output="ambulance",
        formula="recursive",
    )
    assert round(mean_waiting_time, number_of_digits_to_round) == 0.73779145

    mean_waiting_time = get_mean_waiting_time_markov(
        lambda_a=0.2,
        lambda_o=0.2,
        mu=0.2,
        num_of_servers=3,
        threshold=3,
        system_capacity=10,
        parking_capacity=10,
        output="ambulance",
        formula="recursive",
    )
    assert mean_waiting_time == 0

    mean_waiting_time = get_mean_waiting_time_markov(
        lambda_a=0.2,
        lambda_o=0.2,
        mu=0.2,
        num_of_servers=3,
        threshold=4,
        system_capacity=10,
        parking_capacity=10,
        output="both",
        formula="recursive",
    )
    assert round(mean_waiting_time, number_of_digits_to_round) == round(
        1.1051493390764142, number_of_digits_to_round
    )


def test_get_mean_waiting_time_from_closed_form_markov():
    """
    Examples on getting the mean waiting time from a closed form formula
    """
    mean_waiting_time = get_mean_waiting_time_markov(
        lambda_a=0.2,
        lambda_o=0.2,
        mu=0.2,
        num_of_servers=3,
        threshold=4,
        system_capacity=10,
        parking_capacity=10,
        output="others",
        formula="closed_form",
    )
    assert round(mean_waiting_time, number_of_digits_to_round) == 1.47207167

    mean_waiting_time = get_mean_waiting_time_markov(
        lambda_a=0.2,
        lambda_o=0.2,
        mu=0.2,
        num_of_servers=3,
        threshold=4,
        system_capacity=10,
        parking_capacity=10,
        output="ambulance",
        formula="closed_form",
    )
    assert round(mean_waiting_time, number_of_digits_to_round) == 0.73779145

    mean_waiting_time = get_mean_waiting_time_markov(
        lambda_a=0.2,
        lambda_o=0.2,
        mu=0.2,
        num_of_servers=3,
        threshold=3,
        system_capacity=10,
        parking_capacity=10,
        output="ambulance",
        formula="closed_form",
    )
    assert mean_waiting_time == 0

    mean_waiting_time = get_mean_waiting_time_markov(
        lambda_a=0.2,
        lambda_o=0.2,
        mu=0.2,
        num_of_servers=3,
        threshold=4,
        system_capacity=10,
        parking_capacity=10,
        output="both",
        formula="closed_form",
    )
    assert round(mean_waiting_time, number_of_digits_to_round) == round(
        1.1051493390764142, number_of_digits_to_round
    )
