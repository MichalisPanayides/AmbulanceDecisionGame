import numpy as np

from ambulance_game.markov.proportion import (
    general_psi_function,
    specific_psi_function,
    hypoexponential_cdf,
    erlang_cdf,
    get_probability_of_waiting_time_in_system_less_than_target_for_state,
    get_proportion_of_individuals_within_time_target,
    overall_proportion_of_individuals_within_time_target,
    proportion_within_target_using_markov_state_probabilities,
)


number_of_digits_to_round = 8


def test_general_psi_function_examples():
    assert (
        general_psi_function(arg=1, k=1, l=2, exp_rates=(0, 6, 3), freq=(1, 10, 1), a=2)
        == 5 / 16
    )

    assert (
        round(
            general_psi_function(
                arg=5, k=1, l=4, exp_rates=(0, 8, 4), freq=(1, 10, 1), a=2
            ),
            number_of_digits_to_round,
        )
        == round(0.0021713763145861913, number_of_digits_to_round)
    )

    assert (
        round(
            general_psi_function(
                arg=2, k=1, l=7, exp_rates=(0, 9, 3), freq=(1, 15, 1), a=2
            ),
            number_of_digits_to_round,
        )
        == round(-1.8719280000000, number_of_digits_to_round)
    )

    assert (
        round(
            general_psi_function(
                arg=0.0001, k=2, l=1, exp_rates=(0, 4, 2), freq=(1, 10, 1), a=2
            ),
            number_of_digits_to_round,
        )
        == round(-0.009534359306064256, number_of_digits_to_round)
    )


def test_specific_psi_function_examples():
    assert (
        specific_psi_function(
            arg=1, k=1, l=2, exp_rates=(0, 6, 3), freq=(1, 10, 1), a=2
        )
        == 5 / 16
    )
    assert (
        round(
            specific_psi_function(
                arg=5, k=1, l=4, exp_rates=(0, 8, 4), freq=(1, 10, 1), a=2
            ),
            number_of_digits_to_round,
        )
        == round(0.0021713763145861913, number_of_digits_to_round)
    )

    assert (
        round(
            specific_psi_function(
                arg=2, k=1, l=7, exp_rates=(0, 9, 3), freq=(1, 15, 1), a=2
            ),
            number_of_digits_to_round,
        )
        == round(-1.8719280000000, number_of_digits_to_round)
    )
    assert (
        round(
            specific_psi_function(
                arg=0.0001, k=2, l=1, exp_rates=(0, 4, 2), freq=(1, 10, 1), a=2
            ),
            number_of_digits_to_round,
        )
        == round(-0.009534359306064256, number_of_digits_to_round)
    )


def test_compare_specific_and_general_psi_functions():
    for l in range(1, 20):
        assert round(
            general_psi_function(
                arg=10, k=1, l=l, exp_rates=(0, 6, 3), freq=(1, 10, 1), a=2
            ),
            number_of_digits_to_round,
        ) == round(
            specific_psi_function(
                arg=10, k=1, l=l, exp_rates=(0, 6, 3), freq=(1, 10, 1), a=2
            ),
            number_of_digits_to_round,
        )

        assert round(
            general_psi_function(
                arg=30, k=1, l=l, exp_rates=(0, 10, 2), freq=(1, 20, 1), a=2
            ),
            number_of_digits_to_round,
        ) == round(
            specific_psi_function(
                arg=30, k=1, l=l, exp_rates=(0, 10, 2), freq=(1, 20, 1), a=2
            ),
            number_of_digits_to_round,
        )


def test_hypoexponential_cdf_example_1():
    prob = hypoexponential_cdf(
        x=1, exp_rates=(8, 4), freq=(5, 1), psi_func=specific_psi_function
    )
    assert round(prob, number_of_digits_to_round) == round(
        0.6828287622623532, number_of_digits_to_round
    )


def test_hypoexponential_cdf_example_2():
    prob = hypoexponential_cdf(
        x=3, exp_rates=(6, 2), freq=(10, 1), psi_func=specific_psi_function
    )
    assert round(prob, number_of_digits_to_round) == round(
        0.876328452736831, number_of_digits_to_round
    )


def test_erlang_cdf_examples():
    assert round(erlang_cdf(mu=4, n=10, x=4), number_of_digits_to_round) == round(
        0.9567016840581342, number_of_digits_to_round
    )
    assert round(erlang_cdf(mu=2, n=20, x=8), number_of_digits_to_round) == round(
        0.18775147166316208, number_of_digits_to_round
    )
    assert round(erlang_cdf(mu=1, n=5, x=2), number_of_digits_to_round) == round(
        0.052653017343711084, number_of_digits_to_round
    )
    assert round(erlang_cdf(mu=5, n=3, x=7), number_of_digits_to_round) == round(
        0.9999999999995911, number_of_digits_to_round
    )


def test_get_probability_of_waiting_time_in_system_less_than_target_for_state_class_1():
    prob = get_probability_of_waiting_time_in_system_less_than_target_for_state(
        state=(5, 15),
        class_type=0,
        mu=2,
        num_of_servers=3,
        threshold=5,
        target=4,
        psi_func=specific_psi_function,
    )
    assert round(prob, number_of_digits_to_round) == round(
        0.9594781848165894, number_of_digits_to_round
    )


def test_get_probability_of_waiting_time_in_system_less_than_target_for_state_class_2():
    prob = get_probability_of_waiting_time_in_system_less_than_target_for_state(
        state=(6, 12),
        class_type=1,
        mu=2,
        num_of_servers=3,
        threshold=8,
        target=4,
        psi_func=specific_psi_function,
    )
    assert round(prob, number_of_digits_to_round) == round(
        0.9974529800821392, number_of_digits_to_round
    )


def test_get_proportion_of_individuals_within_time_target_class_1():
    all_states = [(0, 0), (0, 1), (0, 2), (1, 2), (0, 3), (1, 3)]
    pi = np.array([[0.1, 0.1, 0.1, 0.3], [np.nan, np.nan, 0.2, 0.2]])
    prop = get_proportion_of_individuals_within_time_target(
        all_states=all_states,
        pi=pi,
        class_type=0,
        mu=1,
        num_of_servers=1,
        threshold=2,
        system_capacity=3,
        buffer_capacity=1,
        target=4,
        psi_func=specific_psi_function,
    )
    assert round(prop, number_of_digits_to_round) == round(
        0.8351592500013925, number_of_digits_to_round
    )


def test_get_proportion_of_individuals_within_time_target_class_2():
    all_states = [(0, 0), (0, 1), (0, 2), (1, 2), (0, 3), (1, 3)]
    pi = np.array([[0.1, 0.1, 0.1, 0.3], [np.nan, np.nan, 0.2, 0.2]])
    prop = get_proportion_of_individuals_within_time_target(
        all_states=all_states,
        pi=pi,
        class_type=1,
        mu=1,
        num_of_servers=1,
        threshold=2,
        system_capacity=3,
        buffer_capacity=1,
        target=4,
        psi_func=specific_psi_function,
    )
    assert round(prop, number_of_digits_to_round) == round(
        0.9206322314821518, number_of_digits_to_round
    )


def test_overall_proportion_of_individuals_within_time_target_example():
    all_states = [(0, 0), (0, 1), (0, 2), (1, 2), (0, 3), (1, 3)]
    pi = np.array([[0.1, 0.1, 0.1, 0.3], [np.nan, np.nan, 0.2, 0.2]])
    prop = overall_proportion_of_individuals_within_time_target(
        all_states=all_states,
        pi=pi,
        lambda_1=2,
        lambda_2=2,
        mu=2.5,
        num_of_servers=2,
        threshold=2,
        system_capacity=3,
        buffer_capacity=1,
        target=1,
        psi_func=specific_psi_function,
    )
    assert round(prop, number_of_digits_to_round) == round(
        0.8973658054784248, number_of_digits_to_round
    )


def test_proportion_within_target_using_markov_state_probabilities_class_1():
    prop = proportion_within_target_using_markov_state_probabilities(
        lambda_1=2,
        lambda_2=2,
        mu=3,
        num_of_servers=2,
        threshold=2,
        system_capacity=3,
        buffer_capacity=1,
        class_type=0,
        target=1,
        psi_func=specific_psi_function,
    )
    assert round(prop, number_of_digits_to_round) == round(
        0.9336306352352214, number_of_digits_to_round
    )


def test_proportion_within_target_using_markov_state_probabilities_class_2():
    prop = proportion_within_target_using_markov_state_probabilities(
        lambda_1=2,
        lambda_2=2,
        mu=3,
        num_of_servers=2,
        threshold=2,
        system_capacity=3,
        buffer_capacity=1,
        class_type=1,
        target=1,
        psi_func=specific_psi_function,
    )
    assert round(prop, number_of_digits_to_round) == round(
        0.9502129316321362, number_of_digits_to_round
    )


def test_proportion_within_target_using_markov_state_probabilities_both_classes():
    prop = proportion_within_target_using_markov_state_probabilities(
        lambda_1=2,
        lambda_2=2,
        mu=3,
        num_of_servers=2,
        threshold=2,
        system_capacity=3,
        buffer_capacity=1,
        class_type=None,
        target=1,
        psi_func=specific_psi_function,
    )
    assert round(prop, number_of_digits_to_round) == round(
        0.9417472329452903, number_of_digits_to_round
    )
