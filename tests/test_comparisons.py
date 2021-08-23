"""
Tests for comparisons of the Markov model and the simulation
"""

import numpy as np
import pytest

from hypothesis import given, settings
from hypothesis.strategies import floats, integers

from ambulance_game.comparisons import (
    get_heatmaps,
    get_mean_blocking_time_from_simulation_state_probabilities,
    get_mean_waiting_time_from_simulation_state_probabilities,
    get_proportion_within_target_from_simulation_state_probabilities,
    plot_output_comparisons,
)

NUMBER_OF_DIGITS_TO_ROUND = 8


def test_get_heatmaps_example_1():
    """
    Test to ensure that the probabilities generated by the simulation and the
    Markov model are as expected.
    """
    heatmaps_probs = get_heatmaps(
        lambda_2=2,
        lambda_1=1,
        mu=2,
        num_of_servers=2,
        threshold=3,
        system_capacity=5,
        buffer_capacity=5,
        seed_num=0,
        runtime=100,
        num_of_trials=10,
        linear_positioning=False,
    )
    expected_probs = (
        np.array(
            [
                [0.15657134, 0.23662749, 0.16391817, 0.13420543, 0.02070944, 0.0036757],
                [np.nan, np.nan, np.nan, 0.08165133, 0.02249408, 0.00498913],
                [np.nan, np.nan, np.nan, 0.05124684, 0.01655216, 0.00379816],
                [np.nan, np.nan, np.nan, 0.03741792, 0.01048049, 0.00129502],
                [np.nan, np.nan, np.nan, 0.02189239, 0.00640466, 0.00116072],
                [np.nan, np.nan, np.nan, 0.01507139, 0.00871438, 0.00112376],
            ]
        ),
        np.array(
            [
                [
                    0.15459909,
                    0.23189863,
                    0.17392397,
                    0.13044298,
                    0.02059626,
                    0.00343271,
                ],
                [np.nan, np.nan, np.nan, 0.07723598, 0.01942191, 0.00438122],
                [np.nan, np.nan, np.nan, 0.05051955, 0.01503237, 0.0039658],
                [np.nan, np.nan, np.nan, 0.03475886, 0.01107021, 0.00316697],
                [np.nan, np.nan, np.nan, 0.02449802, 0.0080307, 0.00239411],
                [np.nan, np.nan, np.nan, 0.01746141, 0.00957775, 0.00359149],
            ]
        ),
        np.array(
            [
                [
                    0.00197225,
                    0.00472886,
                    -0.0100058,
                    0.00376245,
                    0.00011318,
                    0.00024299,
                ],
                [np.nan, np.nan, np.nan, 0.00441536, 0.00307217, 0.0006079],
                [np.nan, np.nan, np.nan, 0.00072728, 0.00151979, -0.00016765],
                [np.nan, np.nan, np.nan, 0.00265906, -0.00058972, -0.00187194],
                [np.nan, np.nan, np.nan, -0.00260564, -0.00162603, -0.00123339],
                [np.nan, np.nan, np.nan, -0.00239002, -0.00086337, -0.00246773],
            ]
        ),
    )

    assert np.allclose(heatmaps_probs, expected_probs, equal_nan=True)


def test_get_heatmaps_example_2():
    """
    Test to ensure that the probabilities generated by the simulation and the
    Markov model are as expected.
    """
    heatmaps_probs = get_heatmaps(
        lambda_2=1.5,
        lambda_1=1.5,
        mu=4,
        num_of_servers=1,
        threshold=2,
        system_capacity=6,
        buffer_capacity=1,
        seed_num=2,
        runtime=150,
        num_of_trials=5,
        linear_positioning=True,
    )

    expected_probs = (
        np.array(
            [
                [
                    0.31415055,
                    0.22936987,
                    0.17661768,
                    0.04897618,
                    0.01226239,
                    0.00191243,
                    0.00063125,
                ],
                [
                    np.nan,
                    np.nan,
                    0.09676506,
                    0.06857442,
                    0.0296508,
                    0.01747934,
                    0.00361002,
                ],
            ]
        ),
        np.array(
            [
                [
                    0.3236358,
                    0.24272685,
                    0.18204514,
                    0.04553079,
                    0.01141196,
                    0.00289688,
                    0.00079006,
                ],
                [
                    np.nan,
                    np.nan,
                    0.09100306,
                    0.05686228,
                    0.02698544,
                    0.01150214,
                    0.00460958,
                ],
            ]
        ),
        np.array(
            [
                [
                    -0.00948526,
                    -0.01335698,
                    -0.00542746,
                    0.00344539,
                    0.00085043,
                    -0.00098445,
                    -0.00015881,
                ],
                [
                    np.nan,
                    np.nan,
                    0.005762,
                    0.01171214,
                    0.00266535,
                    0.0059772,
                    -0.00099956,
                ],
            ]
        ),
    )

    assert np.allclose(heatmaps_probs, expected_probs, equal_nan=True)


def test_get_mean_waiting_time_from_simulation_state_probabilities():
    """
    Test for the mean waiting time using the Markov formula and the simulation
    state probabilities
    """
    mean_waiting_time = get_mean_waiting_time_from_simulation_state_probabilities(
        lambda_2=0.2,
        lambda_1=0.2,
        mu=0.2,
        num_of_servers=3,
        threshold=4,
        system_capacity=10,
        buffer_capacity=10,
        class_type=0,
        seed_num=0,
        runtime=2000,
        num_of_trials=1,
    )
    assert round(mean_waiting_time, NUMBER_OF_DIGITS_TO_ROUND) == round(
        1.3988142785295379, NUMBER_OF_DIGITS_TO_ROUND
    )


def test_get_mean_blocking_time_from_simulation_state_probabilities():
    """
    Test for the mean blocking time using the Markov formula and the simulation
    state probabilities
    """
    mean_blocking_time = get_mean_blocking_time_from_simulation_state_probabilities(
        lambda_2=5,
        lambda_1=6,
        mu=2,
        num_of_servers=7,
        threshold=5,
        system_capacity=15,
        buffer_capacity=7,
        seed_num=0,
        num_of_trials=1,
        runtime=1000,
    )
    assert round(mean_blocking_time, NUMBER_OF_DIGITS_TO_ROUND) == round(
        0.6247616245889802, NUMBER_OF_DIGITS_TO_ROUND
    )


def test_get_proportion_within_target_from_simulation_state_probabilities():
    """
    Test for the proportion of customers that are within the target waiting
    time using the Markov formula and the simulation state probabilities
    """
    mean_proportion = get_proportion_within_target_from_simulation_state_probabilities(
        lambda_1=1,
        lambda_2=1,
        mu=1,
        num_of_servers=3,
        threshold=7,
        system_capacity=10,
        buffer_capacity=5,
        target=4,
        class_type=0,
        seed_num=0,
        num_of_trials=2,
        runtime=100,
    )

    assert round(mean_proportion, NUMBER_OF_DIGITS_TO_ROUND) == round(
        0.9605868280871762, NUMBER_OF_DIGITS_TO_ROUND
    )


def test_plot_output_comparisons_waiting_class_1():
    """
    Test that the values to be plotted by the function for the mean waiting time
    of class 1 individuals are the expected when using:
        - Markov formula and simulation state probabilities
        - Markov formula and Markov state probabilities
        - Simulation
    """
    (
        range_space,
        simulation_times_using_markov_formula,
        markov_times,
        simulation_times,
    ) = plot_output_comparisons(
        lambda_1=3,
        lambda_2=4,
        mu=1,
        num_of_servers=3,
        threshold=6,
        system_capacity=15,
        buffer_capacity=7,
        seed_num=0,
        num_of_trials=1,
        runtime=100,
        measure_to_compare="waiting",
        class_type=0,
        plot_over="mu",
        max_parameter_value=5,
        accuracy=5,
    )

    expected_range_space = [1, 2, 3, 4, 5]
    expected_sim_times_using_formula = [
        2.377120739790196,
        0.7785480327193071,
        0.21825612502962743,
        0.0633853178321979,
        0.02219807426322811,
    ]
    expected_markov_times = [
        2.666380625245361,
        0.7505484517766888,
        0.201787897652177,
        0.06072282228882266,
        0.024434222615639434,
    ]
    expected_sim_times = [
        [2.100498503091243],
        [0.8060558886538617],
        [0.24673859227916475],
        [0.06673599211050996],
        [0.026042424326131127],
    ]
    assert np.all(range_space == expected_range_space)
    assert np.allclose(
        simulation_times_using_markov_formula, expected_sim_times_using_formula
    )
    assert np.allclose(markov_times, expected_markov_times)
    assert np.allclose(simulation_times, expected_sim_times)


def test_plot_output_comparisons_waiting_class_2():
    """
    Test that the values to be plotted by the function for the mean waiting time
    of class 2 individuals are the expected when using:
        - Markov formula and simulation state probabilities
        - Markov formula and Markov state probabilities
        - Simulation
    """
    (
        range_space,
        simulation_times_using_markov_formula,
        markov_times,
        simulation_times,
    ) = plot_output_comparisons(
        lambda_1=3,
        lambda_2=4,
        mu=1,
        num_of_servers=3,
        threshold=6,
        system_capacity=10,
        buffer_capacity=7,
        seed_num=0,
        num_of_trials=1,
        runtime=100,
        measure_to_compare="waiting",
        class_type=1,
        plot_over="system_capacity",
        max_parameter_value=18,
        accuracy=5,
    )

    expected_range_space = [
        10,
        12,
        14,
        16,
        18,
    ]
    expected_sim_times_using_formula = [
        0.9518119232230957,
        0.9314674163209273,
        0.8815151220881429,
        0.9520317760341209,
        0.9522967196743792,
    ]
    expected_markov_times = [
        0.9996062485853283,
        0.9996071004169865,
        0.9996071216135696,
        0.9996071221161823,
        0.9996071221275438,
    ]
    expected_sim_times = [
        [0.8587675978623437],
        [0.9410302653948986],
        [0.6712503805879015],
        [0.7596612894701423],
        [0.7466921877207321],
    ]
    assert np.all(range_space == expected_range_space)
    assert np.allclose(
        simulation_times_using_markov_formula, expected_sim_times_using_formula
    )
    assert np.allclose(markov_times, expected_markov_times)
    assert np.allclose(simulation_times, expected_sim_times)


def test_plot_output_comparisons_waiting_both_classes():
    """
    Test that the values to be plotted by the function for the mean waiting time
    of all individuals are the expected when using:
        - Markov formula and simulation state probabilities
        - Markov formula and Markov state probabilities
        - Simulation
    """
    (
        range_space,
        simulation_times_using_markov_formula,
        markov_times,
        simulation_times,
    ) = plot_output_comparisons(
        lambda_1=3,
        lambda_2=4,
        mu=1,
        num_of_servers=3,
        threshold=5,
        system_capacity=10,
        buffer_capacity=7,
        seed_num=0,
        num_of_trials=1,
        runtime=100,
        measure_to_compare="waiting",
        class_type=None,
        plot_over="threshold",
        max_parameter_value=9,
        accuracy=5,
    )

    expected_range_space = [
        5,
        6,
        7,
        8,
        9,
    ]
    expected_sim_times_using_formula = [
        1.4383683274990688,
        1.6172139699602939,
        1.7871674638990411,
        1.902900393648282,
        2.0799187425189745,
    ]
    expected_markov_times = [
        1.4997317350805834,
        1.6663508613218276,
        1.8329697824825426,
        1.999548467136932,
        2.165791830248812,
    ]
    expected_sim_times = [
        [1.4595100304540891],
        [1.5414680277219233],
        [1.8463653589649593],
        [1.9638358136060718],
        [2.1872623359765617],
    ]
    assert np.all(range_space == expected_range_space)
    assert np.allclose(
        simulation_times_using_markov_formula, expected_sim_times_using_formula
    )
    assert np.allclose(markov_times, expected_markov_times)
    assert np.allclose(simulation_times, expected_sim_times)


def test_plot_output_comparisons_blocking_class_1():
    """
    Test to ensure an error comes up when trying to get the blocking times of
    class 1 individuals
    """
    with pytest.raises(Exception):
        plot_output_comparisons(
            lambda_1=1,
            lambda_2=1,
            mu=1,
            num_of_servers=3,
            threshold=5,
            system_capacity=10,
            buffer_capacity=7,
            seed_num=0,
            num_of_trials=1,
            runtime=100,
            measure_to_compare="blocking",
            class_type=0,
            plot_over="lambda_1",
            max_parameter_value=3,
            accuracy=5,
        )


def test_plot_output_comparisons_blocking_class_2():
    """
    Test that the values to be plotted by the function for the mean blocking time
    of class 2 individuals are the expected when using:
        - Markov formula and simulation state probabilities
        - Markov formula and Markov state probabilities
        - Simulation
    """
    (
        range_space,
        simulation_times_using_markov_formula,
        markov_times,
        simulation_times,
    ) = plot_output_comparisons(
        lambda_1=1,
        lambda_2=1,
        mu=1,
        num_of_servers=3,
        threshold=5,
        system_capacity=10,
        buffer_capacity=7,
        seed_num=0,
        num_of_trials=1,
        runtime=100,
        measure_to_compare="blocking",
        class_type=1,
        plot_over="lambda_2",
        max_parameter_value=3,
        accuracy=None,
    )

    expected_range_space = [
        1,
        1.5,
        2,
        2.5,
        3,
    ]
    expected_sim_times_using_formula = [
        0.09939633736936365,
        0.3428086786668058,
        1.258688113496702,
        1.550748270791677,
        2.4490455912594884,
    ]
    expected_markov_times = [
        0.25749828422874693,
        0.7336269690016299,
        1.4059020459868858,
        2.0166211860863115,
        2.446138025813656,
    ]
    expected_sim_times = [
        [0.05675700649642476],
        [0.2035750550633296],
        [1.0204972927807057],
        [1.4297836865197424],
        [2.276273474404749],
    ]
    assert np.all(range_space == expected_range_space)
    assert np.allclose(
        simulation_times_using_markov_formula, expected_sim_times_using_formula
    )
    assert np.allclose(markov_times, expected_markov_times)
    assert np.allclose(simulation_times, expected_sim_times)


def test_plot_output_comparisons_blocking_both_classes():
    """
    Test that the values to be plotted by the function for the mean waiting time
    of all individuals are the expected when using:
        - Markov formula and simulation state probabilities
        - Markov formula and Markov state probabilities
        - Simulation
    """
    (
        range_space,
        simulation_times_using_markov_formula,
        markov_times,
        simulation_times,
    ) = plot_output_comparisons(
        lambda_1=1,
        lambda_2=1,
        mu=1,
        num_of_servers=1,
        threshold=5,
        system_capacity=10,
        buffer_capacity=7,
        seed_num=0,
        num_of_trials=1,
        runtime=100,
        measure_to_compare="blocking",
        class_type=None,
        plot_over="num_of_servers",
        max_parameter_value=5,
        accuracy=None,
    )

    expected_range_space = [
        1,
        2,
        3,
        4,
        5,
    ]
    expected_sim_times_using_formula = [
        30.454703888754974,
        0.8000539978455747,
        0.09939633736936365,
        0.08297030340373893,
        0.06341488800287158,
    ]
    expected_markov_times = [
        40.065612220723104,
        2.820781651110878,
        0.25749828422874693,
        0.05700263606859959,
        0.024799827726554754,
    ]
    expected_sim_times = [
        [10.427934396602263],
        [0.25420006034794723],
        [0.05675700649642476],
        [0.08092456927729426],
        [0.08979883878110877],
    ]
    assert np.all(range_space == expected_range_space)
    assert np.allclose(
        simulation_times_using_markov_formula, expected_sim_times_using_formula
    )
    assert np.allclose(markov_times, expected_markov_times)
    assert np.allclose(simulation_times, expected_sim_times)


@given(
    lambda_1=floats(min_value=1, max_value=3),
    lambda_2=floats(min_value=1, max_value=3),
    mu=floats(min_value=1, max_value=3),
    num_of_servers=integers(min_value=2, max_value=5),
    threshold=integers(min_value=2, max_value=10),
    system_capacity=integers(min_value=10, max_value=20),
    buffer_capacity=integers(min_value=2, max_value=10),
)
@settings(max_examples=5, deadline=None)
def test_plot_output_comparisons_blocking_property(
    lambda_1, lambda_2, mu, num_of_servers, threshold, system_capacity, buffer_capacity
):
    """
    Test that the values to be plotted by the function for the mean blocking time
    of either CLASS 2 INDIVIDUALS or ALL INDIVIDUALS are the same for all methods
    used:
        - Markov formula and simulation state probabilities
        - Markov formula and Markov state probabilities
        - Simulation
    These values are expected to be the same because class 1 individuals do not
    have any blocking time, and thus the overall blocking time is calculated just
    from class 2 individuals.
    """
    (
        range_space_1,
        simulation_times_using_markov_formula_1,
        markov_times_1,
        simulation_times_1,
    ) = plot_output_comparisons(
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        mu=mu,
        num_of_servers=num_of_servers,
        threshold=threshold,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
        seed_num=0,
        num_of_trials=1,
        runtime=100,
        measure_to_compare="blocking",
        class_type=1,
        plot_over="buffer_capacity",
        max_parameter_value=5,
        accuracy=None,
    )

    (
        range_space_2,
        simulation_times_using_markov_formula_2,
        markov_times_2,
        simulation_times_2,
    ) = plot_output_comparisons(
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        mu=mu,
        num_of_servers=num_of_servers,
        threshold=threshold,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
        seed_num=0,
        num_of_trials=1,
        runtime=100,
        measure_to_compare="blocking",
        class_type=None,
        plot_over="buffer_capacity",
        max_parameter_value=5,
        accuracy=None,
    )

    assert np.all(range_space_1 == range_space_2)
    assert np.all(
        simulation_times_using_markov_formula_1
        == simulation_times_using_markov_formula_2
    )
    assert np.all(markov_times_1 == markov_times_2)
    assert np.all(simulation_times_1 == simulation_times_2)


def test_plot_of_proportion_within_target_class_1():
    """
    Test the values to be plotted by the function for the mean proportion of
    individuals for class 1 are as expected for all methods used:
        - Markov formula and simulation state probabilities
        - Markov formula and Markov state probabilities
        - Simulation
    """
    (
        range_space,
        simulation_props_using_markov_formula,
        markov_props,
        simulation_props,
    ) = plot_output_comparisons(
        lambda_1=1,
        lambda_2=1,
        mu=1,
        num_of_servers=3,
        threshold=2,
        system_capacity=20,
        buffer_capacity=10,
        seed_num=1,
        num_of_trials=2,
        runtime=100,
        target=4,
        class_type=0,
        measure_to_compare="proportion",
        accuracy=5,
        plot_over="threshold",
        max_parameter_value=10,
    )

    expected_range_space = [
        2,
        4,
        6,
        8,
        10,
    ]
    expected_sim_props_using_formula = [
        0.9790136369646812,
        0.9694014142792851,
        0.9607171712756224,
        0.9512206646084153,
        0.9435197873252772,
    ]
    expected_markov_props = [
        0.9769758299950714,
        0.9698065422230608,
        0.9624762629273674,
        0.9564778065163335,
        0.9524639194726416,
    ]

    expected_sim_props = [
        [0.9615384615384616, 1.0],
        [0.9504132231404959, 0.978494623655914],
        [0.9338842975206612, 0.989247311827957],
        [0.9090909090909091, 0.989247311827957],
        [0.9008264462809917, 0.989247311827957],
    ]
    assert np.all(range_space == expected_range_space)
    assert np.allclose(
        simulation_props_using_markov_formula, expected_sim_props_using_formula
    )
    assert np.allclose(markov_props, expected_markov_props)
    assert np.allclose(simulation_props, expected_sim_props)


def test_plot_of_proportion_within_target_class_2():
    """
    Test the values to be plotted by the function for the mean proportion of
    individuals for class 2 are as expected for all methods used:
        - Markov formula and simulation state probabilities
        - Markov formula and Markov state probabilities
        - Simulation
    """
    (
        range_space,
        simulation_props_using_markov_formula,
        markov_props,
        simulation_props,
    ) = plot_output_comparisons(
        lambda_1=1,
        lambda_2=1,
        mu=1,
        num_of_servers=3,
        threshold=2,
        system_capacity=20,
        buffer_capacity=10,
        seed_num=1,
        num_of_trials=2,
        runtime=100,
        target=4,
        class_type=1,
        measure_to_compare="proportion",
        accuracy=5,
        plot_over="threshold",
        max_parameter_value=10,
    )

    expected_range_space = [
        2,
        4,
        6,
        8,
        10,
    ]
    expected_sim_props_using_formula = [
        0.9816843611112658,
        0.9776764633348265,
        0.9695135798097695,
        0.9607212930115949,
        0.9505136921747348,
    ]
    expected_markov_props = [
        0.9816843611112656,
        0.9776309516976318,
        0.9695851706481967,
        0.96203774630283,
        0.9559521606811459,
    ]

    expected_sim_props = [
        [1.0, 0.9880952380952381],
        [0.978021978021978, 0.9770114942528736],
        [0.967032967032967, 0.9655172413793104],
        [0.9560439560439561, 0.9655172413793104],
        [0.9230769230769231, 0.9655172413793104],
    ]
    assert np.all(range_space == expected_range_space)
    assert np.allclose(
        simulation_props_using_markov_formula, expected_sim_props_using_formula
    )
    assert np.allclose(markov_props, expected_markov_props)
    assert np.allclose(simulation_props, expected_sim_props)


def test_plot_of_proportion_within_target_both_classes():
    """
    Test the values to be plotted by the function for the mean proportion of
    individuals for both classes are as expected for all methods used:
        - Markov formula and simulation state probabilities
        - Markov formula and Markov state probabilities
        - Simulation
    """
    (
        range_space,
        simulation_props_using_markov_formula,
        markov_props,
        simulation_props,
    ) = plot_output_comparisons(
        lambda_1=1,
        lambda_2=1,
        mu=1,
        num_of_servers=3,
        threshold=2,
        system_capacity=20,
        buffer_capacity=10,
        seed_num=1,
        num_of_trials=2,
        runtime=100,
        target=4,
        class_type=None,
        measure_to_compare="proportion",
        accuracy=5,
        plot_over="threshold",
        max_parameter_value=10,
    )

    expected_range_space = [
        2,
        4,
        6,
        8,
        10,
    ]
    expected_sim_props_using_formula = [
        0.9803420072819845,
        0.973534925815833,
        0.965115375542696,
        0.955970978810005,
        0.947016739750006,
    ]
    expected_markov_props = [
        0.9793015428995077,
        0.9737157940379565,
        0.966029525931023,
        0.959257362821785,
        0.9542079250880933,
    ]

    expected_sim_props = [
        [0.9786096256684492, 0.9938650306748467],
        [0.9622641509433962, 0.9777777777777777],
        [0.9481132075471698, 0.9777777777777777],
        [0.9292452830188679, 0.9777777777777777],
        [0.910377358490566, 0.9777777777777777],
    ]
    assert np.all(range_space == expected_range_space)
    assert np.allclose(
        simulation_props_using_markov_formula, expected_sim_props_using_formula
    )
    assert np.allclose(markov_props, expected_markov_props)
    assert np.allclose(simulation_props, expected_sim_props)
