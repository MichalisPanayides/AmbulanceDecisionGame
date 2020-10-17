import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sympy as sym
import scipy as sci
import pytest

from hypothesis import (
    given,
    settings,
)
from hypothesis.strategies import (
    floats,
    integers,
    booleans,
)
from hypothesis.extra.numpy import arrays

from ambulance_game.markov.markov import (
    build_states,
    visualise_ambulance_markov_chain,
    get_transition_matrix_entry,
    get_symbolic_transition_matrix,
    get_transition_matrix,
    convert_symbolic_transition_matrix,
    is_steady_state,
    get_steady_state_numerically,
    augment_Q,
    get_steady_state_algebraically,
    get_markov_state_probabilities,
    get_mean_number_of_patients_in_system,
    get_mean_number_of_patients_in_hospital,
    get_mean_number_of_ambulances_blocked,
    get_mean_waiting_time_markov,
    is_blocking_state,
    expected_sojourn_time_in_markov_state,
    prob_service,
    prob_other_arrival,
    get_coefficients_row_of_array_associated_with_state,
    get_blocking_times_array_of_coefficients,
    convert_solution_to_correct_array_format,
    get_blocking_times_of_all_states,
    mean_blocking_time_formula,
    get_mean_blocking_time_markov,
)

number_of_digits_to_round = 8


@given(
    threshold=integers(min_value=0, max_value=100),
    system_capacity=integers(min_value=1, max_value=100),
    parking_capacity=integers(min_value=1, max_value=100),
)
def test_build_states(threshold, system_capacity, parking_capacity):
    """
    Test to ensure that the build_states function returns the correct number of
    states, for different integer values of the threshold, system and parking capacities
    """
    states = build_states(
        threshold=threshold,
        system_capacity=system_capacity,
        parking_capacity=parking_capacity,
    )

    if threshold > system_capacity:
        assert len(states) == system_capacity + 1  # +2
    else:
        states_after_threshold = system_capacity - threshold + 1
        size_of_S2 = states_after_threshold if states_after_threshold >= 0 else 0
        all_states_size = size_of_S2 * (parking_capacity + 1) + threshold
        assert len(states) == all_states_size


@given(
    num_of_servers=integers(min_value=2, max_value=10),
    threshold=integers(min_value=2, max_value=10),
    parking_capacity=integers(min_value=2, max_value=10),
    system_capacity=integers(min_value=2, max_value=10),
)
@settings(deadline=None)
def test_visualise_ambulance_markov_chain(
    num_of_servers, threshold, system_capacity, parking_capacity
):
    """
    Test that checks if a neworkx MultiDiGraph object is returned and that the set
    of all nodes used is the same se as the set of all states that the build_states
    function returns.
    """
    all_states = build_states(
        threshold=threshold,
        system_capacity=system_capacity,
        parking_capacity=parking_capacity,
    )
    set_of_all_states = set(all_states)

    markov_chain_plot = visualise_ambulance_markov_chain(
        num_of_servers=num_of_servers,
        threshold=threshold,
        system_capacity=system_capacity,
        parking_capacity=parking_capacity,
    )
    set_of_nodes = set(markov_chain_plot.nodes)

    assert type(markov_chain_plot) == nx.classes.multidigraph.DiGraph
    assert set_of_all_states == set_of_nodes
    plt.close()  # TODO Investigate if it's possible to remove this line


@given(
    ambulance_state=integers(min_value=0),
    hospital_state=integers(min_value=0),
    lambda_a=floats(min_value=0, allow_nan=False, allow_infinity=False),
    lambda_o=floats(min_value=0, allow_nan=False, allow_infinity=False),
    mu=floats(min_value=0, allow_nan=False, allow_infinity=False),
    num_of_servers=integers(min_value=1),
    threshold=integers(min_value=0),
    symbolic=booleans(),
)
def test_get_transition_matrix_entry(
    ambulance_state,
    hospital_state,
    lambda_a,
    lambda_o,
    mu,
    num_of_servers,
    threshold,
    symbolic,
):
    """
    Ensuring that the state mapping function works as it should for all cases
    of two adjacent states.

    Note here that hypothesis considers all variations of possible inputs along
    with a Boolean variable (symbolic) to indicate whether to test the symbolic
    version of the function or the numeric one.
    """
    Lambda = lambda_a + lambda_o

    if symbolic:
        Lambda = sym.symbols("Lambda")
        lambda_o = sym.symbols("lambda") ** sym.symbols("o")
        lambda_a = sym.symbols("lambda") ** sym.symbols("A")
        mu = sym.symbols("mu")

    origin_state = (ambulance_state, hospital_state)
    destination_state_1 = (ambulance_state, hospital_state + 1)
    destination_state_2 = (ambulance_state + 1, hospital_state)
    destination_state_3 = (ambulance_state, hospital_state - 1)
    destination_state_4 = (ambulance_state - 1, hospital_state)

    entry_1 = get_transition_matrix_entry(
        origin_state,
        destination_state_1,
        threshold=threshold,
        lambda_a=lambda_a,
        lambda_o=lambda_o,
        Lambda=Lambda,
        mu=mu,
        num_of_servers=num_of_servers,
    )
    entry_2 = get_transition_matrix_entry(
        origin_state,
        destination_state_2,
        threshold=threshold,
        lambda_a=lambda_a,
        lambda_o=lambda_o,
        Lambda=Lambda,
        mu=mu,
        num_of_servers=num_of_servers,
    )
    entry_3 = get_transition_matrix_entry(
        origin_state,
        destination_state_3,
        threshold=threshold,
        lambda_a=lambda_a,
        lambda_o=lambda_o,
        Lambda=Lambda,
        mu=mu,
        num_of_servers=num_of_servers,
    )
    entry_4 = get_transition_matrix_entry(
        origin_state,
        destination_state_4,
        threshold=threshold,
        lambda_a=lambda_a,
        lambda_o=lambda_o,
        Lambda=Lambda,
        mu=mu,
        num_of_servers=num_of_servers,
    )

    assert entry_1 == (Lambda if hospital_state < threshold else lambda_o)
    assert entry_2 == lambda_a
    assert entry_3 == (
        mu * hospital_state if hospital_state <= num_of_servers else mu * num_of_servers
    )
    service_rate = threshold if threshold <= num_of_servers else num_of_servers
    assert entry_4 == (service_rate * mu if hospital_state == threshold else 0)


@given(
    num_of_servers=integers(min_value=1, max_value=5),
    threshold=integers(min_value=0, max_value=5),
    system_capacity=integers(min_value=5, max_value=10),
    parking_capacity=integers(min_value=1, max_value=5),
)
@settings(deadline=None, max_examples=20)
def test_get_symbolic_transition_matrix(
    num_of_servers, threshold, system_capacity, parking_capacity
):
    """
    Test that ensures the symbolic matrix function outputs the correct size matrix
    """
    states_after_threshold = system_capacity - threshold + 1
    S_2_size = states_after_threshold if states_after_threshold >= 0 else 0
    matrix_size = S_2_size * (parking_capacity + 1) + threshold
    result = get_symbolic_transition_matrix(
        num_of_servers=num_of_servers,
        threshold=threshold,
        system_capacity=system_capacity,
        parking_capacity=parking_capacity,
    )

    assert result.shape == (matrix_size, matrix_size)


@given(
    system_capacity=integers(min_value=10, max_value=20),
    parking_capacity=integers(min_value=1, max_value=20),
    lambda_a=floats(
        min_value=0.05, max_value=100, allow_nan=False, allow_infinity=False
    ),
    lambda_o=floats(
        min_value=0.05, max_value=100, allow_nan=False, allow_infinity=False
    ),
    mu=floats(min_value=0.05, max_value=5, allow_nan=False, allow_infinity=False),
)
@settings(deadline=None)
def test_get_transition_matrix(
    system_capacity, parking_capacity, lambda_a, lambda_o, mu
):
    """
    Test that ensures numeric transition matrix's shape is as expected and that
    some elements of the diagonal are what they should be. To be exact the first,
    last and middle row are check to see if the diagonal element of them equals
    to minus the sum of the entire row.
    """
    num_of_servers = 10
    threshold = 8

    states_after_threshold = system_capacity - threshold + 1
    S_2_size = states_after_threshold if states_after_threshold >= 0 else 0
    matrix_size = S_2_size * (parking_capacity + 1) + threshold

    transition_matrix = get_transition_matrix(
        lambda_a=lambda_a,
        lambda_o=lambda_o,
        mu=mu,
        num_of_servers=num_of_servers,
        threshold=threshold,
        system_capacity=system_capacity,
        parking_capacity=parking_capacity,
    )

    assert matrix_size == np.shape(transition_matrix)[0]
    mid = int(matrix_size / 2)
    assert transition_matrix[0][0] == -sum(transition_matrix[0][1:])
    assert transition_matrix[-1][-1] == -sum(transition_matrix[-1][:-1])

    mid_row_sum = sum(transition_matrix[mid][:mid]) + sum(
        transition_matrix[mid][mid + 1 :]
    )
    assert np.isclose(transition_matrix[mid][mid], -mid_row_sum)


@given(threshold=integers(min_value=0, max_value=10))
@settings(deadline=None)
def test_convert_symbolic_transition_matrix(threshold):
    """
    Test that ensures that for fixed parameters and different values of the threshold
    the function that converts the symbolic matrix into a numeric one gives the
    same results as the get_transition_matrix function.
    """
    lambda_a = 0.3
    lambda_o = 0.2
    mu = 0.05
    num_of_servers = 10
    system_capacity = 8
    parking_capacity = 2

    transition_matrix = get_transition_matrix(
        lambda_a=lambda_a,
        lambda_o=lambda_o,
        mu=mu,
        num_of_servers=num_of_servers,
        threshold=threshold,
        system_capacity=system_capacity,
        parking_capacity=parking_capacity,
    )

    sym_transition_matrix = get_symbolic_transition_matrix(
        num_of_servers=num_of_servers,
        threshold=threshold,
        system_capacity=system_capacity,
        parking_capacity=parking_capacity,
    )
    converted_matrix = convert_symbolic_transition_matrix(
        sym_transition_matrix, lambda_a, lambda_o, mu
    )

    assert np.allclose(converted_matrix, transition_matrix)


def test_is_steady_state_examples():
    """
    Given two steady states examples with their equivalent matrices, this test ensures
    that the function is_steady_state works as expected
    """
    steady_1 = [6 / 17, 6 / 17, 5 / 17]
    generator_matrix_1 = np.array(
        [[-2 / 3, 1 / 3, 1 / 3], [1 / 2, -1 / 2, 0], [1 / 5, 1 / 5, -2 / 5]]
    )

    steady_2 = np.array([0.0877193, 0.38596491, 0.52631579])
    generator_matrix_2 = np.array([[-0.6, 0.4, 0.2], [0, -0.5, 0.5], [0.1, 0.3, -0.4]])

    steady_3 = np.array([1, 2, 3])
    generator_matrix_3 = np.array([[-4, 2, 2], [0, -2, 2], [1, 5, -6]])

    assert is_steady_state(steady_1, generator_matrix_1)
    assert is_steady_state(steady_2, generator_matrix_2)
    assert not is_steady_state(steady_3, generator_matrix_3)


@given(
    a=floats(min_value=1, max_value=10),
    b=floats(min_value=1, max_value=10),
    c=floats(min_value=1, max_value=10),
    d=floats(min_value=1, max_value=10),
    e=floats(min_value=1, max_value=10),
    f=floats(min_value=1, max_value=10),
)
def test_get_steady_state_numerically_odeint(a, b, c, d, e, f):
    """
    Ensures that getting the steady state numerically using scipy's odeint integration
    function returns the steady state for different transition-like matrices
    """
    Q = np.array([[-a - b, a, b], [c, -c - d, d], [e, f, -e - f]])
    steady = get_steady_state_numerically(Q, integration_function=sci.integrate.odeint)
    assert is_steady_state(steady, Q)


@given(
    a=floats(min_value=1, max_value=10),
    b=floats(min_value=1, max_value=10),
    c=floats(min_value=1, max_value=10),
    d=floats(min_value=1, max_value=10),
    e=floats(min_value=1, max_value=10),
    f=floats(min_value=1, max_value=10),
)
def test_get_steady_state_numerically_solve_ivp(a, b, c, d, e, f):
    """
    Ensures that getting the steady state numerically using scipy's solve_ivp integration
    function returns the steady state for different transition-like matrices
    """
    Q = np.array([[-a - b, a, b], [c, -c - d, d], [e, f, -e - f]])
    steady = get_steady_state_numerically(
        Q, integration_function=sci.integrate.solve_ivp
    )
    assert is_steady_state(steady, Q)


@given(Q=arrays(np.int8, (10, 10)))
def test_augment_Q(Q):
    """
    Tests that the array M that is returned has the same dimensions as Q and that
    the vector b is a one dimensional array of length equivalent to Q that consists
    of only zeros apart from the last element that is 1.
    """
    M, b = augment_Q(Q)
    assert M.shape == (10, 10)
    assert b.shape == (10, 1)
    assert all(b[0:-1]) == 0
    assert b[-1] == 1


@given(
    a=floats(min_value=1, max_value=10),
    b=floats(min_value=1, max_value=10),
    c=floats(min_value=1, max_value=10),
    d=floats(min_value=1, max_value=10),
    e=floats(min_value=1, max_value=10),
    f=floats(min_value=1, max_value=10),
)
def test_get_steady_state_algebraically_solve(a, b, c, d, e, f):
    """
    Ensures that getting the steady state algebraically using numpy's solve function
    returns the steady state for different transition-like matrices
    """
    Q = np.array([[-a - b, a, b], [c, -c - d, d], [e, f, -e - f]])
    steady = get_steady_state_algebraically(Q, algebraic_function=np.linalg.solve)
    assert is_steady_state(steady, Q)


@given(
    a=floats(min_value=1, max_value=10),
    b=floats(min_value=1, max_value=10),
    c=floats(min_value=1, max_value=10),
    d=floats(min_value=1, max_value=10),
    e=floats(min_value=1, max_value=10),
    f=floats(min_value=1, max_value=10),
)
def test_get_steady_state_algebraically_lstsq(a, b, c, d, e, f):
    """
    Ensures that getting the steady state numerically using numpy's
    lstsq function returns the steady state for different transition-like matrices
    """
    Q = np.array([[-a - b, a, b], [c, -c - d, d], [e, f, -e - f]])
    steady = get_steady_state_algebraically(Q, algebraic_function=np.linalg.lstsq)
    assert is_steady_state(steady, Q)


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

    all_states = build_states(threshold, system_capacity, parking_capacity)
    transition_matrix = get_transition_matrix(
        lambda_a,
        lambda_o,
        mu,
        num_of_servers,
        threshold,
        system_capacity,
        parking_capacity,
    )
    pi = get_steady_state_algebraically(
        transition_matrix, algebraic_function=np.linalg.lstsq
    )
    pi_dictionary = get_markov_state_probabilities(
        pi=pi, all_states=all_states, output=dict
    )

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

    all_states = build_states(threshold, system_capacity, parking_capacity)
    transition_matrix = get_transition_matrix(
        lambda_a,
        lambda_o,
        mu,
        num_of_servers,
        threshold,
        system_capacity,
        parking_capacity,
    )
    pi = get_steady_state_algebraically(
        transition_matrix, algebraic_function=np.linalg.lstsq
    )
    pi_array = get_markov_state_probabilities(
        pi=pi, all_states=all_states, output=np.ndarray
    )

    assert round(np.nansum(pi_array), number_of_digits_to_round) == 1


def test_get_mean_number_of_patients_examples():
    """
    Some examples to ensure that the correct mean number of patients are output
    """
    lambda_a = 0.2
    lambda_o = 0.2
    mu = 0.2
    num_of_servers = 3
    threshold = 4
    system_capacity = 20
    parking_capacity = 20

    all_states = build_states(threshold, system_capacity, parking_capacity)
    transition_matrix = get_transition_matrix(
        lambda_a,
        lambda_o,
        mu,
        num_of_servers,
        threshold,
        system_capacity,
        parking_capacity,
    )
    pi = get_steady_state_algebraically(
        transition_matrix, algebraic_function=np.linalg.lstsq
    )
    assert (
        round(
            get_mean_number_of_patients_in_system(pi, all_states),
            number_of_digits_to_round,
        )
        == 2.88827497
    )
    assert (
        round(
            get_mean_number_of_patients_in_hospital(pi, all_states),
            number_of_digits_to_round,
        )
        == 2.44439504
    )
    assert (
        round(
            get_mean_number_of_ambulances_blocked(pi, all_states),
            number_of_digits_to_round,
        )
        == 0.44387993
    )


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


def test_is_blocking_state():
    """
    Ensure that function returns:
        - True when state is of the form (u,v) and u > 0
        - False when state is of the form (u,v) and u = 0
    """
    for v in range(1, 100):
        u = random.randint(1, 100)
        assert is_blocking_state((u, v))

    for v in range(1, 100):
        assert not is_blocking_state((0, v))


def test_expected_sojourn_time_in_markov_state():
    """
    Ensure that the expected time spent in state (u,v) does not depends on the value
    of u (with the exception of u=0).
    """
    assert (
        round(
            expected_sojourn_time_in_markov_state(
                state=(1, 3), lambda_o=0.4, mu=1.2, num_of_servers=4, system_capacity=5
            ),
            number_of_digits_to_round,
        )
        == round(
            expected_sojourn_time_in_markov_state(
                state=(100, 3),
                lambda_o=0.4,
                mu=1.2,
                num_of_servers=4,
                system_capacity=5,
            ),
            number_of_digits_to_round,
        )
        == 0.25
    )

    assert (
        round(
            expected_sojourn_time_in_markov_state(
                state=(1, 5), lambda_o=0.4, mu=1.2, num_of_servers=4, system_capacity=5
            ),
            number_of_digits_to_round,
        )
        == round(
            expected_sojourn_time_in_markov_state(
                state=(100, 5),
                lambda_o=0.4,
                mu=1.2,
                num_of_servers=4,
                system_capacity=5,
            ),
            number_of_digits_to_round,
        )
        == 0.20833333
    )


def test_prob_service():
    """
    Ensure that probability of service remains fixed for all states when C=1
    """
    for v in range(1, 100):
        u = random.randint(1, 100)
        mu = random.randint(1, 100)
        lambda_o = random.random()
        prob = prob_service(state=(u, v), lambda_o=lambda_o, mu=mu, num_of_servers=1)
        assert prob == mu / (lambda_o + mu)


def test_prob_other_arrival():
    """
    Ensure that probability of other arrivals remains fixed for all states when C=1
    """
    for v in range(1, 100):
        u = random.randint(1, 100)
        mu = random.randint(1, 100)
        lambda_o = random.random()
        prob = prob_other_arrival(
            state=(u, v), lambda_o=lambda_o, mu=mu, num_of_servers=1
        )
        assert prob == lambda_o / (lambda_o + mu)


def test_get_coefficients_row_of_array_associated_with_state_example_1():
    M_row, b_element = get_coefficients_row_of_array_associated_with_state(
        state=(2, 1),
        lambda_o=0.3,
        mu=0.5,
        num_of_servers=1,
        threshold=1,
        system_capacity=5,
        parking_capacity=3,
    )
    assert np.allclose(
        M_row, np.array([0.625, 0, 0, 0, 0, -1, 0.375, 0, 0, 0, 0, 0, 0, 0, 0])
    )
    assert b_element == -1.25


def test_get_coefficients_row_of_array_associated_with_state_example_2():
    M_row, b_element = get_coefficients_row_of_array_associated_with_state(
        state=(4, 7),
        lambda_o=2,
        mu=1,
        num_of_servers=3,
        threshold=4,
        system_capacity=10,
        parking_capacity=5,
    )
    assert np.allclose(
        M_row,
        np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.6,
                -1.0,
                0.4,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        ),
    )
    assert b_element == -0.2


def test_get_coefficients_row_of_array_associated_with_state_example_3():
    with pytest.raises(IndexError):
        get_coefficients_row_of_array_associated_with_state(
            state=(4, 7),
            lambda_o=2,
            mu=1,
            num_of_servers=3,
            threshold=10,
            system_capacity=10,
            parking_capacity=5,
        )


def test_get_blocking_times_array_of_coefficients_example_1():
    M, b = get_blocking_times_array_of_coefficients(
        lambda_o=2,
        mu=3,
        num_of_servers=1,
        threshold=3,
        system_capacity=4,
        parking_capacity=2,
    )
    assert np.alltrue(
        M
        == np.array(
            [
                [-1.0, 0.4, 0.0, 0.0],
                [0.6, 0.0, -1.0, 0.4],
                [1.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, -1.0],
            ]
        ),
    )
    assert np.alltrue(b == [-0.2, -0.2, -0.3333333333333333, -0.3333333333333333])


def test_get_blocking_times_array_of_coefficients_example_2():
    M, b = get_blocking_times_array_of_coefficients(
        lambda_o=2,
        mu=1,
        num_of_servers=3,
        threshold=3,
        system_capacity=5,
        parking_capacity=2,
    )
    assert np.alltrue(
        M
        == np.array(
            [
                [-1.0, 0.4, 0.0, 0.0, 0.0, 0.0],
                [0.6, 0.0, 0.0, -1.0, 0.4, 0.0],
                [0.6, -1.0, 0.4, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.6, -1.0, 0.4],
                [0.0, 1.0, -1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, -1.0],
            ]
        ),
    )
    assert np.alltrue(
        b == [-0.2, -0.2, -0.2, -0.2, -0.3333333333333333, -0.3333333333333333]
    )


def test_get_blocking_times_array_of_coefficients_example_3():
    M, b = get_blocking_times_array_of_coefficients(
        lambda_o=0.4,
        mu=0.1,
        num_of_servers=6,
        threshold=4,
        system_capacity=4,
        parking_capacity=7,
    )
    assert np.alltrue(
        M
        == np.array(
            [
                [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0],
            ]
        ),
    )
    assert np.alltrue(b == [-2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5])


def test_convert_solution_to_correct_array_format_examples():
    converted_1 = convert_solution_to_correct_array_format(
        np.array([1, 2, 3, 4, 5, 6]), 2, 4, 2
    )
    assert np.alltrue(
        converted_1 == np.array([[0, 0, 0, 0, 0], [0, 0, 1, 2, 3], [0, 0, 4, 5, 6]])
    )

    converted_2 = convert_solution_to_correct_array_format(
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]), 3, 5, 3
    )
    assert np.alltrue(
        converted_2
        == np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 2, 3],
                [0, 0, 0, 4, 5, 6],
                [0, 0, 0, 7, 8, 9],
            ]
        )
    )


def test_get_blocking_times_of_all_states_example_1():
    """Example of blocking times of all states when the threshold is the same as
    the system capacity (T = N)
    """
    blocking_times = get_blocking_times_of_all_states(
        lambda_o=2,
        mu=3,
        num_of_servers=1,
        threshold=3,
        system_capacity=3,
        parking_capacity=4,
    )
    assert np.allclose(
        blocking_times,
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.33333333],
                [0.0, 0.0, 0.0, 0.66666667],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.33333333],
            ]
        ),
    )


def test_get_blocking_times_of_all_states_example_2():
    """[summary]"""
    blocking_times = get_blocking_times_of_all_states(
        lambda_o=2,
        mu=1,
        num_of_servers=3,
        threshold=1,
        system_capacity=4,
        parking_capacity=5,
    )
    assert np.allclose(
        blocking_times,
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 3.11111111, 4.16666667, 4.72222222, 5.05555556],
                [0.0, 6.22222222, 7.27777778, 7.83333333, 8.16666667],
                [0.0, 9.33333333, 10.38888889, 10.94444444, 11.27777778],
                [0.0, 12.44444444, 13.5, 14.05555556, 14.38888889],
                [0.0, 15.55555556, 16.61111111, 17.16666667, 17.5],
            ]
        ),
    )


def test_get_blocking_times_of_all_states_example_3():
    blocking_times = get_blocking_times_of_all_states(
        lambda_o=4,
        mu=1,
        num_of_servers=5,
        threshold=3,
        system_capacity=6,
        parking_capacity=8,
    )
    assert np.allclose(
        blocking_times,
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.14666667, 1.75666667, 2.11666667, 2.31666667],
                [0.0, 0.0, 0.0, 2.29333333, 2.90333333, 3.26333333, 3.46333333],
                [0.0, 0.0, 0.0, 3.44, 4.05, 4.41, 4.61],
                [0.0, 0.0, 0.0, 4.58666667, 5.19666667, 5.55666667, 5.75666667],
                [0.0, 0.0, 0.0, 5.73333333, 6.34333333, 6.70333333, 6.90333333],
                [0.0, 0.0, 0.0, 6.88, 7.49, 7.85, 8.05],
                [0.0, 0.0, 0.0, 8.02666667, 8.63666667, 8.99666667, 9.19666667],
                [0.0, 0.0, 0.0, 9.17333333, 9.78333333, 10.14333333, 10.34333333],
            ]
        ),
    )


def test_mean_blocking_time_formula_algebraic():
    all_states = [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 4),
        (2, 4),
        (0, 5),
        (1, 5),
        (2, 5),
        (0, 6),
        (1, 6),
        (2, 6),
        (0, 7),
        (1, 7),
        (2, 7),
        (0, 8),
        (1, 8),
        (2, 8),
    ]
    state_probabilities = np.array(
        [
            [
                0.05924777,
                0.14811941,
                0.18514927,
                0.15429106,
                0.12857588,
                0.04291957,
                0.01439794,
                0.00493644,
                0.00185116,
            ],
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                0.064227,
                0.03378794,
                0.01552454,
                0.00676837,
                0.00300093,
            ],
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                0.04110293,
                0.04024539,
                0.02855397,
                0.01753342,
                0.00976702,
            ],
        ]
    )
    blocking_time = mean_blocking_time_formula(
        all_states=all_states,
        pi=state_probabilities,
        lambda_o=3,
        mu=2,
        num_of_servers=3,
        threshold=4,
        system_capacity=8,
        parking_capacity=2,
    )
    assert round(blocking_time, number_of_digits_to_round) == 0.23047954


def test_get_mean_blocking_time_markov_example_1():
    assert (
        round(
            get_mean_blocking_time_markov(
                lambda_a=2,
                lambda_o=3,
                mu=2,
                num_of_servers=3,
                threshold=4,
                system_capacity=8,
                parking_capacity=2,
                formula="algebraic",
            ),
            number_of_digits_to_round,
        )
        == 0.23047954
    )


def test_get_mean_blocking_time_markov_example_2():
    assert (
        round(
            get_mean_blocking_time_markov(
                lambda_a=5,
                lambda_o=6,
                mu=2,
                num_of_servers=7,
                threshold=5,
                system_capacity=15,
                parking_capacity=7,
                formula="algebraic",
            ),
            number_of_digits_to_round,
        )
        == 0.62492091
    )


def test_mean_blocking_time_formula_closed_form():
    # TODO: Make test once closed form formula is found
    assert (
        mean_blocking_time_formula(
            None, None, None, None, None, None, None, None, formula="closed-form"
        )
        == "TBA"
    )
