import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sympy as sym
import scipy as sci

from hypothesis import (
    given,
    settings,
    HealthCheck,
)
from hypothesis.strategies import (
    floats,
    integers,
    booleans,
)
from hypothesis.extra.numpy import arrays

from ambulance_game.markov.markov import (
    build_states,
    visualise_markov_chain,
    get_transition_matrix_entry,
    get_symbolic_transition_matrix,
    get_transition_matrix,
    convert_symbolic_transition_matrix,
    is_steady_state,
    get_steady_state_numerically,
    augment_Q,
    get_steady_state_algebraically,
    get_markov_state_probabilities,
    get_mean_number_of_individuals_in_system,
    get_mean_number_of_individuals_in_service_area,
    get_mean_number_of_individuals_in_buffer_center,
)

number_of_digits_to_round = 8


@given(
    threshold=integers(min_value=0, max_value=100),
    system_capacity=integers(min_value=1, max_value=100),
    buffer_capacity=integers(min_value=1, max_value=100),
)
def test_build_states(threshold, system_capacity, buffer_capacity):
    """
    Test to ensure that the build_states function returns the correct number of
    states, for different integer values of the threshold, system and buffer capacities
    """
    states = build_states(
        threshold=threshold,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
    )

    if threshold > system_capacity:
        assert len(states) == system_capacity + 1  # +2
    else:
        states_after_threshold = system_capacity - threshold + 1
        size_of_S2 = states_after_threshold if states_after_threshold >= 0 else 0
        all_states_size = size_of_S2 * (buffer_capacity + 1) + threshold
        assert len(states) == all_states_size


@given(
    num_of_servers=integers(min_value=2, max_value=10),
    threshold=integers(min_value=2, max_value=10),
    buffer_capacity=integers(min_value=2, max_value=10),
    system_capacity=integers(min_value=2, max_value=10),
)
@settings(deadline=None)
def test_visualise_markov_chain(
    num_of_servers, threshold, system_capacity, buffer_capacity
):
    """
    Test that checks if a neworkx MultiDiGraph object is returned and that the set
    of all nodes used is the same se as the set of all states that the build_states
    function returns.
    """
    all_states = build_states(
        threshold=threshold,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
    )
    set_of_all_states = set(all_states)

    markov_chain_plot = visualise_markov_chain(
        num_of_servers=num_of_servers,
        threshold=threshold,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
    )
    set_of_nodes = set(markov_chain_plot.nodes)

    assert type(markov_chain_plot) == nx.classes.multidigraph.DiGraph
    assert set_of_all_states == set_of_nodes
    plt.close()  # TODO Investigate if it's possible to remove this line


# TODO: Change the test so that the Health check does not need to be suppressed.
# If not suppressed test fails on mac-python 3.7 because data generation is slow
@settings(suppress_health_check=(HealthCheck.too_slow,))
@given(
    u=integers(min_value=0, max_value=1000),
    v=integers(min_value=0, max_value=1000),
    lambda_2=floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
    lambda_1=floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
    mu=floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
    num_of_servers=integers(min_value=1, max_value=100),
    threshold=integers(min_value=0, max_value=100),
    symbolic=booleans(),
)
def test_get_transition_matrix_entry(
    u,
    v,
    lambda_2,
    lambda_1,
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
    Lambda = lambda_2 + lambda_1

    if symbolic:
        Lambda = sym.symbols("Lambda")
        lambda_1 = sym.symbols("lambda_1")
        lambda_2 = sym.symbols("lambda_2")
        mu = sym.symbols("mu")

    origin_state = (u, v)
    destination_state_1 = (u, v + 1)
    destination_state_2 = (u + 1, v)
    destination_state_3 = (u, v - 1)
    destination_state_4 = (u - 1, v)

    entry_1 = get_transition_matrix_entry(
        origin=origin_state,
        destination=destination_state_1,
        threshold=threshold,
        lambda_2=lambda_2,
        lambda_1=lambda_1,
        Lambda=Lambda,
        mu=mu,
        num_of_servers=num_of_servers,
    )
    entry_2 = get_transition_matrix_entry(
        origin=origin_state,
        destination=destination_state_2,
        threshold=threshold,
        lambda_2=lambda_2,
        lambda_1=lambda_1,
        Lambda=Lambda,
        mu=mu,
        num_of_servers=num_of_servers,
    )
    entry_3 = get_transition_matrix_entry(
        origin=origin_state,
        destination=destination_state_3,
        threshold=threshold,
        lambda_2=lambda_2,
        lambda_1=lambda_1,
        Lambda=Lambda,
        mu=mu,
        num_of_servers=num_of_servers,
    )
    entry_4 = get_transition_matrix_entry(
        origin=origin_state,
        destination=destination_state_4,
        threshold=threshold,
        lambda_2=lambda_2,
        lambda_1=lambda_1,
        Lambda=Lambda,
        mu=mu,
        num_of_servers=num_of_servers,
    )

    assert entry_1 == (Lambda if v < threshold else lambda_1)
    assert entry_2 == lambda_2
    assert entry_3 == (mu * v if v <= num_of_servers else mu * num_of_servers)
    service_rate = threshold if threshold <= num_of_servers else num_of_servers
    assert entry_4 == (service_rate * mu if v == threshold else 0)


@given(
    num_of_servers=integers(min_value=1, max_value=5),
    threshold=integers(min_value=0, max_value=5),
    system_capacity=integers(min_value=5, max_value=10),
    buffer_capacity=integers(min_value=1, max_value=5),
)
@settings(deadline=None, max_examples=20)
def test_get_symbolic_transition_matrix(
    num_of_servers, threshold, system_capacity, buffer_capacity
):
    """
    Test that ensures the symbolic matrix function outputs the correct size matrix
    """
    states_after_threshold = system_capacity - threshold + 1
    S_2_size = states_after_threshold if states_after_threshold >= 0 else 0
    matrix_size = S_2_size * (buffer_capacity + 1) + threshold
    result = get_symbolic_transition_matrix(
        num_of_servers=num_of_servers,
        threshold=threshold,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
    )

    assert result.shape == (matrix_size, matrix_size)


@given(
    system_capacity=integers(min_value=10, max_value=20),
    buffer_capacity=integers(min_value=1, max_value=20),
    lambda_2=floats(
        min_value=0.05, max_value=100, allow_nan=False, allow_infinity=False
    ),
    lambda_1=floats(
        min_value=0.05, max_value=100, allow_nan=False, allow_infinity=False
    ),
    mu=floats(min_value=0.05, max_value=5, allow_nan=False, allow_infinity=False),
)
@settings(deadline=None)
def test_get_transition_matrix(
    system_capacity, buffer_capacity, lambda_2, lambda_1, mu
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
    matrix_size = S_2_size * (buffer_capacity + 1) + threshold

    transition_matrix = get_transition_matrix(
        lambda_2=lambda_2,
        lambda_1=lambda_1,
        mu=mu,
        num_of_servers=num_of_servers,
        threshold=threshold,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
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
    lambda_2 = 0.3
    lambda_1 = 0.2
    mu = 0.05
    num_of_servers = 10
    system_capacity = 8
    buffer_capacity = 2

    transition_matrix = get_transition_matrix(
        lambda_2=lambda_2,
        lambda_1=lambda_1,
        mu=mu,
        num_of_servers=num_of_servers,
        threshold=threshold,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
    )

    sym_transition_matrix = get_symbolic_transition_matrix(
        num_of_servers=num_of_servers,
        threshold=threshold,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
    )
    converted_matrix = convert_symbolic_transition_matrix(
        Q_sym=sym_transition_matrix, lambda_2=lambda_2, lambda_1=lambda_1, mu=mu
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

    assert is_steady_state(state=steady_1, Q=generator_matrix_1)
    assert is_steady_state(state=steady_2, Q=generator_matrix_2)
    assert not is_steady_state(state=steady_3, Q=generator_matrix_3)


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
    steady = get_steady_state_numerically(
        Q=Q, integration_function=sci.integrate.odeint
    )
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
        Q=Q, integration_function=sci.integrate.solve_ivp
    )
    assert is_steady_state(state=steady, Q=Q)


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
    steady = get_steady_state_algebraically(Q=Q, algebraic_function=np.linalg.solve)
    assert is_steady_state(state=steady, Q=Q)


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
    steady = get_steady_state_algebraically(Q=Q, algebraic_function=np.linalg.lstsq)
    assert is_steady_state(state=steady, Q=Q)


def test_get_state_probabilities_dict():
    """
    Test to ensure that sum of the values of the pi dictionary equate to 1
    """
    all_states = build_states(
        threshold=3,
        system_capacity=5,
        buffer_capacity=4,
    )
    transition_matrix = get_transition_matrix(
        lambda_2=0.1,
        lambda_1=0.2,
        mu=0.2,
        num_of_servers=3,
        threshold=3,
        system_capacity=5,
        buffer_capacity=4,
    )
    pi = get_steady_state_algebraically(
        Q=transition_matrix, algebraic_function=np.linalg.lstsq
    )
    pi_dictionary = get_markov_state_probabilities(
        pi=pi, all_states=all_states, output=dict
    )

    assert round(sum(pi_dictionary.values()), number_of_digits_to_round) == 1


def test_get_state_probabilities_array():
    """
    Test to ensure that the sum of elements of the pi array equate to 1
    """
    all_states = build_states(
        threshold=3,
        system_capacity=5,
        buffer_capacity=4,
    )
    transition_matrix = get_transition_matrix(
        lambda_2=0.1,
        lambda_1=0.2,
        mu=0.2,
        num_of_servers=3,
        threshold=3,
        system_capacity=5,
        buffer_capacity=4,
    )
    pi = get_steady_state_algebraically(
        Q=transition_matrix, algebraic_function=np.linalg.lstsq
    )
    pi_array = get_markov_state_probabilities(
        pi=pi, all_states=all_states, output=np.ndarray
    )

    assert round(np.nansum(pi_array), number_of_digits_to_round) == 1


def test_get_mean_number_of_individuals_examples():
    """
    Some examples to ensure that the correct mean number of individuals are output
    """
    all_states = build_states(threshold=4, system_capacity=20, buffer_capacity=20)
    transition_matrix = get_transition_matrix(
        lambda_2=0.2,
        lambda_1=0.2,
        mu=0.2,
        num_of_servers=3,
        threshold=4,
        system_capacity=20,
        buffer_capacity=20,
    )
    pi = get_steady_state_algebraically(
        Q=transition_matrix, algebraic_function=np.linalg.lstsq
    )
    assert (
        round(
            get_mean_number_of_individuals_in_system(pi=pi, states=all_states),
            number_of_digits_to_round,
        )
        == 2.88827497
    )
    assert (
        round(
            get_mean_number_of_individuals_in_service_area(pi=pi, states=all_states),
            number_of_digits_to_round,
        )
        == 2.44439504
    )
    assert (
        round(
            get_mean_number_of_individuals_in_buffer_center(pi=pi, states=all_states),
            number_of_digits_to_round,
        )
        == 0.44387993
    )
