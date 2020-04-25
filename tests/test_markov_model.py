import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sympy as sym
import scipy as sci

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
    get_mean_number_of_patients_in_system,
    get_mean_number_of_patients_in_hospital,
    get_mean_ambulances_blocked,
)


@given(
    threshold=integers(min_value=0, max_value=1000),
    system_capacity=integers(min_value=0, max_value=1000),
    parking_capacity=integers(min_value=0, max_value=1000),
)
def test_build_states(threshold, system_capacity, parking_capacity):
    """
    Test to ensure that the build_states function returns the correct number of states, for different integer values of the threshold, system and parking capacities
    """
    states = build_states(
        threshold=threshold,
        system_capacity=system_capacity,
        parking_capacity=parking_capacity,
    )

    if threshold > system_capacity:
        assert len(states) == system_capacity + 1
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
    Test that checks if a neworkx MultiDiGraph object is returned and that the set of all nodes used is the same se as the set of all states that the build_states function returns.
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

    assert type(markov_chain_plot) == nx.classes.multidigraph.MultiDiGraph
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
    Ensuring that the state mapping function works as it should for all cases two "adjacent" states. 
    
    Note here that hypothesis considers all variations of possible inputs along with a boolean variable (symbolic) to indicate whether to test the symblic version of the function or the numeric one.
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
    assert entry_4 == (threshold * mu if hospital_state == threshold else 0)


@given(
    num_of_servers=integers(min_value=1, max_value=5),
    threshold=integers(min_value=0, max_value=5),
    system_capacity=integers(min_value=5, max_value=10),
    parking_capacity=integers(min_value=1, max_value=5),
)
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
def test_get_transition_matrix(
    system_capacity, parking_capacity, lambda_a, lambda_o, mu
):
    """
    Test that ensures numeric transition matrix's shape is as expected and that some elements of the diagonal are what they should be. To be exact the first, last and middle row are check to see if the diagonal element of them equals to minus the sum of the entire row.
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
    Test that ensures that for fixed parameters and different values of the threshold the function that converts the symbolic matrix into a numeirc one gives the same results as the get_transition_matrix function.
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
    Given two steady states examples with their equivalent matrices, this test ensures that the function is_steady_state works as expected
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
    a=floats(min_value=0, max_value=10),
    b=floats(min_value=0, max_value=10),
    c=floats(min_value=0, max_value=10),
    d=floats(min_value=0, max_value=10),
    e=floats(min_value=0, max_value=10),
    f=floats(min_value=0, max_value=10),
)
def test_get_steady_state_numerically_odeint(a, b, c, d, e, f):
    """
    Ensures that getting the steady state numerically using scipy's odeint integration function returns the steady state for different transition-like matrices
    """
    Q = np.array([[-a - b, a, b], [c, -c - d, d], [e, f, -e - f]])
    steady = get_steady_state_numerically(Q, integration_function=sci.integrate.odeint)
    assert is_steady_state(steady, Q)


@given(
    a=floats(min_value=0, max_value=10),
    b=floats(min_value=0, max_value=10),
    c=floats(min_value=0, max_value=10),
    d=floats(min_value=0, max_value=10),
    e=floats(min_value=0, max_value=10),
    f=floats(min_value=0, max_value=10),
)
def test_get_steady_state_numerically_solve_ivp(a, b, c, d, e, f):
    """
    Ensures that getting the steady state numerically using scipy's solve_ivp integration function returns the steady state for different transition-like matrices
    """
    Q = np.array([[-a - b, a, b], [c, -c - d, d], [e, f, -e - f]])
    steady = get_steady_state_numerically(
        Q, integration_function=sci.integrate.solve_ivp
    )
    assert is_steady_state(steady, Q)


@given(Q=arrays(np.int8, (10, 10)))
def test_augment_Q(Q):
    """
    Tests that the array M that is returned has the same dimensions as Q and that the vector b is a one dimensional array of length equivalent to Q that consists of only zeros apart from the last element that is 1.
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
    Ensures that getting the steady state algebraically using numpy's solve function returns the steady state for different transition-like matrices
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
    Ensures that getting the steady state numerically using numoy's lstsq function returns the steady state for different transition-like matrices
    """
    Q = np.array([[-a - b, a, b], [c, -c - d, d], [e, f, -e - f]])
    steady = get_steady_state_algebraically(Q, algebraic_function=np.linalg.lstsq)
    assert is_steady_state(steady, Q)
