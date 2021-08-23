"""
Code to create the Markov chain model.
"""

import itertools

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy as sci
import scipy.integrate  # pylint: disable=unused-import
import sympy as sym


def build_states(threshold, system_capacity, buffer_capacity):
    """Builds the set of states in a list format by combine two sets of states where:
        - states_1 consists of all states before reaching the threshold
            (0, 0), (0, 1), ..., (0, T-1) where T is the threshold
        - states_2 consists of all states after reaching the threshold including
        the threshold (where S is the system capacity)
            (0, T), (0, T+1), ..., (0, S)
            (1, T), (1, T+1), ..., (1, S)
              .         .            .
              .         .            .
              .         .            .
            (P, T), (P, T+1), ..., (P, S)

    Note that if the threshold is greater than the system_capacity then the Markov
    chain will be of the form:
         (0, 0), (0, 1), ..., (0, S)
    Parameters
    ----------
    threshold : int
        Distinguishes between the two sets of states to be combined. In general,
        if the number of individuals in the service area >= threshold then,
        class 2 individuals are not allowed.
    system_capacity : int
        The maximum capacity of the service area (i.e. number of servers + queue size)
    buffer_capacity : int
        The number of buffer spaces

    Returns
    -------
    list
        a list of all the states

    TODO: turn into a generator
    """
    if buffer_capacity < 1:
        raise ValueError(
            "Simulation only implemented for buffer_capacity >= 1"
        )  # TODO Add an option to ciw model to all for no buffer capacity.

    if threshold > system_capacity:
        return [(0, v) for v in range(0, system_capacity + 1)]
        # states_1 = [(0, v) for v in range(0, system_capacity + 1)]
        # states_2 = [(1, system_capacity)]
        # return states_1 + states_2

    states_1 = [(0, v) for v in range(0, threshold)]
    states_2 = [
        (u, v)
        for v in range(threshold, system_capacity + 1)
        for u in range(buffer_capacity + 1)
    ]
    all_states = states_1 + states_2

    return all_states


def visualise_markov_chain(
    num_of_servers,
    threshold,
    system_capacity,
    buffer_capacity,
    nodesize_free=2000,
    nodesize_full=2000,
    fontsize=12,
):
    """This function's purpose is the visualisation of the Markov chain system using
    the networkx library. The networkx object that is created positions all states
    based on their (u, v) labels.

    Parameters
    ----------
    num_of_servers : int
        All states (u,v) such that v >= num_of_servers are coloured red to indicate
        the point that the system has no free servers
    threshold : int
        The number where v=threshold that indicates the split of the two sets
    buffer_capacity : int
        The maximum number of u in all states (u,v)
    system_capacity : int
        The maximum number of v in all states (u,v)

    Returns
    -------
    object
        a networkx object that consists of the Markov chain
    """

    all_states = build_states(
        threshold=threshold,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
    )
    graph = nx.DiGraph()
    for _, origin_state in enumerate(all_states):
        for _, destination_state in enumerate(all_states):
            column_adjacent = (
                destination_state[0] - origin_state[0] == 1
                and destination_state[1] - origin_state[1] == 0
            )
            row_adjacent = (
                destination_state[1] - origin_state[1] == 1
                and destination_state[0] - origin_state[0] == 0
            )
            if row_adjacent or column_adjacent:
                graph.add_edge(origin_state, destination_state, color="blue")

    plt.figure(figsize=((system_capacity + 1) * 1.5, (buffer_capacity + 1) * 1.5))
    pos = {state: [state[1], -state[0]] for state in all_states}
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=nodesize_free,
        nodelist=[state for state in all_states if state[1] <= num_of_servers],
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=nodesize_full,
        nodelist=[state for state in all_states if state[1] > num_of_servers],
        node_color="red",
    )
    nx.draw_networkx_edges(graph, pos, arrowstyle="fancy")
    nx.draw_networkx_labels(graph, pos, font_size=fontsize)

    plt.axis("off")
    return graph


def get_transition_matrix_entry(
    origin, destination, threshold, lambda_2, lambda_1, Lambda, mu, num_of_servers
):
    """Obtains the entry of the transition matrix based on the state mapping function.
    For a given entry of the transition matrix, the function uses the difference
    between the origin and destination states (u_i,v_i) - (u_j,v_j) along with the
    threshold to determine what is the rate of going from the origin state to the
    destination state.

    This function is used for both the symbolic and numeric transition matrix.

    Parameters
    ----------
    origin : tuple
        The origin state (u_i, v_i)
    destination : tuple
        The destination state (u_j,v_j)
    threshold : int
        Indication of when to stop using Lambda as the arrival rate and split it
        into lambda_2 and lambda_1
    lambda_2 : float or sympy.Symbol object
    lambda_1 : float or sympy.Symbol object
    Lambda : float or sympy.Symbol object
        The sum of lambda_2 and lambda_1 OR the symbol Λ
    mu : float or sympy.Symbol object
    num_of_servers : int
        Indication of when to stabilise the service rate

    Returns
    -------
    float or sympy.Symbol object
        The numeric or symbolic entry of the matrix
    """
    delta = np.array(origin) - np.array(destination)
    if np.all(delta == (0, -1)):
        if origin[1] < threshold:
            return Lambda
        return lambda_1
    if np.all(delta == (-1, 0)):
        return lambda_2
    if np.all(delta == (0, 1)) or (np.all(delta == (1, 0)) and origin[1] == threshold):
        return min(origin[1], num_of_servers) * mu
    return 0


def get_symbolic_transition_matrix(
    num_of_servers, threshold, system_capacity, buffer_capacity
):
    """Obtain the transition matrix with symbols instead of the actual values of
    lambda_2, lambda_1 and mu.

    Returns
    -------
    sympy.matrices object
        The symbolic transition matrix
    """
    Lambda = sym.symbols("Lambda")
    lambda_1 = sym.symbols("lambda_1")
    lambda_2 = sym.symbols("lambda_2")
    mu = sym.symbols("mu")

    all_states = build_states(
        threshold=threshold,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
    )
    Q = sym.zeros(len(all_states))
    # if threshold > system_capacity:
    #     threshold = system_capacity
    for (i, origin_state), (j, destination_state) in itertools.product(
        enumerate(all_states), repeat=2
    ):
        Q[i, j] = get_transition_matrix_entry(
            origin=origin_state,
            destination=destination_state,
            threshold=threshold,
            lambda_2=lambda_2,
            lambda_1=lambda_1,
            Lambda=Lambda,
            mu=mu,
            num_of_servers=num_of_servers,
        )

    sum_of_rates = -np.sum(Q, axis=1)
    Q = Q + sym.Matrix(np.diag(sum_of_rates))

    return Q


def get_transition_matrix(
    lambda_2, lambda_1, mu, num_of_servers, threshold, system_capacity, buffer_capacity
):
    """Obtain the numerical transition matrix that consists of all rates between
    any two states.

    Parameters
    ----------
    num_of_servers : int
        The number of servers
    threshold : int
        The threshold that indicates when to start blocking class 2 individuals
    system_capacity : int
        The total capacity of the system
    buffer_capacity : int
        The buffer capacity

    Returns
    -------
    numpy.ndarray
        The transition matrix Q
    """
    all_states = build_states(
        threshold=threshold,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
    )
    size = len(all_states)
    Q = np.zeros((size, size))
    # if threshold > system_capacity:
    #     threshold = system_capacity
    for (i, origin_state), (j, destination_state) in itertools.product(
        enumerate(all_states), repeat=2
    ):
        Q[i, j] = get_transition_matrix_entry(
            origin=origin_state,
            destination=destination_state,
            threshold=threshold,
            lambda_2=lambda_2,
            lambda_1=lambda_1,
            Lambda=lambda_2 + lambda_1,
            mu=mu,
            num_of_servers=num_of_servers,
        )
    sum_of_rates = np.sum(Q, axis=1)
    np.fill_diagonal(Q, -sum_of_rates)
    return Q


def convert_symbolic_transition_matrix(Q_sym, lambda_2, lambda_1, mu):
    """Converts the symbolic matrix obtained from the get_symbolic_transition_matrix()
    function to the corresponding numerical matrix. The output of this function
    should be the same as the output of get_transition_matrix()

    Parameters
    ----------
    Q_sym : sympy.matrices object
        The symbolic transition matrix obtained from get_symbolic_transition_matrix()

    Returns
    -------
    numpy.ndarray
        The transition matrix Q

    TODO: get rid of first four lines somehow
    """
    sym_Lambda = sym.symbols("Lambda")
    sym_lambda_1 = sym.symbols("lambda_1")
    sym_lambda_2 = sym.symbols("lambda_2")
    sym_mu = sym.symbols("mu")

    Q = np.array(
        Q_sym.subs(
            {
                sym_Lambda: lambda_2 + lambda_1,
                sym_lambda_1: lambda_1,
                sym_lambda_2: lambda_2,
                sym_mu: mu,
            }
        )
    ).astype(np.float64)
    return Q


def is_steady_state(state, Q):
    """Checks if a give vector π is a steady state vector of the Markov chain by
    confirming that:
            πQ = 0

    Parameters
    ----------
    state : numpy.ndarray
        A vector with probabilities to be checked if is the steady state
    Q : numpy.ndarray
        The numeric transition matrix of the corresponding Markov chain

    Returns
    -------
    bool
        True: if the dot product πQ is very close to 0
    """
    return np.allclose(np.dot(state, Q), 0)


def get_steady_state_numerically(
    Q, max_t=100, number_of_timepoints=1000, integration_function=sci.integrate.odeint
):
    """Finds the steady state of the Markov chain numerically using either scipy
    odeint() or solve_ivp() functions. For each method used a certain set of steps
    occur:
        - Get an initial state vector (1/n, 1/n, 1/n, ..., 1/n) where n is the
        total number of states (or size of Q)
        - Enter a loop and exit the loop only when a steady state is found
        - Get the integration interval to be used by the solver: t_span
        - Based on the integration function that will be used, use the corresponding
        derivative function
        - Get the state vector and check if it is a steady state
        - if not repeat

    -> odeint():
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
    -> solve_ivp():
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html

    Parameters
    ----------
    Q : numpy.ndarray
        Transition Matrix
    max_t : int, optional
        maximum time that the differential equation will be solved, by default 100
    number_of_timepoints : int, optional
        the number of timepoints between 0 and max_t, by default 1000
    integration_function : function, optional
        The integration function to be used, by default sci.integrate.odeint

    Returns
    -------
    numpy.ndarray
        The steady state vector of the Markov chain
    """

    def derivative_odeint(x, t):  # pylint: disable=unused-argument
        return np.dot(x, Q)

    def derivative_solve_ivp(t, x):  # pylint: disable=unused-argument
        return np.dot(x, Q)

    dimension = Q.shape[0]
    state = np.ones(dimension) / dimension
    while not is_steady_state(state=state, Q=Q):
        t_span = np.linspace(0, max_t, number_of_timepoints)
        if integration_function == sci.integrate.odeint:
            sol = integration_function(func=derivative_odeint, y0=state, t=t_span)
            state = sol[-1]
        elif integration_function == sci.integrate.solve_ivp:
            sol = integration_function(
                fun=derivative_solve_ivp, y0=state, t_span=t_span
            )
            state = sol.y[:, -1]
    return state


def augment_Q(Q):
    """Augment the transition matrix Q such that it is in the from required in order
    to solve. In essence this function gets M and b where:
            - M = the transpose of {the transition matrix Q with the last column
            replaced with a column of ones}
            - b = a vector of the same size as Q with zeros apart from the last
            entry that is 1

    Parameters
    ----------
    Q : numpy.ndarray
        transition matrix

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        The matrix M and vector b to be used to find π such that Mπ = b
    """
    dimension = Q.shape[0]
    M = np.vstack((Q.transpose()[:-1], np.ones(dimension)))
    b = np.vstack((np.zeros((dimension - 1, 1)), [1]))
    return M, b


def get_steady_state_algebraically(Q, algebraic_function=np.linalg.solve):
    """Obtain the steady state of the Markov chain algebraically by either using
    a linear algebraic approach numpy.linalg.solve() or the least squares method
    numpy.linalg.lstsq(). For both methods the following steps are taken:
        - Get M and b from the augment_Q() function
        - Using solve() -> find π such that Mπ=b
        - Using lstsq() -> find π such that the squared Euclidean 2-norm between Mπ and
                           b is minimised

    -> solve():
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.solve.html
    -> lstsq():
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.lstsq.html

    Parameters
    ----------
    Q : numpy.ndarray
        Transition matrix
    algebraic_function : function, optional
        The function to be used to solve the algebraic problem,
        by default np.linalg.solve

    Returns
    -------
    numpy.ndarray
        The steady state vector of the Markov chain
    """
    M, b = augment_Q(Q)
    if algebraic_function == np.linalg.solve:
        state = algebraic_function(M, b).transpose()[0]
    elif algebraic_function == np.linalg.lstsq:
        state = algebraic_function(M, b, rcond=None)[0][:, 0]
    return state


def get_markov_state_probabilities(
    pi, all_states, output=np.ndarray, system_capacity=None, buffer_capacity=None
):
    """Calculates the vector pi in a dictionary format where the values are the
    probabilities that the system is in a current state (listed as key of the
    dictionary).

    Returns
    -------
    dictionary
        A dictionary with the Markov states as keys and the equivalent probabilities
        as values
    """
    if output == dict:
        states_probabilities_dictionary = {}
        for index, _ in enumerate(all_states):
            states_probabilities_dictionary[all_states[index]] = pi[index]
        return states_probabilities_dictionary
    if output == np.ndarray:
        if buffer_capacity is None:
            buffer_capacity = max([state[0] for state in all_states])
        if system_capacity is None:
            system_capacity = max([state[1] for state in all_states])
        states_probabilities_array = np.full(
            (buffer_capacity + 1, system_capacity + 1), np.NaN
        )
        for index, _ in enumerate(all_states):
            states_probabilities_array[all_states[index]] = pi[index]
        return states_probabilities_array
    return None


def get_mean_number_of_individuals_in_system(pi, states):
    """Mean number of individuals in the system = Σ[π_i * (u_i + v_i)]

    Parameters
    ----------
    pi : numpy.ndarray
        steady state vector
    states : list
        list of tuples that contains all states

    Returns
    -------
    float
        Mean number of individuals in the whole model
    """
    states = np.array(states)
    mean_inds_in_system = np.sum((states[:, 0] + states[:, 1]) * pi)
    return mean_inds_in_system


def get_mean_number_of_individuals_in_service_area(pi, states):
    """Mean number of individuals in the service area = Σ[π_i * v_i]

    Parameters
    ----------
    pi : numpy.ndarray
        steady state vector
    states : list
        list of tuples that contains all states

    Returns
    -------
    float
        Mean number of individuals
    """
    states = np.array(states)
    mean_inds_in_service_area = np.sum(states[:, 1] * pi)
    return mean_inds_in_service_area


def get_mean_number_of_individuals_in_buffer_center(pi, states):
    """Mean number of class 2 individuals blocked = Σ[π_i * u_i]

    Parameters
    ----------
    pi : numpy.ndarray
        steady state vector
    states : list
        list of tuples that contains all states

    Returns
    -------
    float
        Mean number of blocked class 2 individuals
    """
    states = np.array(states)
    mean_blocked = np.sum(states[:, 0] * pi)
    return mean_blocked
