import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
import itertools
import scipy as sci
import scipy.integrate


def build_states(threshold, system_capacity, parking_capacity):
    """Builds the set of states in a list fromat by combine two sets of states where:
        - states_1 consists of all states before reaching the threshold
            (0, 0), ..., (0, T-1) where T is the threshold
        - states_2 consists of all states after reaching the theshold including the threshold
            (0, T), (0, T+1), ..., (0, S)  
            (1, T), (1, T+1), ..., (1, S)
              .         .            .
              .         .            .
              .         .            .
            (P, T), (P, T+1), ..., (P, S)

    Parameters
    ----------
    threshold : int
        Distinguishes between the two sets of states to be combined. In general, if the num of individuals in the hospital >= threshold ambulance patients are not allowed.
    system_capacity : int
        The maximum capacity of the hospital (i.e. number of servers + queue size)
    parking_capacity : int
        The number of parking spaces
    
    Returns
    -------
    list
        a list of all the states
    """
    states_1 = [(0, v) for v in range(0, threshold)]
    states_2 = [
        (u, v)
        for v in range(threshold, system_capacity + 1)
        for u in range(parking_capacity + 1)
    ]
    all_states = states_1 + states_2

    return all_states


######################
## Visualising Model #
######################


def visualise_ambulance_markov_chain(
    num_of_servers, threshold, system_capacity, parking_capacity
):
    """This function's purpose is the visualisation of the markov chain system using the networkx library. The networkx object that is created positions all states based on their (u, v) labels.
    
    Parameters
    ----------
    num_of_servers : int
        All states (u,v) such that v >= num_of_servers are coloured red to indicate the point that the system has no free servers
    threshold : int
        The number where v=threshold that indicates the split of the two sets
    parking_capacity : int
        The maximum number of u in all states (u,v)
    system_capacity : int
        The maximum number of v in all states (u,v)

    Returns
    -------
    object
        a netwrokrx object that consists of the markov chain
    """

    all_states = build_states(threshold, system_capacity, parking_capacity)
    G = nx.MultiDiGraph()
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
                G.add_edge(origin_state, destination_state, color="blue")

    plt.figure(figsize=(1.5 * (parking_capacity + 1), 1.5 * system_capacity))
    pos = {state: list(state) for state in all_states}
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=2000,
        nodelist=[state for state in all_states if state[1] < num_of_servers],
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=2000,
        nodelist=[state for state in all_states if state[1] >= num_of_servers],
        node_color="red",
    )
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos)

    plt.axis("off")

    return G


#######################
## Transition Matrix ##
#######################

def get_transition_matrix_entry(
    origin, destination, threshold, lambda_a, lambda_o, Lambda, mu, num_of_servers
):
    """Obtains the entry of the transition matrix based on the state mapping function. For a given entry of the transition matrix, the function uses the difference between the origin and destination states (u_i,v_i) - (u_j,v_j) along with the threshold to determine what is the rate of going from the origin state to the destenation state.  
    
    This function is used for both the symbolic and numeric transition matrix. 
    
    Parameters
    ----------
    origin : tuple
        The origin state (u_i, v_i) 
    destination : tuple
        The destination state (u_j,v_j)
    threshold : int
        Indication of when to stop using Lambda as the arrival rate and split it into lambda_a and lambda_o
    lambda_a : float or sympy.Symbol object
    lambda_o : float or sympy.Symbol object
    Lambda : float or sympy.Symbol object
        The sum of lambda_a and lambda_o OR the symbol Λ
    mu : float or sympy.Symbol object
    num_of_servers : int
        Indication of when to stablise the service rate
    
    Returns
    -------
    float or sympy.Symbol object
        The numeric or symbolic entry of the matrix
    """
    row_diff = origin[0] - destination[0]
    column_diff = origin[1] - destination[1]

    if row_diff == 0 and column_diff == -1:
        if origin[1] < threshold:
            return Lambda
        return lambda_o
    elif row_diff == -1 and column_diff == 0:
        return lambda_a
    elif row_diff == 0 and column_diff == 1:
        if origin[1] <= num_of_servers:
            return origin[1] * mu
        else:
            return num_of_servers * mu
    elif row_diff == 1 and column_diff == 0 and origin[1] == threshold:
        return threshold * mu
    else:
        return 0


def get_symbolic_transition_matrix(
    num_of_servers, threshold, system_capacity, parking_capacity
):
    """Obtain the transition matrix with symbols instead of the actual values of lambda_a, lambda_o and mu.
    
    Returns
    -------
    sympy.matrices object
        The symbolic transition matrix
    """
    Lambda = sym.symbols("Lambda")
    lambda_o = sym.symbols("lambda") ** sym.symbols("o")
    lambda_a = sym.symbols("lambda") ** sym.symbols("A")
    mu = sym.symbols("mu")

    all_states = build_states(threshold, system_capacity, parking_capacity)
    Q = sym.zeros(len(all_states))

    for (i, origin_state), (j, destination_state) in itertools.product(
        enumerate(all_states), repeat=2
    ):
        Q[i, j] = get_transition_matrix_entry(
            origin=origin_state,
            destination=destination_state,
            threshold=threshold,
            lambda_a=lambda_a,
            lambda_o=lambda_o,
            Lambda=Lambda,
            mu=mu,
            num_of_servers=num_of_servers,
        )

    sum_of_rates = -np.sum(Q, axis=1)
    Q = Q + sym.Matrix(np.diag(sum_of_rates))

    return Q


def get_transition_matrix(
    lambda_a, lambda_o, mu, num_of_servers, threshold, system_capacity, parking_capacity
):
    """Obtain the numerical transition matrix that consists of all rates between any two states.
    
    Parameters
    ----------
    num_of_servers : int
        The number of servers of the hospital
    threshold : int
        The threshold that indicates when to start blocking ambulances
    system_capacity : int
        The total capacity of the system
    parking_capacity : int
        The parking capacity
    
    Returns
    -------
    numpy.ndarray
        The transition matrix Q
    """
    all_states = build_states(threshold, system_capacity, parking_capacity)
    size = len(all_states)
    Q = np.zeros((size, size))

    for (i, origin_state), (j, destination_state) in itertools.product(
        enumerate(all_states), repeat=2
    ):
        Q[i, j] = get_transition_matrix_entry(
            origin=origin_state,
            destination=destination_state,
            threshold=threshold,
            lambda_a=lambda_a,
            lambda_o=lambda_o,
            Lambda=lambda_a + lambda_o,
            mu=mu,
            num_of_servers=num_of_servers,
        )
    sum_of_rates = np.sum(Q, axis=1)
    np.fill_diagonal(Q, -sum_of_rates)
    return Q


def convert_symbolic_transition_matrix(Q_sym, lambda_a, lambda_o, mu):
    """Converts the symbolic matrix obtained from the get_symbolic_trnasition_matrix() function to the corresponding numerical matrix. The output of this function should be the same as the output of get_transition_matrix()
    
    Parameters
    ----------
    Q_sym : sympy.matrices object
        The symbolic transition matrix obtained from get_symbolic_transition_matrix()
    
    Returns
    -------
    numpy.ndarray
        The transition matrix Q
    """
    sym_Lambda = sym.symbols("Lambda")
    sym_lambda_o = sym.symbols("lambda") ** sym.symbols("o")
    sym_lambda_a = sym.symbols("lambda") ** sym.symbols("A")
    sym_mu = sym.symbols("mu")

    Q = np.array(
        Q_sym.subs(
            {
                sym_Lambda: lambda_a + lambda_o,
                sym_lambda_o: lambda_o,
                sym_lambda_a: lambda_a,
                sym_mu: mu,
            }
        )
    ).astype(np.float64)
    return Q


#########################
## Steady state vector ##
#########################


def is_steady_state(state, Q):
    """Checks if a give vector π is a steady state vector of the Markov chain by confirming that:
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
    """Finds the steady state of the Markov chain numerically using either scipy's odeint() or solve_ivp() functions. For each method used a certain set of steps occur:
        - Get an initial state vector (1/n, 1/n, 1/n, ..., 1/n) where n is the total number of states (or size of Q)
        - Enter a loop and exit the loop only when a steady state is found
        - Get the integration interval to be used by the solver: t_span
        - Based on the integration function that will be used, use the correpsonding derivative function
        - Get the state vector and check if it is a steady state
        - if not repeat

    -> odeint(): https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
    -> solve_ivp(): https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp
    
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
    def derivative_odeint(x, t):
        return np.dot(x, Q)

    def derivative_solve_ivp(t, x):
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
    """Augment the trnasition matrix Q such that it is in the from required in order to solve. In essence this function gets M and b where:
            - M = the transpose of {the transition matrix Q with the last column replaced with a column of ones}
            - b = a vector of the same size as Q with zeros apart from the last entry that is 1

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
    """Obtain the steady state of the Markov chain agebraically by either using a linear algebraic approach numpy.linalg.solve() or the least squares method numpy.linalg.lstsq(). For both methods the following steps are taken:
        - Get M and b from the augment_Q() function
        - Using solve() -> find π such that Mπ=b 
        - Using lstsq() -> find π such that the squared Euclidean 2-norm between Mπ and b is minimised

    -> solve(): https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.solve.html
    -> lstsq(): https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.lstsq.html
    
    Parameters
    ----------
    Q : numpy.ndarray
        Transition matrix
    algebraic_function : function, optional
        The function to be used to solve the algebraic problem, by default np.linalg.solve
    
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


##########################
## Performance measures ##
##########################


def get_mean_number_of_patients_in_system(pi, states):
    """Mean number of patients in the system = Σ[π_i * (u_i + v_i)]
    
    Parameters
    ----------
    pi : numpy.ndarray
        steady state vector
    states : list
        list of tuples that contains all states
    
    Returns
    -------
    float
        Mean number of patients in the whole model
    """
    states = np.array(states)
    mean_patients_in_system = np.sum((states[:, 0] + states[:, 1]) * pi)
    return mean_patients_in_system


def get_mean_number_of_patients_in_hospital(pi, states):
    """Mean number of patients in the hospital = Σ[π_i * v_i]
    
    Parameters
    ----------
    pi : numpy.ndarray
        steady state vector
    states : list
        list of tuples that contains all states
    
    Returns
    -------
    float
        Mean number of patients in the hospital
    """
    states = np.array(states)
    mean_patients_in_hospital = np.sum(states[:, 1] * pi)
    return mean_patients_in_hospital


def get_mean_ambulances_blocked(pi, states):
    """Mean number of ambulances blocked = Σ[π_i * u_i]
    
    Parameters
    ----------
    pi : numpy.ndarray
        steady state vector
    states : list
        list of tuples that contains all states
    
    Returns
    -------
    float
        Mean number of blocked ambulances
    """
    states = np.array(states)
    mean_ambulances_blocked = np.sum(states[:, 0] * pi)
    return mean_ambulances_blocked
