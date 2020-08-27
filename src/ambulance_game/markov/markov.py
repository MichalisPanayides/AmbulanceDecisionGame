import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
import itertools
import scipy as sci
import scipy.integrate
import functools


def build_states(threshold, system_capacity, parking_capacity):
    """Builds the set of states in a list fromat by combine two sets of states where:
        - states_1 consists of all states before reaching the threshold
            (0, 0), (0, 1), ..., (0, T-1) where T is the threshold
        - states_2 consists of all states after reaching the theshold including the threshold (where S is the system capacity)
            (0, T), (0, T+1), ..., (0, S)
            (1, T), (1, T+1), ..., (1, S)
              .         .            .
              .         .            .
              .         .            .
            (P, T), (P, T+1), ..., (P, S)

    Note that if the threshold is greater than the system_capacity then the markov chain will be of the form:
         (0, 0), (0, 1), ..., (0, S)
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

    TODO: turn into a generator
    """
    if parking_capacity < 1:
        raise ValueError(
            "Simulation only implemented for parking_capacity >= 1"
        )  # TODO Add an option to ciw model to all for no parking capacity.

    if threshold > system_capacity:
        return [(0, v) for v in range(0, system_capacity + 1)]
        # states_1 = [(0, v) for v in range(0, system_capacity + 1)]
        # states_2 = [(1, system_capacity)]
        # return states_1 + states_2

    states_1 = [(0, v) for v in range(0, threshold)]
    states_2 = [
        (u, v)
        for v in range(threshold, system_capacity + 1)
        for u in range(parking_capacity + 1)
    ]
    all_states = states_1 + states_2

    return all_states


def visualise_ambulance_markov_chain(
    num_of_servers,
    threshold,
    system_capacity,
    parking_capacity,
    nodesize=2000,
    fontsize=12,
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
    G = nx.DiGraph()
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

    plt.figure(figsize=((system_capacity + 1) * 1.5, (parking_capacity + 1) * 1.5))
    pos = {state: [state[1], -state[0]] for state in all_states}
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=nodesize,
        nodelist=[state for state in all_states if state[1] < num_of_servers],
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=nodesize,
        nodelist=[state for state in all_states if state[1] >= num_of_servers],
        node_color="red",
    )
    nx.draw_networkx_edges(G, pos, arrowstyle="fancy")
    nx.draw_networkx_labels(G, pos, font_size=fontsize)

    plt.axis("off")

    return G


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
    delta = np.array(origin) - np.array(destination)
    if np.all(delta == (0, -1)):
        if origin[1] < threshold:
            return Lambda
        return lambda_o
    if np.all(delta == (-1, 0)):
        return lambda_a
    if np.all(delta == (0, 1)) or (np.all(delta == (1, 0)) and origin[1] == threshold):
        return min(origin[1], num_of_servers) * mu
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
    lambda_o = sym.symbols("lambda^o")
    lambda_a = sym.symbols("lambda^A")
    mu = sym.symbols("mu")

    all_states = build_states(threshold, system_capacity, parking_capacity)
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
    # if threshold > system_capacity:
    #     threshold = system_capacity
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

    TODO: get rid of first four lines somehow
    """
    sym_Lambda = sym.symbols("Lambda")
    sym_lambda_o = sym.symbols("lambda^o")
    sym_lambda_a = sym.symbols("lambda^A")
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


def get_markov_state_probabilities(
    pi, all_states, output=np.ndarray, system_capacity=None, parking_capacity=None
):
    """Calculates the vector pi in a dictionary format where the values are the probabilities that the system is in a cuurent state (listed as key of the dictionary).

    Returns
    -------
    dictionary
        A dictionary with the markov states as keys and the equivalent probabilities as values
    """
    if output == dict:
        states_probabilities_dictionary = {}
        for i in range(len(all_states)):
            states_probabilities_dictionary[all_states[i]] = pi[i]
        return states_probabilities_dictionary
    elif output == np.ndarray:
        if parking_capacity == None:
            parking_capacity = max([state[0] for state in all_states])
        if system_capacity == None:
            system_capacity = max([state[1] for state in all_states])
        states_probabilities_array = np.full(
            (parking_capacity + 1, system_capacity + 1), np.NaN
        )
        for index in range(len(all_states)):
            states_probabilities_array[all_states[index]] = pi[index]
        return states_probabilities_array


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


def get_mean_number_of_ambulances_blocked(pi, states):
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


def is_waiting_state(state, num_of_servers):
    """Checks if waiting occurs in the given state. In essence, all states (u,v) where v < C are not considered waiting states.
    Set of waiting states: S_w = {(u,v) ∈ S | v > C}

    Parameters
    ----------
    state : tuple
        a tuples of the form (u,v)
    num_of_servers : int
        the number of servers = C
    Returns
    -------
    boolean
        An indication of whether or not any wait occurs on the given state
    """
    return state[1] > num_of_servers


def is_accepting_state(
    state, patient_type, threshold, system_capacity, parking_capacity
):
    """Checks if a state given is an accepting state. Accepting states are defined as the states of the system where patient arrivals may occur. In essence these states are all states apart from the one when the system cannot accept additional arrivals. Because there are two types of patients arrival though, the set of accepting states is different for ambulance and other patients:

    Ambulance patients: S_A = {(u,v) ∈ S | u < N}
    Other patients: S_A = {(u,v) ∈ S | v < M}

    Parameters
    ----------
    state : tuple
        a tuples of the form (u,v)
    patient_type : string
        A string to distingish between ambulance and other patients
    system_capacity : int
        The capacity of the system (hospital) = N
    parking_capacity : int
        The capacity of the parking space = M

    Returns
    -------
    boolean
        An indication of whether or not an arrival of the given type (patient_type) can occur
    """
    if patient_type == "ambulance":
        condition = (
            (state[0] < parking_capacity)
            if (threshold <= system_capacity)
            else (state[1] < system_capacity)
        )
    if patient_type == "others":
        condition = state[1] < system_capacity
    return condition


def expected_time_in_markov_state_ignoring_arrivals(
    state,
    patient_type,
    num_of_servers,
    mu,
    threshold,
):
    """Get the expected waiting time in a markov state when ignoring any subsequent arrivals.When considering ambulance patients waiting time and the patients are in a blocked state (v > 0) then by the definition of the problem the waiting time in sthat state is set to 0. Additionally, all states where u > 0 and v = T automatically get a waiting time of 0 because other patients only pass one of the states of that column (only state (0,T) is not zero). Otherwise the function's output is:
        - c(u,v) = 1/vμ   if v < C
        - c(u,v) = 1/Cμ   if v >= C

    Parameters
    ----------
    state : tuple
        a tuples of the form (u,v)
    patient_type : string
        A string to distingish between ambulance and other patients
    num_of_servers : int
        The number of servers = C
    mu : float
        The service rate = μ

    Returns
    -------
    float
        The expected waiting time in the given state
    """
    if state[0] > 0 and (state[1] == threshold or patient_type == "ambulance"):
        return 0
    return 1 / (min(state[1], num_of_servers) * mu)


@functools.lru_cache(maxsize=None)
def get_recursive_waiting_time(
    state,
    patient_type,
    lambda_a,
    lambda_o,
    mu,
    num_of_servers,
    threshold,
    system_capacity,
    parking_capacity,
):
    """Performs a recursive algorithm to get the expected waiting time of patients when they enter the model at a given state. Given an arriving state the algorithm moves down to all subsequent states until it reaches one that is not a waiting state.

    Others:
        - If (u,v) not a waiting state: return 0
        - Arriving state s_a = (u, v + 1)
        - Next state s_d = (0, v - 1)
        - w(u,v) = c(s_a) + w(s_d)

    Ambulance:
        - If (u,v) not a waiting state: return 0
        - Next state:       s_n = (u-1, v),    if u >=1 and v=T
                            s_n = (u, v - 1),  otherwise
        - w(u,v) = c(u,v) + w(s_n)

    Note:   For all "others" patients the recursive formula acts in a linear manner meaning that an individual will have the same waiting time when arriving at either state of the same column e.g (2, 3) or (5, 3).

    Parameters
    ----------
    state : tuple
    patient_type : string
    lambda_a : float
    lambda_o : float
    mu : float
    num_of_servers : int
    threshold : int
    system_capacity : int
    parking_capacity : int

    Returns
    -------
    float
        The expected waiting time from the arriving state of an individual until service
    """
    if not is_waiting_state(state, num_of_servers):
        return 0
    if state[0] >= 1 and state[1] == threshold:
        next_state = (state[0] - 1, state[1])
    else:
        next_state = (state[0], state[1] - 1)

    wait = expected_time_in_markov_state_ignoring_arrivals(
        state, patient_type, num_of_servers, mu, threshold
    )
    wait += get_recursive_waiting_time(
        next_state,
        patient_type,
        lambda_a,
        lambda_o,
        mu,
        num_of_servers,
        threshold,
        system_capacity,
        parking_capacity,
    )
    return wait


def mean_waiting_time_formula(
    all_states,
    pi,
    patient_type,
    lambda_a,
    lambda_o,
    mu,
    num_of_servers,
    threshold,
    system_capacity,
    parking_capacity,
    formula="closed_form",
):
    """Get the mean waiting time by using the recursive formula or a closed-form formula

    Recursive Formula:
        W = Σ[w(u,v) * π(u,v)] / Σ[π(u,v)] ,

        where:  - both summations occur over all accepting states (u,v)
                - w(u,v) is the recursive waiting time of state (u,v)
                - π(u,v) is the probability of being at state (u,v)

    Parameters
    ----------
    all_states : list
    pi : array
    patient_type : str
    lambda_a : float
    lambda_o : float
    mu : float
    num_of_servers : int
    threshold : int
    system_capacity : int
    parking_capacity : int
    formula : str, optional

    Returns
    -------
    float
        The mean waiting time for the specified patient type
    """
    if formula == "recursive":
        mean_waiting_time = 0
        probability_of_accepting = 0
        for u, v in all_states:
            if is_accepting_state(
                (u, v), patient_type, threshold, system_capacity, parking_capacity
            ):
                arriving_state = (u, v + 1)
                if patient_type == "ambulance" and v >= threshold:
                    arriving_state = (u + 1, v)

                current_state_wait = get_recursive_waiting_time(
                    arriving_state,
                    patient_type,
                    lambda_a,
                    lambda_o,
                    mu,
                    num_of_servers,
                    threshold,
                    system_capacity,
                    parking_capacity,
                )
                mean_waiting_time += current_state_wait * pi[u, v]
                probability_of_accepting += pi[u, v]
        mean_waiting_time /= probability_of_accepting

    if formula == "closed_form":
        sojourn_time = 1 / (num_of_servers * mu)
        if patient_type == "others":
            mean_waiting_time = np.sum(
                [
                    (state[1] - num_of_servers + 1) * pi[state] * sojourn_time
                    for state in all_states
                    if is_accepting_state(
                        state,
                        patient_type,
                        threshold,
                        system_capacity,
                        parking_capacity,
                    )
                    and state[1] >= num_of_servers
                ]
            ) / np.sum(
                [
                    pi[state]
                    for state in all_states
                    if is_accepting_state(
                        state,
                        patient_type,
                        threshold,
                        system_capacity,
                        parking_capacity,
                    )
                ]
            )
        if patient_type == "ambulance":
            mean_waiting_time = np.sum(
                [
                    (min(state[1] + 1, threshold) - num_of_servers)
                    * pi[state]
                    * sojourn_time
                    for state in all_states
                    if is_accepting_state(
                        state,
                        patient_type,
                        threshold,
                        system_capacity,
                        parking_capacity,
                    )
                    and min(state[1], threshold) >= num_of_servers
                ]
            ) / np.sum(
                [
                    pi[state]
                    for state in all_states
                    if is_accepting_state(
                        state,
                        patient_type,
                        threshold,
                        system_capacity,
                        parking_capacity,
                    )
                ]
            )

    return mean_waiting_time


def get_mean_waiting_time_markov(
    lambda_a,
    lambda_o,
    mu,
    num_of_servers,
    threshold,
    system_capacity,
    parking_capacity,
    output="both",
    formula="closed_form",
):
    """Gets the mean waiting time for a markov chain model

    Parameters
    ----------
    lambda_a : float
    lambda_o : float
    mu : float
    num_of_servers : int
    threshold : int
    system_capacity : int
    parking_capacity : int
    formula : str, optional
    output : str, optional

    Returns
    -------
    float
        The mean waiting time of the in the system of either ambulance patients, other patients or the overall of both
    """
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
        transition_matrix, algebraic_function=np.linalg.solve
    )
    all_states = build_states(threshold, system_capacity, parking_capacity)
    state_probabilities = get_markov_state_probabilities(
        pi, all_states, output=np.ndarray
    )
    if output == "both":
        mean_waiting_time_other = mean_waiting_time_formula(
            all_states,
            state_probabilities,
            "others",
            lambda_a,
            lambda_o,
            mu,
            num_of_servers,
            threshold,
            system_capacity,
            parking_capacity,
            formula=formula,
        )
        mean_waiting_time_ambulance = mean_waiting_time_formula(
            all_states,
            state_probabilities,
            "ambulance",
            lambda_a,
            lambda_o,
            mu,
            num_of_servers,
            threshold,
            system_capacity,
            parking_capacity,
            formula=formula,
        )

        prob_accept_others = np.sum(
            [
                state_probabilities[state]
                for state in all_states
                if is_accepting_state(
                    state, "others", threshold, system_capacity, parking_capacity
                )
            ]
        )
        prob_accept_ambulance = np.sum(
            [
                state_probabilities[state]
                for state in all_states
                if is_accepting_state(
                    state, "ambulance", threshold, system_capacity, parking_capacity
                )
            ]
        )

        ambulance_rate = (lambda_a * prob_accept_ambulance) / (
            (lambda_a * prob_accept_ambulance) + (lambda_o * prob_accept_others)
        )
        others_rate = (lambda_o * prob_accept_others) / (
            (lambda_a * prob_accept_ambulance) + (lambda_o * prob_accept_others)
        )

        return (
            mean_waiting_time_ambulance * ambulance_rate
            + mean_waiting_time_other * others_rate
        )

    mean_waiting_time = mean_waiting_time_formula(
        all_states,
        state_probabilities,
        output,
        lambda_a,
        lambda_o,
        mu,
        num_of_servers,
        threshold,
        system_capacity,
        parking_capacity,
        formula=formula,
    )
    return mean_waiting_time
