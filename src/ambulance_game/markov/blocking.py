import numpy as np

from .markov import (
    build_states,
    get_markov_state_probabilities,
    get_steady_state_algebraically,
    get_transition_matrix,
)
from .utils import (
    expected_time_in_markov_state_ignoring_class_2_arrivals,
    is_accepting_state,
    is_blocking_state,
    prob_class_1_arrival,
    prob_service,
)


def get_coefficients_row_of_array_associated_with_state(
    state, lambda_1, mu, num_of_servers, threshold, system_capacity, buffer_capacity
):
    """Constructs a row of the coefficients matrix. The row to be constructed
    corresponds to the blocking time equation for a given state (u,v) where:

    b(u,v) = c(u,v) + p_s(u,v) * b(u,v−1) + p_o(u,v) * b(u,v+1)

    i.e. the blocking time for state (u,v) is equal to:
        -> the sojourn time of that state PLUS
        -> the probability of service multiplied by the blocking time of
        state (u, v-1) (i.e. the state to end up when a service occurs) PLUS
        -> the probability of class 1 arrivals multiplied by the blocking time
        of state (u, v+1)

    Some other cases of this formula:
        -> when (u,v) not a blocking state: b(u,v) = 0
        -> when v = T: b(u,v) =  c(u,v) + p_s(u,v) * b(u-1,v) + p_o(u,v) * b(u,v+1)
        -> when v = N: (p_s = 1 AND p_o = 0)
                -> if v=T:      b(u,v) = c(u,v) + b(u-1, v)
                -> otherwise:   b(u,v) = c(u,v) + b(u, v-1)

    The main equation can also be written as:
        p_s(u,v) * b(u,v−1) - b(u,v) + p_o(u,v) * b(u,v+1) = -c(u,v)
    where all b(u,v) are considered as unknown variables and
        X = [b(1,T), ... ,b(1,N), b(2,T), ... ,b(2,N), ... , b(M,T), ... , b(M,N)]

    The outputs of this function are:
        - the vector M_{(u,v)} s.t. M_{(u,v)} * X = -c(u,v)
        - The value of -c(u,v)

    Parameters
    ----------
    state : tuple
    lambda_1 : float
    mu : float
    num_of_servers : int
    threshold : int
    system_capacity : int
    buffer_capacity : int

    Returns
    -------
    tuple, float
        the row of the matrix that corresponds to the equation b(u,v) where (u,v)
        is the given state
    """
    if not is_blocking_state(state):
        return 0

    if state[0] >= 1 and state[1] == threshold:
        service_state = (state[0] - 1, state[1])
    else:
        service_state = (state[0], state[1] - 1)
    class_1_arrival_state = (state[0], state[1] + 1)

    lhs_coefficient_row = np.zeros([buffer_capacity, system_capacity - threshold + 1])
    lhs_coefficient_row[state[0] - 1, state[1] - threshold] = -1
    if service_state[0] > 0:
        if state[1] < system_capacity:
            entry = prob_service(
                state=state, lambda_1=lambda_1, mu=mu, num_of_servers=num_of_servers
            )
        else:
            entry = 1
        lhs_coefficient_row[service_state[0] - 1, service_state[1] - threshold] = entry
    if class_1_arrival_state[1] <= system_capacity:
        lhs_coefficient_row[
            class_1_arrival_state[0] - 1, class_1_arrival_state[1] - threshold
        ] = prob_class_1_arrival(state, lambda_1, mu, num_of_servers)
    lhs_coefficient_row = np.reshape(
        lhs_coefficient_row, (1, len(lhs_coefficient_row) * len(lhs_coefficient_row[0]))
    )[0]

    rhs_value = -expected_time_in_markov_state_ignoring_class_2_arrivals(
        state=state,
        lambda_1=lambda_1,
        mu=mu,
        num_of_servers=num_of_servers,
        system_capacity=system_capacity,
    )

    return lhs_coefficient_row, rhs_value


def get_blocking_time_linear_system(
    lambda_1, mu, num_of_servers, threshold, system_capacity, buffer_capacity
):
    """
    Obtain the linear system M X = b by finding the array M and
    the column vector b that are required. Here M is denoted as "all_coefficients_array"
    and b as "constant_column".

    The function stacks the outputs of
    get_coefficients_row_of_array_associated_with_state() for all blocking states
    (i.e. those where u>0) together. In essence all outputs are stacked together
    to form a square matrix (M) and equivalently a column vector (b) that will
    be used to find X s.t. M*X=b

    Parameters
    ----------
    lambda_1 : float
    mu : float
    num_of_servers : int
    threshold : int
    system_capacity : int
    buffer_capacity : int

    Returns
    -------
    numpy.array, list
        The numpy array (M) and the vector (b) such that M*X = b where X is the
        vector with the variables of blocking times per state to be calculated
    """
    all_coefficients_array = np.array([])
    for state in build_states(
        threshold=threshold,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
    ):
        if is_blocking_state(state):
            system_coefficients = get_coefficients_row_of_array_associated_with_state(
                state=state,
                lambda_1=lambda_1,
                mu=mu,
                num_of_servers=num_of_servers,
                threshold=threshold,
                system_capacity=system_capacity,
                buffer_capacity=buffer_capacity,
            )
            if len(all_coefficients_array) == 0:
                all_coefficients_array = [system_coefficients[0]]
                constant_column = [system_coefficients[1]]
            else:
                all_coefficients_array = np.vstack(
                    [all_coefficients_array, system_coefficients[0]]
                )
                constant_column.append(system_coefficients[1])
    return all_coefficients_array, constant_column


def convert_solution_to_correct_array_format(
    array, threshold, system_capacity, buffer_capacity
):
    """Convert the solution into a format that matches the state probabilities array.
    The given array is a one-dimensional array with the blocking times of each state
    given in the following format:
    [b(1,T), b(1,T+1), ... ,b(1,N), b(2,T), ... ,b(2,N), ... , b(M,T), ... , b(M,N)]

    The converted array becomes:

        b(0,0), b(0,1) , ... , b(0,T), ... , b(0,N)
                               b(1,T), ... , b(1,N)
                                  .   .         .
                                  .      .      .
                                  .         .   .
                               b(M,T), ... , b(M,N)

    Parameters
    ----------
    array : numpy.array
        array M to be converted
    threshold : int
    system_capacity : int
    buffer_capacity : int

    Returns
    -------
    numpy.array
        Converted array with dimensions N x M
    """
    new_array = np.reshape(array, (buffer_capacity, system_capacity - threshold + 1))
    top_row = [0 for _ in range(system_capacity - threshold + 1)]
    new_array = np.vstack([top_row, new_array])
    right_columns = [[0 for _ in range(threshold)] for _ in range(buffer_capacity + 1)]
    new_array = np.hstack([right_columns, new_array])
    return new_array


def get_blocking_times_of_all_states_using_direct_approach(
    lambda_1, mu, num_of_servers, threshold, system_capacity, buffer_capacity
):
    """Solve M*X = b using numpy.linalg.solve() where:
        M = The array containing the coefficients of all b(u,v) equations
        b = Vector of constants of equations
        X = All b(u,v) variables of the equations

    Parameters
    ----------
    lambda_1 : float
    mu : float
    num_of_servers : int
    threshold : int
    system_capacity : int
    buffer_capacity : int

    Returns
    -------
    numpy.array
        An MxN array that contains the blocking time for each state
    """
    M, b = get_blocking_time_linear_system(
        lambda_1, mu, num_of_servers, threshold, system_capacity, buffer_capacity
    )
    state_blocking_times = np.linalg.solve(M, b)
    state_blocking_times = convert_solution_to_correct_array_format(
        state_blocking_times, threshold, system_capacity, buffer_capacity
    )
    return state_blocking_times


def mean_blocking_time_formula_using_direct_approach(
    all_states,
    pi,
    lambda_1,
    mu,
    num_of_servers,
    threshold,
    system_capacity,
    buffer_capacity,
):
    """Performs the blocking time formula for the Markov chain model. The formula
    calculates all  blocking times for accepting states and multiplies them with the
    probability of being at that state.

    [Σ b(u,v) * π(u,v)] / [Σ π(u,v)]

    Parameters
    ----------
    all_states : tuple
    pi : numpy.array
    lambda_1 : float
    mu : float
    num_of_servers : float
    threshold : int
    system_capacity : int
    buffer_capacity : int

    Returns
    -------
    float
        the mean blocking time
    """
    mean_blocking_time = 0
    prob_accept_class_2_ind = 0
    blocking_times = get_blocking_times_of_all_states_using_direct_approach(
        lambda_1=lambda_1,
        mu=mu,
        num_of_servers=num_of_servers,
        threshold=threshold,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
    )
    for u, v in all_states:
        if is_accepting_state(
            state=(u, v),
            class_type=1,
            threshold=threshold,
            system_capacity=system_capacity,
            buffer_capacity=buffer_capacity,
        ):
            arriving_state = (u + 1, v) if v >= threshold else (u, v + 1)
            mean_blocking_time += blocking_times[arriving_state] * pi[u, v]
            prob_accept_class_2_ind += pi[u, v]
    return mean_blocking_time / prob_accept_class_2_ind


def mean_blocking_time_formula_using_closed_form_approach(
    all_states,
    pi,
    lambda_1,
    mu,
    num_of_servers,
    threshold,
    system_capacity,
    buffer_capacity,
):
    """
    Get the mean blocking time using the closed form solution.
    """
    # TODO: Build closed-form formula
    raise NotImplementedError("To be implemented")


def get_mean_blocking_time_using_markov_state_probabilities(
    lambda_2,
    lambda_1,
    mu,
    num_of_servers,
    threshold,
    system_capacity,
    buffer_capacity,
    blocking_formula=mean_blocking_time_formula_using_direct_approach,
):
    """Calculates the mean blocking time of the Markov model.

    Parameters
    ----------
    lambda_2 : float
    lambda_1 : float
    mu : float
    num_of_servers : int
    threshold : int
    system_capacity : int
    buffer_capacity : int
    blocking_formula : function

    Returns
    -------
    float
        the mean blocking time of the Markov model
    """
    transition_matrix = get_transition_matrix(
        lambda_2=lambda_2,
        lambda_1=lambda_1,
        mu=mu,
        num_of_servers=num_of_servers,
        threshold=threshold,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
    )
    all_states = build_states(
        threshold=threshold,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
    )
    pi = get_steady_state_algebraically(
        Q=transition_matrix, algebraic_function=np.linalg.solve
    )
    pi = get_markov_state_probabilities(pi, all_states, output=np.ndarray)
    mean_blocking_time = blocking_formula(
        all_states=all_states,
        pi=pi,
        lambda_1=lambda_1,
        mu=mu,
        num_of_servers=num_of_servers,
        threshold=threshold,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
    )
    return mean_blocking_time
