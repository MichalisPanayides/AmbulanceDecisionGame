"""
Code to calculate the mean waiting time.
"""

import functools
import itertools

import numpy as np

from .markov import (
    build_states,
    get_markov_state_probabilities,
    get_steady_state_algebraically,
    get_transition_matrix,
)
from .utils import (
    expected_time_in_markov_state_ignoring_arrivals,
    is_accepting_state,
    is_waiting_state,
)


@functools.lru_cache(maxsize=None)
def get_waiting_time_for_each_state_recursively(
    state,
    class_type,
    lambda_2,
    lambda_1,
    mu,
    num_of_servers,
    threshold,
    system_capacity,
    buffer_capacity,
):
    """Performs a recursive algorithm to get the expected waiting time of individuals
    when they enter the model at a given state. Given an arriving state the algorithm
    moves down to all subsequent states until it reaches one that is not a waiting
    state.

    Class 1:
        - If (u,v) not a waiting state: return 0
        - Next state s_d = (0, v - 1)
        - w(u,v) = c(u,v) + w(s_d)

    Class 2:
        - If (u,v) not a waiting state: return 0
        - Next state:   s_n = (u-1, v),    if u >= 1 and v=T
                        s_n = (u, v - 1),  otherwise
        - w(u,v) = c(u,v) + w(s_n)

    Note: For all class 1 individuals the recursive formula acts in a linear manner
    meaning that an individual will have the same waiting time when arriving at
    any state of the same column e.g (2, 3) or (5, 3).

    Parameters
    ----------
    state : tuple
    class_type : int
    lambda_2 : float
    lambda_1 : float
    mu : float
    num_of_servers : int
    threshold : int
    system_capacity : int
    buffer_capacity : int

    Returns
    -------
    float
        The expected waiting time from the arriving state of an individual until
        service
    """
    if not is_waiting_state(state, num_of_servers):
        return 0
    if state[0] >= 1 and state[1] == threshold:
        next_state = (state[0] - 1, state[1])
    else:
        next_state = (state[0], state[1] - 1)

    wait = expected_time_in_markov_state_ignoring_arrivals(
        state=state,
        class_type=class_type,
        num_of_servers=num_of_servers,
        mu=mu,
        threshold=threshold,
    )
    wait += get_waiting_time_for_each_state_recursively(
        state=next_state,
        class_type=class_type,
        lambda_2=lambda_2,
        lambda_1=lambda_1,
        mu=mu,
        num_of_servers=num_of_servers,
        threshold=threshold,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
    )
    return wait


def mean_waiting_time_formula_using_recursive_approach(
    all_states,
    pi,
    class_type,
    lambda_2,
    lambda_1,
    mu,
    num_of_servers,
    threshold,
    system_capacity,
    buffer_capacity,
    **kwargs,  # pylint: disable=unused-argument
):
    """
    Get the mean waiting time by using a recursive formula.
    This function solves the following expression:

    W = Σ[w(u,v) * π(u,v)] / Σ[π(u,v)] ,

    where:  - both summations occur over all accepting states (u,v)
            - w(u,v) is the recursive waiting time of state (u,v)
            - π(u,v) is the probability of being at state (u,v)

    All w(u,v) terms are calculated recursively by going through the waiting
    times of all previous states.

    Parameters
    ----------
    all_states : list
    pi : array
    class_type : int
    lambda_2 : float
    lambda_1 : float
    mu : float
    num_of_servers : int
    threshold : int
    system_capacity : int
    buffer_capacity : int

    Returns
    -------
    float
    """
    mean_waiting_time = 0
    probability_of_accepting = 0
    for u, v in all_states:
        if is_accepting_state(
            state=(u, v),
            class_type=class_type,
            threshold=threshold,
            system_capacity=system_capacity,
            buffer_capacity=buffer_capacity,
        ):
            arriving_state = (u, v + 1)
            if class_type == 1 and v >= threshold:
                arriving_state = (u + 1, v)

            current_state_wait = get_waiting_time_for_each_state_recursively(
                state=arriving_state,
                class_type=class_type,
                lambda_2=lambda_2,
                lambda_1=lambda_1,
                mu=mu,
                num_of_servers=num_of_servers,
                threshold=threshold,
                system_capacity=system_capacity,
                buffer_capacity=buffer_capacity,
            )
            mean_waiting_time += current_state_wait * pi[u, v]
            probability_of_accepting += pi[u, v]
    return mean_waiting_time / probability_of_accepting


def get_coefficients_row_of_array_for_state(
    state, class_type, mu, num_of_servers, threshold, system_capacity, buffer_capacity
):
    """
    For direct approach: Constructs a row of the coefficients matrix. The row to
    be constructed corresponds to the waiting time equation for a given state
    (u,v) where:

    w(u,v) = 0                      , if (u,v) not in WaitingStates
           = c(u,v) + w(u - 1, v)   , if u > 0 and v = T
           = c(u,v) + w(u, v - 1)   ,

    i.e. the waiting time for state (u,v) is equal to:
        -> the sojourn time of that state PLUS
        -> the waiting time of the next state

    The equations can also be written as:
        -w(u,v) + w(u - 1, v) = -c(u,v)
        -w(u,v) + w(u, v - 1) = -c(u,v)
    where all w(u,v) are considered as unknown variables and
        X = [w(1,T), ... ,w(1,N), w(2,T), ... ,w(2,N), ... , w(M,T), ... , w(M,N)]

    The outputs of this function are:
        - the vector M_{(u,v)} s.t. M_{(u,v)} * X = -c(u,v)
        - The value of -c(u,v)
    """
    lhs_coefficient_row = np.zeros([buffer_capacity + 1, system_capacity + 1])
    lhs_coefficient_row[state[0], state[1]] = -1
    for (u, v) in itertools.product(range(1, buffer_capacity + 1), range(threshold)):
        lhs_coefficient_row[u, v] = np.NaN

    rhs_value = 0
    if is_waiting_state(state, num_of_servers):
        if state[0] >= 1 and state[1] == threshold:
            next_state = (state[0] - 1, state[1])
        else:
            next_state = (state[0], state[1] - 1)

        lhs_coefficient_row[next_state[0], next_state[1]] = 1
        rhs_value = -expected_time_in_markov_state_ignoring_arrivals(
            state=state,
            class_type=class_type,
            mu=mu,
            num_of_servers=num_of_servers,
            threshold=threshold,
        )

    vectorised_array = np.hstack(
        (
            lhs_coefficient_row[0, :threshold],
            lhs_coefficient_row[:, threshold:].flatten("F"),
        )
    )
    return vectorised_array, rhs_value


def get_waiting_time_linear_system(
    class_type, mu, num_of_servers, threshold, system_capacity, buffer_capacity
):
    """
    For direct approach: Obtain the linear system M X = b by finding the array M
    and the column vector b that are required. Here M is denoted as
    "all_coefficients_array" and b as "constant_column".

    The function stacks the outputs of get_coefficients_row_of_array_for_state()
    for all states. In essence all outputs are stacked together to form a square
    matrix (M) and equivalently a column vector (b) that will be used to find X
    s.t. M*X=b
    """
    all_coefficients_array = np.array([])
    all_states = build_states(
        threshold=threshold,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
    )
    for state in all_states:
        lhs_vector, rhs_value = get_coefficients_row_of_array_for_state(
            state=state,
            class_type=class_type,
            mu=mu,
            num_of_servers=num_of_servers,
            threshold=threshold,
            system_capacity=system_capacity,
            buffer_capacity=buffer_capacity,
        )
        if len(all_coefficients_array) == 0:
            all_coefficients_array = [lhs_vector]
            constant_column = [rhs_value]
        else:
            all_coefficients_array = np.vstack([all_coefficients_array, lhs_vector])
            constant_column.append(rhs_value)
    return all_coefficients_array, constant_column


def convert_solution_to_correct_array_format(
    array, all_states, system_capacity, buffer_capacity
):
    """
    For direct approach: Convert the solution into a format that matches the
    state probabilities array. The given array is a one-dimensional array with
    the blocking times of each state given in the following format:
    [w(1,T), w(1,T+1), ... ,w(1,N), w(2,T), ... ,w(2,N), ... , w(M,T), ... , w(M,N)]

    The converted array becomes:

        w(0,0), w(0,1) , ... , w(0,T), ... , w(0,N)
                               w(1,T), ... , w(1,N)
                                  .   .         .
                                  .      .      .
                                  .         .   .
                               w(M,T), ... , w(M,N)
    """
    array_with_correct_shape = np.zeros([buffer_capacity + 1, system_capacity + 1])
    for index, (u, v) in enumerate(all_states):
        array_with_correct_shape[u, v] = array[index]
    return array_with_correct_shape


def get_waiting_times_of_all_states_using_direct_approach(
    class_type,
    all_states,
    mu,
    num_of_servers,
    threshold,
    system_capacity,
    buffer_capacity,
):
    """
    For direct approach: Solve M*X = b using numpy.linalg.solve() where:
        M = The array containing the coefficients of all w(u,v) equations
        b = Vector of constants of equations
        X = All w(u,v) variables of the equations
    """
    M, b = get_waiting_time_linear_system(
        class_type=class_type,
        mu=mu,
        num_of_servers=num_of_servers,
        threshold=threshold,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
    )
    state_waiting_times = np.linalg.solve(M, b)
    state_waiting_times = convert_solution_to_correct_array_format(
        array=state_waiting_times,
        all_states=all_states,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
    )
    return state_waiting_times


def mean_waiting_time_formula_using_direct_approach(
    all_states,
    pi,
    class_type,
    lambda_2,
    lambda_1,
    mu,
    num_of_servers,
    threshold,
    system_capacity,
    buffer_capacity,
    **kwargs,  # pylint: disable=unused-argument
):
    """
    Get the mean waiting time by using a direct approach.
    """
    waiting_times = get_waiting_times_of_all_states_using_direct_approach(
        class_type=class_type,
        all_states=all_states,
        mu=mu,
        num_of_servers=num_of_servers,
        threshold=threshold,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
    )

    mean_waiting_time, prob_accept_class_2_ind = 0, 0
    for (u, v) in all_states:
        if is_accepting_state(
            state=(u, v),
            class_type=class_type,
            threshold=threshold,
            system_capacity=system_capacity,
            buffer_capacity=buffer_capacity,
        ):
            arriving_state = (u, v + 1)
            if class_type == 1 and v >= threshold:
                arriving_state = (u + 1, v)
            mean_waiting_time += waiting_times[arriving_state] * pi[u, v]
            prob_accept_class_2_ind += pi[u, v]

    return mean_waiting_time / prob_accept_class_2_ind


def mean_waiting_time_formula_using_closed_form_approach(
    all_states,
    pi,
    class_type,
    mu,
    num_of_servers,
    threshold,
    system_capacity,
    buffer_capacity,
    **kwargs,  # pylint: disable=unused-argument
):
    """
    Get the mean waiting time by using a closed-form formula.

    Parameters
    ----------
    all_states : list
    pi : array
    class_type : int
    mu : float
    num_of_servers : int
    threshold : int
    system_capacity : int
    buffer_capacity : int

    Returns
    -------
    float
    """
    sojourn_time = 1 / (num_of_servers * mu)
    if class_type == 0:
        mean_waiting_time = np.sum(
            [
                (state[1] - num_of_servers + 1) * pi[state] * sojourn_time
                for state in all_states
                if is_accepting_state(
                    state=state,
                    class_type=class_type,
                    threshold=threshold,
                    system_capacity=system_capacity,
                    buffer_capacity=buffer_capacity,
                )
                and state[1] >= num_of_servers
            ]
        ) / np.sum(
            [
                pi[state]
                for state in all_states
                if is_accepting_state(
                    state=state,
                    class_type=class_type,
                    threshold=threshold,
                    system_capacity=system_capacity,
                    buffer_capacity=buffer_capacity,
                )
            ]
        )
    # TODO: Break function into 2 functions
    if class_type == 1:
        mean_waiting_time = np.sum(
            [
                (min(state[1] + 1, threshold) - num_of_servers)
                * pi[state]
                * sojourn_time
                for state in all_states
                if is_accepting_state(
                    state=state,
                    class_type=class_type,
                    threshold=threshold,
                    system_capacity=system_capacity,
                    buffer_capacity=buffer_capacity,
                )
                and min(state[1], threshold) >= num_of_servers
            ]
        ) / np.sum(
            [
                pi[state]
                for state in all_states
                if is_accepting_state(
                    state=state,
                    class_type=class_type,
                    threshold=threshold,
                    system_capacity=system_capacity,
                    buffer_capacity=buffer_capacity,
                )
            ]
        )
    return mean_waiting_time


def overall_waiting_time_formula(
    all_states,
    pi,
    lambda_2,
    lambda_1,
    mu,
    num_of_servers,
    threshold,
    system_capacity,
    buffer_capacity,
    waiting_formula,
    **kwargs,  # pylint: disable=unused-argument
):
    """
    Gets the overall waiting time for all individuals by calculating both class 1
    and class 2 waiting times. Thus, considering the probability that an individual
    is lost to the system (for both classes) calculates the overall waiting time.

    Parameters
    ----------
    all_states : list
    pi : array
    lambda_1 : float
    lambda_2 : float
    mu : float
    num_of_servers : int
    threshold : int
    system_capacity : int
    buffer_capacity : int
    waiting_formula : function

    Returns
    -------
    float
        The overall mean waiting time by combining class 1 and class 2 individuals
    """
    mean_waiting_times_for_each_class = [
        waiting_formula(
            all_states=all_states,
            pi=pi,
            class_type=class_type,
            lambda_2=lambda_2,
            lambda_1=lambda_1,
            mu=mu,
            num_of_servers=num_of_servers,
            threshold=threshold,
            system_capacity=system_capacity,
            buffer_capacity=buffer_capacity,
        )
        for class_type in range(2)
    ]

    prob_accept = [
        np.sum(
            [
                pi[state]
                for state in all_states
                if is_accepting_state(
                    state=state,
                    class_type=class_type,
                    threshold=threshold,
                    system_capacity=system_capacity,
                    buffer_capacity=buffer_capacity,
                )
            ]
        )
        for class_type in range(2)
    ]

    class_rates = [
        prob_accept[class_type]
        / ((lambda_2 * prob_accept[1]) + (lambda_1 * prob_accept[0]))
        for class_type in range(2)
    ]
    class_rates[0] *= lambda_1
    class_rates[1] *= lambda_2

    mean_waiting_time = np.sum(
        [
            mean_waiting_times_for_each_class[class_type] * class_rates[class_type]
            for class_type in range(2)
        ]
    )
    return mean_waiting_time


def get_mean_waiting_time_using_markov_state_probabilities(
    lambda_2,
    lambda_1,
    mu,
    num_of_servers,
    threshold,
    system_capacity,
    buffer_capacity,
    class_type=None,
    waiting_formula=mean_waiting_time_formula_using_closed_form_approach,
):
    """
    Gets the mean waiting time by using either the recursive formula,
    closed-form formula or the direct approach. This function solves the
    following expression:

    W = Σ[w(u,v) * π(u,v)] / Σ[π(u,v)] ,

    where:  - both summations occur over all accepting states (u,v)
            - w(u,v) is the recursive waiting time of state (u,v)
            - π(u,v) is the probability of being at state (u,v)

    All three formulas aim to solve the same expression by using different
    approaches to calculate the terms w(u,v).

    Parameters
    ----------
    lambda_2 : float
    lambda_1 : float
    mu : float
    num_of_servers : int
    threshold : int
    system_capacity : int
    buffer_capacity : int
    class_type : int, optional
    formula : str, optional

    Returns
    -------
    float
        The mean waiting time in the system of either class 1,
        class 2 individuals or the overall of both
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
    pi = get_markov_state_probabilities(pi=pi, all_states=all_states, output=np.ndarray)
    if class_type is None:
        get_mean_waiting_time = overall_waiting_time_formula
    else:
        get_mean_waiting_time = waiting_formula

    mean_waiting_time = get_mean_waiting_time(
        all_states=all_states,
        pi=pi,
        class_type=class_type,
        lambda_2=lambda_2,
        lambda_1=lambda_1,
        mu=mu,
        num_of_servers=num_of_servers,
        threshold=threshold,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
        waiting_formula=waiting_formula,
    )

    return mean_waiting_time
