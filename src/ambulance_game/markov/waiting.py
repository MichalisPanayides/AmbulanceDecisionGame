import numpy as np
import functools

from .utils import (
    is_waiting_state,
    is_accepting_state,
    expected_time_in_markov_state_ignoring_arrivals,
)

from .markov import (
    build_states,
    get_transition_matrix,
    get_steady_state_algebraically,
    get_markov_state_probabilities,
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
    *args,
    **kwargs,
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
            if class_type == 2 and v >= threshold:
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
    *args,
    **kwargs,
):
    raise NotImplementedError("To be implemented")


def mean_waiting_time_formula_using_closed_form_approach(
    all_states,
    pi,
    class_type,
    mu,
    num_of_servers,
    threshold,
    system_capacity,
    buffer_capacity,
    *args,
    **kwargs,
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
    if class_type == 1:
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
    if class_type == 2:
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
    class_type,
    lambda_2,
    lambda_1,
    mu,
    num_of_servers,
    threshold,
    system_capacity,
    buffer_capacity,
    waiting_formula,
):
    """
    Gets the overall waiting time for all individuals by calculating both class 1
    and class 2 waiting times. Thus, considering the probability that an individual
    is lost to the system (for both classes) calculates the overall waiting time.

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
        for class_type in (1, 2)
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
        for class_type in (1, 2)
    ]

    class_rates = [
        prob_accept[i] / ((lambda_2 * prob_accept[1]) + (lambda_1 * prob_accept[0]))
        for i in range(2)
    ]
    class_rates[0] *= lambda_1
    class_rates[1] *= lambda_2

    mean_waiting_time = np.sum(
        [mean_waiting_times_for_each_class[i] * class_rates[i] for i in range(2)]
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
    class_type=3,
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
    if class_type == 3:
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
