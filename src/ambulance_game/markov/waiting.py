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
    patient_type,
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
    patient_type : string
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
        state, patient_type, num_of_servers, mu, threshold
    )
    wait += get_waiting_time_for_each_state_recursively(
        next_state,
        patient_type,
        lambda_2,
        lambda_1,
        mu,
        num_of_servers,
        threshold,
        system_capacity,
        buffer_capacity,
    )
    return wait


def mean_waiting_time_formula_using_recursive_approach(
    all_states,
    pi,
    patient_type,
    lambda_2,
    lambda_1,
    mu,
    num_of_servers,
    threshold,
    system_capacity,
    buffer_capacity,
    *args
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
    patient_type : str
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
            (u, v), patient_type, threshold, system_capacity, buffer_capacity
        ):
            arriving_state = (u, v + 1)
            if patient_type == "ambulance" and v >= threshold:
                arriving_state = (u + 1, v)

            current_state_wait = get_waiting_time_for_each_state_recursively(
                arriving_state,
                patient_type,
                lambda_2,
                lambda_1,
                mu,
                num_of_servers,
                threshold,
                system_capacity,
                buffer_capacity,
            )
            mean_waiting_time += current_state_wait * pi[u, v]
            probability_of_accepting += pi[u, v]
    return mean_waiting_time / probability_of_accepting


def mean_waiting_time_formula_using_algebraic_approach(
    all_states,
    pi,
    patient_type,
    lambda_2,
    lambda_1,
    mu,
    num_of_servers,
    threshold,
    system_capacity,
    buffer_capacity,
    *args
):
    raise NotImplementedError("To be implemented")


def mean_waiting_time_formula_using_closed_form_approach(
    all_states,
    pi,
    patient_type,
    mu,
    num_of_servers,
    threshold,
    system_capacity,
    buffer_capacity,
    **kwargs
):
    """
    Get the mean waiting time by using a closed-form formula.

    Parameters
    ----------
    all_states : list
    pi : array
    patient_type : str
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
                    buffer_capacity,
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
                    buffer_capacity,
                )
            ]
        )
    # TODO: Break function into 2 functions
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
                    buffer_capacity,
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
                    buffer_capacity,
                )
            ]
        )
    return mean_waiting_time


def mean_waiting_time_formula(
    all_states,
    pi,
    patient_type,
    lambda_2,
    lambda_1,
    mu,
    num_of_servers,
    threshold,
    system_capacity,
    buffer_capacity,
    formula="closed_form",
):
    """
    Get the mean waiting time by using either the recursive formula,
    closed-form formula or the algebraic formula. This function solves the
    following expression:

    W = Σ[w(u,v) * π(u,v)] / Σ[π(u,v)] ,

    where:  - both summations occur over all accepting states (u,v)
            - w(u,v) is the recursive waiting time of state (u,v)
            - π(u,v) is the probability of being at state (u,v)

    All three formulas aim to solve the same expression by using different
    approaches to calculate the terms w(u,v).

    Parameters
    ----------
    all_states : list
    pi : array
    patient_type : str
    lambda_2 : float
    lambda_1 : float
    mu : float
    num_of_servers : int
    threshold : int
    system_capacity : int
    buffer_capacity : int
    formula : str, optional

    Returns
    -------
    float
        The mean waiting time for the specified class
    """
    if formula == "recursive":
        mean_waiting_time = mean_waiting_time_formula_using_recursive_approach(
            all_states=all_states,
            pi=pi,
            patient_type=patient_type,
            lambda_2=lambda_2,
            lambda_1=lambda_1,
            mu=mu,
            num_of_servers=num_of_servers,
            threshold=threshold,
            system_capacity=system_capacity,
            buffer_capacity=buffer_capacity,
        )

    if formula == "algebraic":
        mean_waiting_time = mean_waiting_time_formula_using_algebraic_approach(
            all_states=all_states,
            pi=pi,
            patient_type=patient_type,
            lambda_2=lambda_2,
            lambda_1=lambda_1,
            mu=mu,
            num_of_servers=num_of_servers,
            threshold=threshold,
            system_capacity=system_capacity,
            buffer_capacity=buffer_capacity,
        )

    if formula == "closed_form":
        mean_waiting_time = mean_waiting_time_formula_using_closed_form_approach(
            all_states=all_states,
            pi=pi,
            patient_type=patient_type,
            lambda_2=lambda_2,
            lambda_1=lambda_1,
            mu=mu,
            num_of_servers=num_of_servers,
            threshold=threshold,
            system_capacity=system_capacity,
            buffer_capacity=buffer_capacity,
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
    patient_type="both",
    formula="closed_form",
):
    """Gets the mean waiting time for a Markov chain model

    Parameters
    ----------
    lambda_2 : float
    lambda_1 : float
    mu : float
    num_of_servers : int
    threshold : int
    system_capacity : int
    buffer_capacity : int
    patient_type : str, optional
    formula : str, optional

    Returns
    -------
    float
        The mean waiting time in the system of either class 1,
        class 2 individuals or the overall of both
    """
    transition_matrix = get_transition_matrix(
        lambda_2,
        lambda_1,
        mu,
        num_of_servers,
        threshold,
        system_capacity,
        buffer_capacity,
    )
    all_states = build_states(threshold, system_capacity, buffer_capacity)
    pi = get_steady_state_algebraically(
        transition_matrix, algebraic_function=np.linalg.solve
    )
    pi = get_markov_state_probabilities(pi, all_states, output=np.ndarray)
    if patient_type == "both":
        mean_waiting_time_class_1 = mean_waiting_time_formula(
            all_states=all_states,
            pi=pi,
            patient_type="others",
            lambda_2=lambda_2,
            lambda_1=lambda_1,
            mu=mu,
            num_of_servers=num_of_servers,
            threshold=threshold,
            system_capacity=system_capacity,
            buffer_capacity=buffer_capacity,
            formula=formula,
        )
        mean_waiting_time_class_2 = mean_waiting_time_formula(
            all_states=all_states,
            pi=pi,
            patient_type="ambulance",
            lambda_2=lambda_2,
            lambda_1=lambda_1,
            mu=mu,
            num_of_servers=num_of_servers,
            threshold=threshold,
            system_capacity=system_capacity,
            buffer_capacity=buffer_capacity,
            formula=formula,
        )

        prob_accept_class_1 = np.sum(
            [
                pi[state]
                for state in all_states
                if is_accepting_state(
                    state, "others", threshold, system_capacity, buffer_capacity
                )
            ]
        )
        prob_accept_class_2 = np.sum(
            [
                pi[state]
                for state in all_states
                if is_accepting_state(
                    state, "ambulance", threshold, system_capacity, buffer_capacity
                )
            ]
        )

        class_2_rate = (lambda_2 * prob_accept_class_2) / (
            (lambda_2 * prob_accept_class_2) + (lambda_1 * prob_accept_class_1)
        )
        class_1_rate = (lambda_1 * prob_accept_class_1) / (
            (lambda_2 * prob_accept_class_2) + (lambda_1 * prob_accept_class_1)
        )

        return (
            mean_waiting_time_class_2 * class_2_rate
            + mean_waiting_time_class_1 * class_1_rate
        )

    mean_waiting_time = mean_waiting_time_formula(
        all_states,
        pi,
        patient_type,
        lambda_2,
        lambda_1,
        mu,
        num_of_servers,
        threshold,
        system_capacity,
        buffer_capacity,
        formula=formula,
    )
    return mean_waiting_time
