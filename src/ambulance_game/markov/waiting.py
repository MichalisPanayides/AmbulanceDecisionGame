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
    """Performs a recursive algorithm to get the expected waiting time of patients
    when they enter the model at a given state. Given an arriving state the algorithm
    moves down to all subsequent states until it reaches one that is not a waiting
    state.

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

    Note:   For all "others" patients the recursive formula acts in a linear manner
    meaning that an individual will have the same waiting time when arriving at
    either state of the same column e.g (2, 3) or (5, 3).

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
    """Get the mean waiting time by using the recursive formula or a closed-form
    formula

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
    patient_type="both",
    formula="closed_form",
):
    """Gets the mean waiting time for a Markov chain model

    Parameters
    ----------
    lambda_a : float
    lambda_o : float
    mu : float
    num_of_servers : int
    threshold : int
    system_capacity : int
    parking_capacity : int
    patient_type : str, optional
    formula : str, optional

    Returns
    -------
    float
        The mean waiting time of the in the system of either ambulance patients,
        other patients or the overall of both
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
    all_states = build_states(threshold, system_capacity, parking_capacity)
    pi = get_steady_state_algebraically(
        transition_matrix, algebraic_function=np.linalg.solve
    )
    pi = get_markov_state_probabilities(pi, all_states, output=np.ndarray)
    if patient_type == "both":
        mean_waiting_time_other = mean_waiting_time_formula(
            all_states,
            pi,
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
            pi,
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
                pi[state]
                for state in all_states
                if is_accepting_state(
                    state, "others", threshold, system_capacity, parking_capacity
                )
            ]
        )
        prob_accept_ambulance = np.sum(
            [
                pi[state]
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
        pi,
        patient_type,
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
