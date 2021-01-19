import numpy as np


def is_waiting_state(state, num_of_servers):
    """Checks if waiting occurs in the given state. In essence, all states (u,v)
    where v > C are considered waiting states.

    Set of waiting states: S_w = {(u,v) ∈ S | v > C}

    Parameters
    ----------
    state : tuple
        a tuples of the form (u,v)
    num_of_servers : int
        the number of servers = C
    Returns
    -------
    Boolean
        An indication of whether or not any wait occurs on the given state
    """
    return state[1] > num_of_servers


def is_blocking_state(state):
    """
    Checks if blocking occurs in the given state. In essence, all states (u,v)
    where u > 0 are considered blocking states

    Set of blocking states: S_b = {(u,v) ∈ S | u > 0}
    """
    return state[0] > 0


def is_accepting_state(state, class_type, threshold, system_capacity, buffer_capacity):
    """
    Checks if a state given is an accepting state. Accepting states are defined
    as the states of the system where arrivals may occur. In essence
    these states are all states apart from the one when the system cannot accept
    additional arrivals. Because there are two types of arrivals though, the set
    of accepting states is different for class 1 and class 2 individuals:

    Class 2 individuals: S_A = {(u,v) ∈ S | u < N}
    Class 1 individuals: S_A = {(u,v) ∈ S | v < M}

    Parameters
    ----------
    state : tuple
        a tuples of the form (u,v)
    class_type : int
        A string to distinguish between class 1 (=0) and class 2 individuals (=1)
    system_capacity : int
        The capacity of the system (hospital) = N
    buffer_capacity : int
        The capacity of the buffer space = M

    Returns
    -------
    Boolean
        An indication of whether or not an arrival of the given type
        (class_type) can occur
    """
    if class_type == 1:
        condition = (
            (state[0] < buffer_capacity)
            if (threshold <= system_capacity)
            else (state[1] < system_capacity)
        )
    if class_type == 0:
        condition = state[1] < system_capacity
    return condition


def expected_time_in_markov_state_ignoring_arrivals(
    state,
    class_type,
    num_of_servers,
    mu,
    threshold,
):
    """Get the expected waiting time in a Markov state when ignoring any subsequent
    arrivals. When considering the waiting time of class 2 individuals, and when
    these individuals are
    in a blocked state (v > 0) then by the definition of the problem the waiting
    time in that state is set to 0. Additionally, all states where u > 0 and v = T
    automatically get a waiting time of 0 because class 1 individuals only pass
    one of the states of that column (only state (0,T) is not zero).
    Otherwise the function's
    output is:
        - c(u,v) = 1/vμ   if v < C
        - c(u,v) = 1/Cμ   if v >= C

    Parameters
    ----------
    state : tuple
        a tuples of the form (u,v)
    class_type : int
        A string to distinguish between class 1(=0) and class 2(=1) individuals
    num_of_servers : int
        The number of servers = C
    mu : float
        The service rate = μ

    Returns
    -------
    float
        The expected waiting time in the given state
    """
    if state[0] > 0 and (state[1] == threshold or class_type == 1):
        return 0
    return 1 / (min(state[1], num_of_servers) * mu)


def expected_time_in_markov_state_ignoring_class_2_arrivals(
    state, lambda_1, mu, num_of_servers, system_capacity
):
    """
    The expected time of the Markov chain model at the state given.
    Note here that for a state (u,v) where v = system capacity (C) no class 1 arrival
    can occur and thus the rate at which the model leaves that state changes.
    """
    if state[1] == system_capacity:
        return 1 / (min(state[1], num_of_servers) * mu)
    return 1 / (min(state[1], num_of_servers) * mu + lambda_1)


def prob_service(state, lambda_1, mu, num_of_servers):
    """
    Gets the probability of finishing a service
    """
    return (min(state[1], num_of_servers) * mu) / (
        lambda_1 + (mu * min(state[1], num_of_servers))
    )


def prob_class_1_arrival(state, lambda_1, mu, num_of_servers):
    """Gets the probability of a class 1 arrival to occur"""
    return lambda_1 / (lambda_1 + (mu * min(state[1], num_of_servers)))


def get_probability_of_accepting(
    all_states,
    pi,
    threshold,
    system_capacity,
    buffer_capacity,
):
    """
    Generates the probability of acceptance for both class types of a given
    Markov model.

    Parameters
    ----------
    all_states : list
    pi : numpy.array
    threshold : int
    system_capacity : int
    buffer_capacity : int

    Returns
    -------
    list
        The probability of accepting an individual upon its arrival for class 0
        and class 1
    """
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
    return prob_accept


def get_proportion_of_individuals_not_lost(
    all_states,
    pi,
    lambda_1,
    lambda_2,
    threshold,
    system_capacity,
    buffer_capacity,
):
    """
    Generates the proportion of individuals that will normally travel through the
    system and will not be lost to the system.

    Parameters
    ----------
    all_states : lists
    pi : numpy.array
    lambda_1 : float
    lambda_2 : float
    threshold : int
    system_capacity : int
    buffer_capacity : int

    Returns
    -------
    list
        The proportion of not lost individuals of both class 0 and class 1
        individuals
    """
    prob_accept = get_probability_of_accepting(
        all_states,
        pi,
        threshold,
        system_capacity,
        buffer_capacity,
    )
    class_rates = [
        prob_accept[class_type]
        / ((lambda_2 * prob_accept[1]) + (lambda_1 * prob_accept[0]))
        for class_type in range(2)
    ]
    class_rates[0] *= lambda_1
    class_rates[1] *= lambda_2

    return class_rates
