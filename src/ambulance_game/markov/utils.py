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
        A string to distinguish between ambulance and other patients
    system_capacity : int
        The capacity of the system (hospital) = N
    parking_capacity : int
        The capacity of the parking space = M

    Returns
    -------
    Boolean
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
    """Get the expected waiting time in a Markov state when ignoring any subsequent
    arrivals.When considering ambulance patients waiting time and the patients are
    in a blocked state (v > 0) then by the definition of the problem the waiting
    time in that state is set to 0. Additionally, all states where u > 0 and v = T
    automatically get a waiting time of 0 because other patients only pass one of
    the states of that column (only state (0,T) is not zero). Otherwise the function's
    output is:
        - c(u,v) = 1/vμ   if v < C
        - c(u,v) = 1/Cμ   if v >= C

    Parameters
    ----------
    state : tuple
        a tuples of the form (u,v)
    patient_type : string
        A string to distinguish between ambulance and other patients
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


# TODO Modify name to fit generic formulation
def expected_time_in_markov_state_ignoring_ambulance_arrivals(
    state, lambda_o, mu, num_of_servers, system_capacity
):
    """
    The expected time of the Markov chain model at the state given.
    Note here that for a state (u,v) where v = system capacity (C) no other arrivals
    can occur and thus the rate at which the model leaves that state changes.
    """
    if state[1] == system_capacity:
        return 1 / (min(state[1], num_of_servers) * mu)
    return 1 / (min(state[1], num_of_servers) * mu + lambda_o)


def prob_service(state, lambda_o, mu, num_of_servers):
    """
    Gets the probability of finishing a service
    """
    return (min(state[1], num_of_servers) * mu) / (
        lambda_o + (mu * min(state[1], num_of_servers))
    )


def prob_other_arrival(state, lambda_o, mu, num_of_servers):
    """Gets the probability of an "other" patient arriving"""
    return lambda_o / (lambda_o + (mu * min(state[1], num_of_servers)))
