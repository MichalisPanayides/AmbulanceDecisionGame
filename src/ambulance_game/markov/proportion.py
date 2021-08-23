import math
import operator
from functools import reduce

import numpy as np
import sympy as sym

from .markov import (
    build_states,
    get_markov_state_probabilities,
    get_steady_state_algebraically,
    get_transition_matrix,
)
from .utils import get_proportion_of_individuals_not_lost, is_accepting_state


def product_of_all_elements(iterable):
    """
    Returns the product of all elements in iterable.
    """
    return reduce(operator.mul, iterable, 1)


def general_psi_function(arg, k, l, exp_rates, freq, a):
    """
    A general version of the Ψ function that is needed to get the cumulative
    distribution function of the hypoexponential distribution. This function
    can be used for any parameters and the outcome of it applies to any random
    variable that is hypoexponentially distributed.

    Parameters
    ----------
    arg : float
        The argument of the cdf
    k : int
        Variable that goes from 0 to the number of distinct parameters
    l : int
        Variable that goes from 0 to the frequencies of each parameter_k
    exp_rates : tuple
        Distinct exponential parameters
    freq : tuple
        Frequencies of each distinct parameter
    a : int
        The length of exp_rates and freq array

    Returns
    -------
    float
        The output of the Ψ function that is needed for the cdf of the
        Hypoexponential distribution.
    """
    t = sym.Symbol("t")
    product = product_of_all_elements(
        [(exp_rates[j] + t) ** (-freq[j]) for j in range(a + 1) if j != k]
    )
    psi_val = -sym.diff(product, t, l - 1)
    psi_val = psi_val.subs({t: arg})
    return float(psi_val)


def specific_psi_function(
    arg, k, l, exp_rates, freq, a
):  # pylint: disable=unused-argument
    """
    The specific version of the Ψ function that is used for the purpose of
    this study. This function is a simplification of the general_psi_function
    when the following are fixed:
        exp_rates = (0, C*mu, mu)
        freq = (1, ??, 1)
        a = 2
    Due to the way the hypoexponential cdf works the function is called only for
    values of k=1 and k=2. For these values the following hold:
                                    - k = 1 -> l = 1, ..., n
                                    - k = 2 -> l = 1

    Parameters
    ----------
    arg : float
        The argument of the cdf
    k : int
        Variable that goes from 0 to the number of distinct parameters
    l : int
        Variable that goes from 0 to the frequencies of each parameter_k
    exp_rates : tuple
        Distinct exponential parameters
    freq : tuple
        Frequencies of each distinct parameter
    a : int
        The length of exp_rates and freq array

    Returns
    -------
    float
        The output of the Ψ function that is needed for the cdf of the
        Hypoexponential distribution.
    """
    if k == 1:
        psi_val = (1 / (arg ** l)) - (1 / (arg + exp_rates[2]) ** l)
        psi_val *= (-1) ** l * math.factorial(l - 1) / exp_rates[2]
        return psi_val
    if k == 2:
        psi_val = -1 / (arg * (arg + exp_rates[1]) ** freq[1])
        return psi_val
    return 0


def hypoexponential_cdf(x, exp_rates, freq, psi_func=specific_psi_function):
    """
    The function represents the cumulative distribution function of the
    hypoexponential distribution. It calculates the probability that a
    hypoexponentially distributed random variable has a value less than x.

    In other words calculate P(S < x) where S ~ Hypo(λ, r)
        where: λ is a vector with distinct exponential parameters
           and r is a vector with the frequency of each distinct parameter

    Note that: a Hypoexponentially distributed random variable can be described
               as the sum of Erlang distributed random variables

    Parameters
    ----------
    x : float
        The target we want to calculate the probability for
    exp_rates : tuple
        The distinct exponential parameters
    freq : tuple
        The frequency of the exponential parameters
    psi_func : function, optional
        The function to be used to get Ψ, by default specific_psi_function

    Returns
    -------
    float
        P(S < x) where S ~ Hypo(λ, r)
    """
    a = len(exp_rates)
    exp_rates = (0,) + exp_rates
    freq = (1,) + freq

    summation = 0
    for k in range(1, a + 1):
        for l in range(1, (freq[k] + 1)):
            psi = psi_func(
                arg=-exp_rates[k],
                k=k,
                l=l,
                exp_rates=exp_rates,
                freq=freq,
                a=a,
            )
            iteration = psi * (x ** (freq[k] - l)) * np.exp(-exp_rates[k] * x)
            iteration /= np.math.factorial(freq[k] - l) * np.math.factorial(l - 1)
            summation += float(iteration)
    output = 1 - (
        product_of_all_elements([exp_rates[j] ** freq[j] for j in range(1, a + 1)])
        * summation
    )
    return output


def erlang_cdf(mu, n, x):
    """
    Cumulative distribution function of the erlang distribution.

    P(X < x) where X ~ Erlang(mu, n)

    Parameters
    ----------
    mu : float
        The parameter of the Erlang distribution
    n : int
        The number of Exponential distributions that are added together
    x : float
        The argument of the function

    Returns
    -------
    float
        The probability that the erlang distributed r.v. is less than x
    """
    return 1 - np.sum(
        [
            (np.math.exp(-mu * x) * (mu * x) ** i * (1 / np.math.factorial(i)))
            for i in range(n)
        ]
    )


def get_probability_of_waiting_time_in_system_less_than_target_for_state(
    state,
    class_type,
    mu,
    num_of_servers,
    threshold,
    target,
    psi_func=specific_psi_function,
):
    """
    The function decides what probability distribution to use based on the state
    we are currently on and the class type given.

    The two distributions that are used are the Erlang and the Hypoexponential
    distribution. The time it takes the system to exit a state and enter the
    next one is known to be exponentially distributed. The sum of exponentially
    distributed random variables is known to result in either an Erlang
    distribution or a Hypoexponential distribution (where the former is used
    when the exponentially distributed r.v. that we are summing have the same
    parameters and the latter when they have at least two distinct parameters).

    The function works as follows:
        - Checks whether the arriving individual will have to wait
        - Finds the total number of states an individual will have to visit
        - Depending on whether the parameters of the distributions to sum are
           the same or not, call the appropriate cdf function.


    Parameters
    ----------
    state : tuple
    class_type : int
    mu : float
    num_of_servers : int
    threshold : int
    target : int
    psi_func : function, optional

    Returns
    -------
    float
        The probability of spending less time than the target in the system when
        the individual has arrived at a given state
    """
    if class_type == 0:
        arrive_on_waiting_space = state[1] > num_of_servers
        rep = state[1] - num_of_servers
    elif class_type == 1:
        arrive_on_waiting_space = (
            state[1] > num_of_servers and threshold > num_of_servers
        )
        rep = min(state[1], threshold) - num_of_servers
    else:
        return ValueError("Class type bust be 0, 1 or None")

    if arrive_on_waiting_space:
        if num_of_servers == 1:
            prob = erlang_cdf(mu=mu, n=rep + 1, x=target)
        else:
            param = num_of_servers * mu
            prob = hypoexponential_cdf(
                x=target,
                exp_rates=(param, mu),
                freq=(rep, 1),
                psi_func=psi_func,
            )
    else:
        prob = erlang_cdf(mu=mu, n=1, x=target)

    return prob


def get_proportion_of_individuals_within_time_target(
    all_states,
    pi,
    class_type,
    mu,
    num_of_servers,
    threshold,
    system_capacity,
    buffer_capacity,
    target,
    psi_func=specific_psi_function,
    **kwargs,  # pylint: disable=unused-argument
):
    """
    Gets the probability that a certain class of individuals is within a given
    time target. This functions runs for every state the function
    get_probability_of_waiting_time_in_system_less_than_target_for_state() and
    by using the state probabilities to get the average proportion of individuals
    within target.

    Parameters
    ----------
    all_states : list
    pi : numpy.array
    class_type : int
    mu : float
    num_of_servers : int
    threshold : int
    system_capacity : int
    buffer_capacity : int
    target : float
    psi_func : function, optional

    Returns
    -------
    float
        The probability of spending less time than the target in the system
    """
    proportion_within_limit = 0
    probability_of_accepting = 0

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

            proportion_within_limit_at_state = (
                get_probability_of_waiting_time_in_system_less_than_target_for_state(
                    state=arriving_state,
                    class_type=class_type,
                    mu=mu,
                    num_of_servers=num_of_servers,
                    threshold=threshold,
                    target=target,
                    psi_func=psi_func,
                )
            )
            proportion_within_limit += pi[u, v] * proportion_within_limit_at_state
            probability_of_accepting += pi[u, v]
    return proportion_within_limit / probability_of_accepting


def overall_proportion_of_individuals_within_time_target(
    all_states,
    pi,
    lambda_1,
    lambda_2,
    mu,
    num_of_servers,
    threshold,
    system_capacity,
    buffer_capacity,
    target,
    psi_func=specific_psi_function,
    **kwargs,  # pylint: disable=unused-argument
):
    """
    The function gets the overall proportion of both classes of individuals by
    running get_proportion_of_individuals_within_time_target() for both classes.

    Parameters
    ----------
    all_states : tuple
    pi : numpy.array
    lambda_1 : float
    lambda_2 : float
    mu : float
    num_of_servers : int
    threshold : int
    system_capacity : int
    buffer_capacity : int
    target : float
    psi_func : function, optional

    Returns
    -------
    float
        The overall proportion of both classes of individuals
    """
    mean_prop_for_each_class = [
        get_proportion_of_individuals_within_time_target(
            all_states=all_states,
            pi=pi,
            class_type=class_type,
            mu=mu,
            num_of_servers=num_of_servers,
            threshold=threshold,
            system_capacity=system_capacity,
            buffer_capacity=buffer_capacity,
            target=target,
            psi_func=psi_func,
        )
        for class_type in range(2)
    ]

    class_rates = get_proportion_of_individuals_not_lost(
        all_states=all_states,
        pi=pi,
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        threshold=threshold,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
    )

    overall_prop_within_target = np.sum(
        [
            mean_prop_for_each_class[class_type] * class_rates[class_type]
            for class_type in range(2)
        ]
    )
    return overall_prop_within_target


def proportion_within_target_using_markov_state_probabilities(
    lambda_1,
    lambda_2,
    mu,
    num_of_servers,
    threshold,
    system_capacity,
    buffer_capacity,
    class_type,
    target,
    psi_func=specific_psi_function,
):
    """
    Get the proportion of individuals within target by using the state
    probabilities generated from the Markov model.

    Parameters
    ----------
    lambda_1 : float
    lambda_2 : float
    mu : float
    num_of_servers : int
    threshold : int
    system_capacity : int
    buffer_capacity : int
    class_type : int
    target : float
    psi_func : function, optional

    Returns
    -------
    float
        The proportion of individuals within target
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

    if class_type is None:
        proportion_formula = overall_proportion_of_individuals_within_time_target
    else:
        proportion_formula = get_proportion_of_individuals_within_time_target

    prop = proportion_formula(
        all_states=all_states,
        pi=pi,
        class_type=class_type,
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        mu=mu,
        num_of_servers=num_of_servers,
        threshold=threshold,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
        target=target,
        psi_func=psi_func,
    )
    return prop
