"""
Code for custom distribution classes and other distribution related functions
"""

import random

import ciw


class StateDependentExponential(
    ciw.dists.Distribution
):  # pylint: disable=too-few-public-methods
    """
    A class that inherits from the `Distribution` class in the ciw module. This
    class is meant to be used in the simulation module as a state dependent
    distribution for the service of individuals.

    This distribution takes `rates` as an argument; a disctionary with keys
    states `(u,v)` and values the service rate at that state.
    """

    def __init__(self, rates):
        if any(rate <= 0 for rate in rates.values()):
            raise ValueError(
                "Exponential distribution must sample positive numbers only."
            )
        self.rates = rates

    def sample(self, t=None, ind=None):
        """
        This method is used to sample the service time for an individual based
        on the current state
        """
        state = ind.simulation.statetracker.state
        rate = self.rates[tuple(state)]
        return random.expovariate(rate)


def is_mu_state_dependent(mu):
    """
    Check if mu is a dictionary with keys that are states and values that are
    service rates.
    """
    for key in mu.keys():
        if len(key) != 2 or not isinstance(key[0], int) or not isinstance(key[1], int):
            return False
    return True


def is_mu_server_dependent():
    """
    Checks if mu is a dictionary with keys that are servers and values that are
    service rates.
    """
    return False


def is_mu_state_server_dependent():
    """
    Checks if mu is a dictionary of distionaries. The keys are servers id and
    the values are another dictionary with keys the states and values the
    service rates.
    """
    return False


def get_service_distribution(mu):
    """
    Get the service distribution out of:
        - ciw.dists.Exponential
        - StateDependentExponential
        - ServerDependentExponential
        - StateServerDependentExponential
    """
    if isinstance(mu, (float, int)):
        return ciw.dists.Exponential(mu)
    if isinstance(mu, dict):
        if is_mu_state_dependent(mu):
            return StateDependentExponential(mu)
        elif is_mu_server_dependent():
            raise NotImplementedError(
                "Server dependent service rates are not yet implemented."
            )
        elif is_mu_state_server_dependent():
            raise NotImplementedError(
                "State and server dependent distributions are not implemented yet."
            )
    raise ValueError("mu must be either an integer or a dictionary")
