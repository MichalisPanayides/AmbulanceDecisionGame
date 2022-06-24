"""Tests for code in dists.py"""
import ciw
import pytest

from hypothesis import given
from hypothesis.strategies import floats
from ambulance_game.simulation import dists


def test_class_state_dependent_exponential_value_error():
    """
    Test that the exponential distribution returns a value error when the
    rate is negative for the class StateDependentExponential.
    """
    rates = {(i, j): -0.05 for i in range(10) for j in range(10)}
    with pytest.raises(ValueError):
        dists.StateDependentExponential(rates)


def test_class_server_dependent_exponential_value_error():
    """
    Test that the exponential distribution returns a value error when the
    rate is negative for the class ServerDependentExponential.
    """
    rates = {k: -0.05 for k in range(10)}
    with pytest.raises(ValueError):
        dists.ServerDependentExponential(rates)


def test_class_state_server_dependent_exponential_value_error():
    """
    Test that the exponential distribution returns a value error when the
    rate is negative for the class StateServerDependentExponential.
    """
    rates = {
        k: {(i, j): -0.05 for i in range(10) for j in range(10)} for k in range(10)
    }
    with pytest.raises(ValueError):
        dists.StateServerDependentExponential(rates)


def test_is_state_dependent():
    """
    Tests that the is_state_dependent function returns True when the
    dictionary given is of the form {(i, j): mu}.
    """
    rates = {(i, j): i + j for i in range(10) for j in range(10)}
    assert dists.is_state_dependent(rates)

    rates = {i: 0.3 for i in range(10)}
    assert not dists.is_state_dependent(rates)


def test_is_server_dependent():
    """
    Tests that the is_state_dependent function returns True when the dictionary
    given is of the form {i: mu} and False when of the form {(i,j): mu}.
    """
    rates = {i: 0.3 for i in range(10)}
    assert dists.is_server_dependent(rates)

    rates = {(i, j): i + j for i in range(10) for j in range(10)}
    assert not dists.is_server_dependent(rates)


def test_is_state_server_dependent():
    """
    Tests that the is_state_dependent function returns True when a dictionary
    of dictionaries is given in the right format otherwise False.
    """
    rates = {}
    for server in range(3):
        rates[server] = {(u, v): 0.5 for u in range(2) for v in range(4)}

    assert dists.is_state_server_dependent(rates)

    rates[1][4] = 45
    assert not dists.is_state_server_dependent(rates)


@given(mu=floats(min_value=0.1, max_value=3))
def test_get_service_distribution(mu):
    """
    Tests that the get_service_distribution function returns the correct distribution
    """
    assert isinstance(dists.get_service_distribution(mu), ciw.dists.Exponential)

    rates = {(u, v): mu for u in range(10) for v in range(10)}
    assert isinstance(
        dists.get_service_distribution(rates), dists.StateDependentExponential
    )

    rates = {server: mu for server in range(5)}
    assert isinstance(
        dists.get_service_distribution(rates), dists.ServerDependentExponential
    )

    rates = {}
    for server in range(3):
        rates[server] = {(u, v): mu for u in range(2) for v in range(4)}
    assert isinstance(
        dists.get_service_distribution(rates), dists.StateServerDependentExponential
    )


def test_get_service_distribution_value_error():
    """
    Tests that the get_service_distribution function raises a value error when
    the rates given are not in the correct format.
    """
    mu = [1.2, 1.3, 1.4]
    with pytest.raises(ValueError):
        dists.get_service_distribution(mu)


@given(
    arrival_rate=floats(min_value=0, exclude_min=True),
)
def test_get_arrival_distribution_exponential(arrival_rate):
    """
    Test that an Exponential distribution object form the ciw library is
    returned given a positive arrival rate
    """
    assert isinstance(
        dists.get_arrival_distribution(arrival_rate), ciw.dists.Exponential
    )


def test_get_arrival_distribution_no_arrivals():
    """
    Test that a NoArrivals distribution object form the ciw library is
    returned given a zero arrival rate
    """
    assert isinstance(dists.get_arrival_distribution(0), ciw.dists.NoArrivals)


def test_get_arrival_distribution_value_error():
    """
    Test that a ValueError is raised when a negative number is given
    """
    with pytest.raises(ValueError):
        dists.get_arrival_distribution(-2)
