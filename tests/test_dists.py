"""Tests for code in dists.py"""
import ciw
import pytest

from hypothesis import given
from hypothesis.strategies import floats
from ambulance_game.simulation import dists


def test_class_state_dependent_exponential_value_error():
    """
    Test that the exponential distribution returns a value error when the
    rate is negative.
    """
    rates = {(i, j): -0.05 for i in range(10) for j in range(10)}
    with pytest.raises(ValueError):
        dists.StateDependentExponential(rates)


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
    Tests that the is_state_dependent function returns a value error when
    the dictionary given is of the form {i: mu}.
    """
    rates = {}
    for server in range(3):
        rates[server] = {(u, v): 0.5 for u in range(2) for v in range(4)}

    assert dists.is_state_server_dependent(rates)


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
