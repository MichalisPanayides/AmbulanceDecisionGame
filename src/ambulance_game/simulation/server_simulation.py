import random

import ciw

from .simulation import build_custom_node


class StateDependentExponential(ciw.dists.Distribution):
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


def build_state_dependent_model(
    lambda_2,
    lambda_1,
    rates,
    num_of_servers,
    system_capacity=float("inf"),
    buffer_capacity=float("inf"),
):
    """
    Builds a ciw object that represents a model of a queuing network with two
    service centres; the service area and the buffer space. Individuals arrive
    at the service area and at the buffer space with rates that follow the
    exponential distribution of λ_1 and λ_2 respectively. The service
    distribution follows a constant distribution of 0 for the buffer space and
    an exponential distribution that is dependent to the state of the system
    given by the `rates` dictionary. The variables "num_of_servers"and
    "buffer_capacity" indicate the capacities of the two centres. Finally, the
    queue capacity is set to the difference between the number of servers and
    the system capacity for the service area centre and for the buffer space it
    is set to zero, as there should not be any waiting there, just blockage.

    Parameters
    ----------
    lambda_2 : float
        Arrival rate of class 2 individuals
    lambda_1 : float
        Arrival rate of class 1 individuals
    rates : dict
        Rates that indivduals will be served
    num_of_servers : int
        The num_of_servers of the service area
    system_capacity : int, optional
        The capacity of the service area, by default float("inf")
    buffer_capacity : int, optional
        The capacity of the buffer space, by default float("inf")

    Returns
    -------
    object
        A ciw object that represents the model
    """
    model = ciw.create_network(
        arrival_distributions=[
            ciw.dists.Exponential(lambda_2),
            ciw.dists.Exponential(lambda_1),
        ],
        service_distributions=[
            ciw.dists.Deterministic(0),
            StateDependentExponential(rates=rates),
        ],
        routing=[[0.0, 1.0], [0.0, 0.0]],
        number_of_servers=[buffer_capacity, num_of_servers],
        queue_capacities=[0, system_capacity - num_of_servers],
    )
    return model


def simulate_state_dependent_model(
    lambda_2,
    lambda_1,
    rates,
    num_of_servers,
    threshold,
    seed_num=None,
    runtime=1440,
    system_capacity=float("inf"),
    buffer_capacity=float("inf"),
    num_of_trials=1,
    tracker=ciw.trackers.NodePopulation(),
):
    """Simulate the model by using the custom node and returning the simulation
    object.

    This is similar to `simulate_model()` in `simulation.py` but here we give
    the service rate as a dictionry (rates) with all possible rates at each
    state.

    Returns
    -------
    object
        An object that contains all simulation records
    """

    if buffer_capacity < 1:
        raise ValueError(
            "Simulation only implemented for buffer_capacity >= 1"
        )  # TODO Add an option to ciw model to all for no buffer capacity.

    if threshold > system_capacity:
        buffer_capacity = 1
        # TODO: Different approach to handle this situation

    if seed_num is None:
        seed_num = random.random()

    all_simulations = []
    for trial in range(num_of_trials):
        model = build_state_dependent_model(
            lambda_2=lambda_2,
            lambda_1=lambda_1,
            rates=rates,
            num_of_servers=num_of_servers,
            system_capacity=system_capacity,
            buffer_capacity=buffer_capacity,
        )
        node = build_custom_node(threshold)
        ciw.seed(seed_num + trial)
        simulation = ciw.Simulation(model, node_class=node, tracker=tracker)
        simulation.simulate_until_max_time(runtime)
        all_simulations.append(simulation)

    return all_simulations if len(all_simulations) > 1 else all_simulations[0]
