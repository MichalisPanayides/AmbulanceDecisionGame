"""
Code for the simulation of the model.
"""

import collections
import copy
import itertools
import random

import ciw
import numpy as np

from .dists import get_service_distribution


def build_model(
    lambda_2,
    lambda_1,
    mu,
    num_of_servers,
    system_capacity=float("inf"),
    buffer_capacity=float("inf"),
):
    """Builds a ciw object that represents a model of a queuing network with two
    service centres; the service area and the buffer space. individuals arrive at the
    service area and at the buffer space with rates that follow the exponential
    distribution of λ_1 and λ_2 respectively. The service distribution follows
    a constant distribution of 0 for the buffer space and an exponential
    distribution with a rate of μ (mu) for the service area. The variables
    "num_of_servers" and "buffer_capacity" indicate the capacities of the two centres.
    Finally, the queue capacity is set to the difference between the number of servers
    and the system capacity for the service area centre and for the buffer space it is
    set to zero, as there should not occur any waiting there, just blockage.

    This function has been updated such that the type of `mu` defines the service
    rate distribution:
        - `int`, `float`: Exponential distribution with rate `mu`
        - `dict` {(u,v): mu_(u,v)}: State dependent exponential distribution
        - `dict` {k: mu_k}: Server dependent exponential distribution
        - `dict` {k: {(u,v): mu_(uv)^k}}: State and server dependent exponential

    Parameters
    ----------
    lambda_2 : float
        Arrival rate of class 2 individuals
    lambda_1 : float
        Arrival rate of class 1 individuals
    mu : float, dict
        Service rate of service area
    num_of_servers : integer
        The num_of_servers of the service area
    """
    service_dist = get_service_distribution(mu)
    model = ciw.create_network(
        arrival_distributions=[
            ciw.dists.Exponential(lambda_2),
            ciw.dists.Exponential(lambda_1),
        ],
        service_distributions=[ciw.dists.Deterministic(0), service_dist],
        routing=[[0.0, 1.0], [0.0, 0.0]],
        number_of_servers=[buffer_capacity, num_of_servers],
        queue_capacities=[0, system_capacity - num_of_servers],
    )
    return model


def build_custom_node(threshold=float("inf")):
    """Build a custom node to replace the default ciw.Node. Inherits from the original
    ciw.Node class and replaces methods release_blocked_individual and finish_service.
    The methods are modified in such a way such that all individuals that are in
    the buffer space node (node 1) remain blocked as long as the number of individuals
    in the service area node (node 2) exceeds the threshold.

    Parameters
    ----------
    threshold : int, optional
        The capacity threshold to be used by the method

    Returns
    -------
    class
        A custom node class that inherits from ciw.Node
    """

    class CustomNode(ciw.Node):
        """
        Overrides the default release_blocked_individual and finish_service
        methods of the ciw.Node class
        """

        def release_blocked_individual(self):
            """
            Releases an individual who becomes unblocked when
            another individual is released:
              - check if individual in node 2 and should remain blocked
                i.e. if the number of individuals in that node > threshold
              - check if anyone is blocked by this node
              - find the individual who has been blocked the longest
              - remove that individual from blocked queue
              - check if that individual had their service interrupted
              - release that individual from their node
            """
            continue_blockage = (
                self.number_of_individuals >= threshold and self.id_number == 2
            )
            if (
                self.len_blocked_queue > 0
                and self.number_of_individuals < self.node_capacity
                and not continue_blockage
            ):
                node_to_receive_from = self.simulation.nodes[self.blocked_queue[0][0]]
                individual_to_receive_index = [
                    ind.id_number for ind in node_to_receive_from.all_individuals
                ].index(self.blocked_queue[0][1])
                individual_to_receive = node_to_receive_from.all_individuals[
                    individual_to_receive_index
                ]
                self.blocked_queue.pop(0)
                self.len_blocked_queue -= 1
                if individual_to_receive.interrupted:  # pragma: no cover
                    individual_to_receive.interrupted = False
                    node_to_receive_from.interrupted_individuals.remove(
                        individual_to_receive
                    )
                    node_to_receive_from.number_interrupted_individuals -= 1
                node_to_receive_from.release(individual_to_receive_index, self)

        def finish_service(self):
            """
            The next individual finishes service:
              - finds the individual to finish service
              - check if they need to change class
              - find their next node
              - release the individual if there is capacity at destination,
                otherwise cause blockage
              - note that blockage also occurs when we are at node 1 and the
                number of individuals on node 2 are more than the 'threshold'
            """
            next_individual, next_individual_index = self.find_next_individual()
            self.change_customer_class(next_individual)
            next_node = self.next_node(next_individual)
            next_individual.destination = next_node.id_number
            if not np.isinf(self.c):
                next_individual.server.next_end_service_date = float("Inf")
            blockage = (
                next_node.number_of_individuals >= threshold and self.id_number == 1
            )
            if (
                next_node.number_of_individuals < next_node.node_capacity
            ) and not blockage:
                self.release(next_individual_index, next_node)
            else:
                self.block_individual(next_individual, next_node)

    return CustomNode


def simulate_model(
    lambda_2,
    lambda_1,
    mu,
    num_of_servers,
    threshold,
    seed_num=None,
    runtime=1440,
    system_capacity=float("inf"),
    buffer_capacity=float("inf"),
    num_of_trials=1,
    tracker=ciw.trackers.NodePopulation(),
):
    """Simulate the model by using the custom node and returning the simulation object.

    It is important to note that when the threshold is greater than the system capacity
    the buffer capacity is forced to be 1 because otherwise, when the service area
    gets full, class 2 individuals will flood the buffer area which is not what
    should happen in this particular scenario.

    Additionally, the buffer capacity should always be greater or equal to 1

    Parameters
    ----------
    seed_num : [float], optional
        A seed number in order to be able to replicate results,
        by default random.random()

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
        model = build_model(
            lambda_2=lambda_2,
            lambda_1=lambda_1,
            mu=mu,
            num_of_servers=num_of_servers,
            system_capacity=system_capacity,
            buffer_capacity=buffer_capacity,
        )
        node = build_custom_node(threshold)
        ciw.seed(seed_num + trial)
        simulation = ciw.Simulation(model, node_class=node, tracker=tracker)
        simulation.simulate_until_max_time(runtime)
        all_simulations.append(copy.deepcopy(simulation))

    return all_simulations if len(all_simulations) > 1 else all_simulations[0]


def get_simulated_state_probabilities(
    simulation_object, output=np.ndarray, system_capacity=None, buffer_capacity=None
):
    """Calculates the vector pi in a dictionary format or an array format. For the
    dictionary format the keys are the states (i,j) and the values are the probabilities
    that the system is in a current state. For the 2-dimensional array format the
    probability of being in state (i,j) is placed in the equivalent (i,j) position
    in the numpy array.

    Parameters
    ----------
    simulation_object : object
        The simulation object from ciw to extract pi from
    type : type, optional
        The type of format to be returned, by default np.ndarray

    Returns
    -------
    dictionary OR np.ndarray
        - A dictionary with the Markov states as keys and the equivalent probabilities
          as values
        - A numpy.ndarray Π where: Π(i,j) = probability of being in state (i,j)
    """
    state_probabilities_dictionary = (
        simulation_object.statetracker.state_probabilities()
    )
    if output == dict:
        return state_probabilities_dictionary

    if buffer_capacity is None:
        buffer_capacity = max(
            [state[0] for state in state_probabilities_dictionary.keys()]
        )
    if system_capacity is None:
        system_capacity = max(
            [state[1] for state in state_probabilities_dictionary.keys()]
        )
    state_probabilities_array = np.full(
        (buffer_capacity + 1, system_capacity + 1), np.NaN
    )
    for key, value in state_probabilities_dictionary.items():
        if value > 0:
            state_probabilities_array[key] = value
    return state_probabilities_array


def get_average_simulated_state_probabilities(
    lambda_2,
    lambda_1,
    mu,
    num_of_servers,
    threshold,
    system_capacity,
    buffer_capacity,
    seed_num=None,
    runtime=1440,
    num_of_trials=10,
    output=np.ndarray,
):
    """
    This function runs the get_simulated_state_probabilities() for multiple iterations
    to eliminate any stochasticity from the results

    Parameters
    ----------
    output : type, optional
        The format of the output state probabilities, by default np.ndarray
    """
    if seed_num is None:
        seed_num = random.random()

    if output == dict:
        average_state_probabilities = {}
        for trial in range(num_of_trials):
            simulation_object = simulate_model(
                lambda_2=lambda_2,
                lambda_1=lambda_1,
                mu=mu,
                num_of_servers=num_of_servers,
                threshold=threshold,
                seed_num=seed_num + trial,
                runtime=runtime,
                system_capacity=system_capacity,
                buffer_capacity=buffer_capacity,
            )
            state_probabilities = get_simulated_state_probabilities(
                simulation_object=simulation_object,
                output=output,
                system_capacity=system_capacity,
                buffer_capacity=buffer_capacity,
            )
            if len(average_state_probabilities) == 0:
                average_state_probabilities = state_probabilities
            else:
                for key in state_probabilities:
                    if key in average_state_probabilities:
                        average_state_probabilities[key] += state_probabilities[key]
                    else:
                        average_state_probabilities[key] = state_probabilities[key]
        for key, value in average_state_probabilities.items():
            average_state_probabilities[key] = value / num_of_trials
    else:
        all_simulations = []
        for trial in range(num_of_trials):
            simulation_object = simulate_model(
                lambda_2=lambda_2,
                lambda_1=lambda_1,
                mu=mu,
                num_of_servers=num_of_servers,
                threshold=threshold,
                seed_num=seed_num + trial,
                runtime=runtime,
                system_capacity=system_capacity,
                buffer_capacity=buffer_capacity,
            )
            all_simulations.append(copy.deepcopy(simulation_object))

        average_state_probabilities = (
            get_average_simulated_state_probabilities_from_simulations(
                simulations=all_simulations,
                system_capacity=system_capacity,
                buffer_capacity=buffer_capacity,
            )
        )

    return average_state_probabilities


def get_average_simulated_state_probabilities_from_simulations(
    simulations,
    system_capacity,
    buffer_capacity,
    output=np.ndarray,
):
    """
    This function runs the get_simulated_state_probabilities() for multiple iterations
    to eliminate any stochasticity from the results

    Parameters
    ----------
    output : type, optional
        The format of the output state probabilities, by default np.ndarray
    """
    # TODO: Implement output = dict
    average_state_probabilities = np.full(
        (buffer_capacity + 1, system_capacity + 1), np.NaN
    )
    for simulation in simulations:
        state_probabilities = get_simulated_state_probabilities(
            simulation_object=simulation,
            output=output,
            system_capacity=system_capacity,
            buffer_capacity=buffer_capacity,
        )
        for row, col in itertools.product(
            range(buffer_capacity + 1), range(system_capacity + 1)
        ):
            updated_entry = np.nansum(
                [
                    average_state_probabilities[row, col],
                    state_probabilities[row, col],
                ]
            )
            average_state_probabilities[row, col] = (
                updated_entry if updated_entry != 0 else np.NaN
            )
    average_state_probabilities /= len(simulations)

    return average_state_probabilities


def extract_times_from_records(simulation_records, warm_up_time):
    """Get the required times (waiting, service, blocking) out of ciw's records
    where all individuals are treated the same way. This function can't distinguish
    between class 1 and class 2 individuals. It returns the aggregated waiting
    time, service times BUT only blocking times of class 2 individuals.

    Parameters
    ----------
    simulation_records : list
        A list of all simulated records
    warm_up_time : int
        The time we start collecting results

    Returns
    -------
    list, list, list
        Three lists that contain waiting, service and blocking times
    """
    waiting_times = [
        r.waiting_time
        for r in simulation_records
        if r.arrival_date > warm_up_time and r.node == 2
    ]
    serving_times = [
        r.service_time
        for r in simulation_records
        if r.arrival_date > warm_up_time and r.node == 2
    ]
    blocking_times = [
        r.time_blocked
        for r in simulation_records
        if r.arrival_date > warm_up_time and r.node == 1
    ]
    return waiting_times, serving_times, blocking_times


def extract_times_from_individuals(individuals, warm_up_time, class_type):
    """
    Extract waiting times and service times for all individuals and proceed to extract
    blocking times for just class 2 individuals. The function uses individual's records
    and determines the type of class that each individual is, based on the number
    of nodes visited.
    """
    waiting_times = []
    serving_times = []
    blocking_times = []

    first_node_to_visit = 2 - class_type
    total_node_visits = class_type + 1

    for ind in individuals:
        if (
            ind.data_records[0].node == first_node_to_visit
            and len(ind.data_records) == total_node_visits
            and ind.data_records[total_node_visits - 1].arrival_date > warm_up_time
        ):
            waiting_times.append(ind.data_records[total_node_visits - 1].waiting_time)
            serving_times.append(ind.data_records[total_node_visits - 1].service_time)
        if (
            first_node_to_visit == ind.data_records[0].node == 1
            and ind.data_records[0].arrival_date > warm_up_time
        ):
            blocking_times.append(ind.data_records[0].time_blocked)

    return waiting_times, serving_times, blocking_times


def get_list_of_results(results):
    """Modify the outputs so that they are returned in a list format where it is
    sometimes easier to be used by other functions.

    Parameters
    ----------
    results : list
        A list of named tuples for each iteration

    Returns
    -------
    list, list, list
        Three lists that include all waits, services and blocks of all runs of
        all individuals
    """
    all_waits = [w.waiting_times for w in results]
    all_services = [s.service_times for s in results]
    all_blocks = [b.blocking_times for b in results]
    all_props = [p.proportion_within_target for p in results]
    return all_waits, all_services, all_blocks, all_props


def extract_total_individuals_and_the_ones_within_target_for_both_classes(
    individuals, target
):
    """
    Extract the total number of individuals that pass through the model and the
    total number of individuals that exit the model within the given target.

    Parameters
    ----------
    individuals : list of ciw.individual.Individual objects
    target : float

    Returns
    -------
    int, int, int, int
        - The number of class 2 individuals that pass through the model
        - The number of class 2 individuals that pass through within the target
        - The number of class 1 individuals that pass through the model
        - The number of class 1 individuals that pass through within the target
    """
    class_2_inds, class_2_inds_within_target = 0, 0
    class_1_inds, class_1_inds_within_target = 0, 0
    for individual in individuals:
        ind_class = len(individual.data_records) - 1
        rec = individual.data_records[-1]
        if rec.node == 2 and ind_class == 0:
            class_1_inds += 1
            if rec.waiting_time + rec.service_time < target:
                class_1_inds_within_target += 1
        elif rec.node == 2 and ind_class == 1:
            class_2_inds += 1
            if rec.waiting_time + rec.service_time < target:
                class_2_inds_within_target += 1

    return (
        class_2_inds,
        class_2_inds_within_target,
        class_1_inds,
        class_1_inds_within_target,
    )


def get_mean_proportion_of_individuals_within_target_for_multiple_runs(
    lambda_2,
    lambda_1,
    mu,
    num_of_servers,
    threshold,
    system_capacity,
    buffer_capacity,
    seed_num,
    num_of_trials,
    runtime,
    target,
):
    """
    Get the average proportion of individuals within target by running the
    simulation multiple times

    Parameters
    ----------
    lambda_2 : float
    lambda_1 : float
    mu : float
    num_of_servers : int
    threshold : int
    system_capacity : int
    buffer_capacity : int
    seed_num : float
    num_of_trials : int
    runtime : int
    target : int

    Returns
    -------
    float, float, float
        - The combined mean proportion of individuals within target
        - The mean proportion of class 1 individuals within target
        - The mean proportion of class 2 individuals within target
    """
    class_2_proportions = []
    class_1_proportions = []
    combined_proportions = []

    if seed_num is None:
        seed_num = random.random()

    for trial in range(num_of_trials):
        individuals = simulate_model(
            lambda_2=lambda_2,
            lambda_1=lambda_1,
            mu=mu,
            num_of_servers=num_of_servers,
            threshold=threshold,
            system_capacity=system_capacity,
            buffer_capacity=buffer_capacity,
            seed_num=seed_num + trial,
            runtime=runtime,
        ).get_all_individuals()
        (
            class_2_inds,
            class_2_inds_within_target,
            class_1_inds,
            class_1_inds_within_target,
        ) = extract_total_individuals_and_the_ones_within_target_for_both_classes(
            individuals, target
        )

        class_2_proportions.append(
            (class_2_inds_within_target / class_2_inds) if class_2_inds != 0 else 1
        )
        class_1_proportions.append(
            (class_1_inds_within_target / class_1_inds) if class_1_inds != 0 else 1
        )
        combined_proportions.append(
            (class_2_inds_within_target + class_1_inds_within_target)
            / (class_2_inds + class_1_inds)
            if (class_2_inds + class_1_inds) != 0
            else 1
        )

    return combined_proportions, class_1_proportions, class_2_proportions


def get_multiple_runs_results(
    lambda_2,
    lambda_1,
    mu,
    num_of_servers,
    threshold,
    target=1,
    seed_num=None,
    warm_up_time=100,
    num_of_trials=10,
    runtime=1440,
    output_type="tuple",
    system_capacity=float("inf"),
    buffer_capacity=float("inf"),
    class_type=None,
):
    """Get the waiting times, service times and blocking times for multiple runs
    of the simulation. The function may return the times for class 2 individuals,
    class 1 individuals or the aggregated total of the two.

    Parameters
    ----------
    warm_up_time : int, optional
        Time to start collecting results, by default 100
    num_of_trials : int, optional
        Number of trials to run the model, by default 10
    output_type : str, optional
        The results' output type (either tuple or list)], by default "tuple"
    class_type : int, optional
        An integer to identify what type of class to get the times for, where
        class_type=(0,1,None) to denote class 1, class 2 or both

    Returns
    -------
    list
        A list of records where each record consists of the waiting, service and
        blocking times of one trial. Alternatively if the output_type = "list" then
        returns three lists with all waiting, service and blocking times
    """
    if seed_num is None:  # pragma: no cover
        seed_num = random.random()

    all_simulations = []
    for trial in range(num_of_trials):
        simulation = simulate_model(
            lambda_2=lambda_2,
            lambda_1=lambda_1,
            mu=mu,
            num_of_servers=num_of_servers,
            threshold=threshold,
            seed_num=seed_num + trial,
            runtime=runtime,
            system_capacity=system_capacity,
            buffer_capacity=buffer_capacity,
        )
        all_simulations.append(copy.deepcopy(simulation))

    results = get_multiple_runs_results_from_simulations(
        simulations=all_simulations,
        target=target,
        warm_up_time=warm_up_time,
        class_type=class_type,
    )

    if output_type == "list":
        all_waits, all_services, all_blocks, all_props = get_list_of_results(results)
        return [all_waits, all_services, all_blocks, all_props]
    return results


def get_multiple_runs_results_from_simulations(
    simulations, target, class_type=None, warm_up_time=100
):
    """Get the waiting times, service times and blocking times for the given
    simulations. The function may return the times for class 2 individuals,
    class 1 individuals or the aggregated total of the two.

    Parameters
    ----------
    simulations : list
        A list of all simulation objects
    target : float
        The target value to get the proportion of individuals within
    warm_up_time : int, optional
        Time to start collecting results, by default 100
    class_type : int, optional
        An integer to identify what type of class to get the times for, where
        class_type=(0,1,None) to denote class 1, class 2 or both

    Returns
    -------
    list
        A list of records where each record consists of the waiting, service and
        blocking times of one trial
    """
    records = collections.namedtuple(
        "records", "waiting_times service_times blocking_times proportion_within_target"
    )
    results = []
    if isinstance(simulations, ciw.Simulation):
        simulations = [simulations]

    for simulation in simulations:
        if class_type is None:
            sim_results = simulation.get_all_records()
            waiting_times, serving_times, blocking_times = extract_times_from_records(
                sim_results, warm_up_time
            )
        individuals = simulation.get_all_individuals()
        if class_type in (0, 1):
            (
                waiting_times,
                serving_times,
                blocking_times,
            ) = extract_times_from_individuals(
                individuals=individuals,
                warm_up_time=warm_up_time,
                class_type=class_type,
            )
        (
            class_2_inds,
            class_2_inds_within_target,
            class_1_inds,
            class_1_inds_within_target,
        ) = extract_total_individuals_and_the_ones_within_target_for_both_classes(
            individuals=individuals, target=target
        )
        if class_type is None:
            proportion_within_target = (
                (class_1_inds_within_target + class_2_inds_within_target)
                / (class_1_inds + class_2_inds)
                if class_1_inds + class_2_inds != 0
                else np.nan
            )
        elif class_type == 0:
            proportion_within_target = (
                class_1_inds_within_target / class_1_inds
                if class_1_inds != 0
                else np.nan
            )
        elif class_type == 1:
            proportion_within_target = (
                class_2_inds_within_target / class_2_inds
                if class_2_inds != 0
                else np.nan
            )
        results.append(
            records(
                waiting_times, serving_times, blocking_times, proportion_within_target
            )
        )
    return results


def get_decision_values(
    simulations_1,
    simulations_2,
    threshold_1,
    threshold_2,
    system_capacity_1,
    system_capacity_2,
    buffer_capacity_1,
    buffer_capacity_2,
    warm_up_time=100,
    alpha=0,
):
    """
    Gets the weighted decision values that will be used to calculate the split
    between individuals. The decision values are calculated as the weighted
    average of the blocking times and the proportion of individuals lost to
    the system.

    Parameters
    ----------
    simulations_1 : list
        A list of all simulation objects for the first system
    simulations_2 : list
        A list of all simulation objects for the second system
    warm_up_time : float, optional
    alpha : int, optional
        The weight to give to the blocking times and to lost individuals, by
        default 0

    Returns
    -------
    float, float
        The two decision values for the first system and the second system
    """
    all_blocking_times_1 = []
    all_blocking_times_2 = []
    for sim_1, sim_2 in zip(simulations_1, simulations_2):
        blocking_times_1 = [
            r.time_blocked
            for r in sim_1.get_all_records()
            if r.arrival_date > warm_up_time and r.node == 1
        ]
        all_blocking_times_1.append(blocking_times_1)
        blocking_times_2 = [
            r.time_blocked
            for r in sim_2.get_all_records()
            if r.arrival_date > warm_up_time and r.node == 1
        ]
        all_blocking_times_2.append(blocking_times_2)
    mean_blocking_time_1 = np.nanmean([np.nanmean(b) for b in all_blocking_times_1])
    mean_blocking_time_2 = np.nanmean([np.nanmean(b) for b in all_blocking_times_2])

    if np.isinf(system_capacity_1):
        prob_lost_1 = 0
    else:
        state_probs_1 = get_average_simulated_state_probabilities_from_simulations(
            simulations_1,
            system_capacity=system_capacity_1,
            buffer_capacity=buffer_capacity_1,
        )
        prob_lost_1 = (
            np.nansum(state_probs_1[-1])
            if threshold_1 <= threshold_2
            else np.nansum(state_probs_1[:, -1])
        )

    if np.isinf(system_capacity_2):
        prob_lost_2 = 0
    else:
        state_probs_2 = get_average_simulated_state_probabilities_from_simulations(
            simulations_2,
            system_capacity=system_capacity_2,
            buffer_capacity=buffer_capacity_2,
        )

        prob_lost_2 = (
            np.nansum(state_probs_2[-1])
            if threshold_1 <= threshold_2
            else np.nansum(state_probs_2[:, -1])
        )

    decision_value_1 = alpha * (prob_lost_1) + (1 - alpha) * np.mean(
        mean_blocking_time_1
    )
    decision_value_2 = alpha * (prob_lost_2) + (1 - alpha) * np.mean(
        mean_blocking_time_2
    )

    return decision_value_1, decision_value_2


def get_mean_blocking_difference_using_simulation(
    prop_1,
    lambda_2,
    lambda_1_1,
    lambda_1_2,
    mu_1,
    mu_2,
    num_of_servers_1,
    num_of_servers_2,
    threshold_1,
    threshold_2,
    system_capacity_1,
    system_capacity_2,
    buffer_capacity_1,
    buffer_capacity_2,
    alpha=0,
    seed_num_1=None,
    seed_num_2=None,
    num_of_trials=10,
    warm_up_time=100,
    runtime=1440,
):
    """Given a predefined proportion of class's 2 arrival rate calculate the
    mean difference between blocking times of two systems with given set of parameters.
    Note that all parameters that end in "_1" correspond to the first
    system while "_2" to the second.

    Parameters
    ----------
    prop_1 : float
        Proportion of class's 2 arrival rate that will be distributed to system 1
    lambda_2 : float
        Total class 2 arrival rate

    Returns
    -------
    float
        The difference between the mean blocking time of the two systems
    """
    lambda_2_1 = prop_1 * lambda_2
    lambda_2_2 = (1 - prop_1) * lambda_2

    simulations_1 = simulate_model(
        lambda_2=lambda_2_1,
        lambda_1=lambda_1_1,
        mu=mu_1,
        num_of_servers=num_of_servers_1,
        threshold=threshold_1,
        system_capacity=system_capacity_1,
        buffer_capacity=buffer_capacity_1,
        seed_num=seed_num_1,
        num_of_trials=num_of_trials,
        runtime=runtime,
    )

    simulations_2 = simulate_model(
        lambda_2=lambda_2_2,
        lambda_1=lambda_1_2,
        mu=mu_2,
        num_of_servers=num_of_servers_2,
        threshold=threshold_2,
        system_capacity=system_capacity_2,
        buffer_capacity=buffer_capacity_2,
        seed_num=seed_num_2,
        num_of_trials=num_of_trials,
        runtime=runtime,
    )

    simulations_1 = (
        [simulations_1] if isinstance(simulations_1, ciw.Simulation) else simulations_1
    )
    simulations_2 = (
        [simulations_2] if isinstance(simulations_2, ciw.Simulation) else simulations_2
    )

    decision_value_1, decision_value_2 = get_decision_values(
        simulations_1=simulations_1,
        simulations_2=simulations_2,
        threshold_1=threshold_1,
        threshold_2=threshold_2,
        system_capacity_1=system_capacity_1,
        system_capacity_2=system_capacity_2,
        buffer_capacity_1=buffer_capacity_1,
        buffer_capacity_2=buffer_capacity_2,
        warm_up_time=warm_up_time,
        alpha=alpha,
    )

    return decision_value_1 - decision_value_2
