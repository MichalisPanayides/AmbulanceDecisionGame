import numpy as np
import random
import ciw
import collections
import scipy.optimize


def build_model(
    lambda_a,
    lambda_o,
    mu,
    num_of_servers,
    system_capacity=float("inf"),
    parking_capacity=float("inf"),
):
    """ Builds the required ciw network

    Parameters
    ----------
    lambda_a : float
        Arrival rate of ambulance patients
    lambda_o : float
        Arrival rate of other patients
    mu : float
        Service rate of hospital
    num_of_servers : integer
        The num_of_servers of the hospital
    """

    model = ciw.create_network(
        arrival_distributions=[
            ciw.dists.Exponential(lambda_a),
            ciw.dists.Exponential(lambda_o),
        ],
        service_distributions=[ciw.dists.Deterministic(0), ciw.dists.Exponential(mu)],
        routing=[[0.0, 1.0], [0.0, 0.0]],
        number_of_servers=[parking_capacity, num_of_servers],
        queue_capacities=[0, system_capacity - num_of_servers],
    )
    return model


def build_custom_node(threshold=8):
    """Build a custome node to replace the default ciw.Node
    
    Parameters
    ----------
    threshold : int, optional
        The capacity threshold to be used by the method, by default 7
    
    Returns
    -------
    class
        A custom node class that inherits from ciw.Node
    """

    class CustomNode(ciw.Node):
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
                if individual_to_receive.interrupted:
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
                number of individuals on node 2 are more than the 'thershold'
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
    lambda_a,
    lambda_o,
    mu,
    num_of_servers,
    threshold,
    seed_num=None,
    runtime=1440,
    system_capacity=float("inf"),
    parking_capacity=float("inf"),
    tracker=ciw.trackers.NodePopulation()
):
    """Simulating the model and returning the simulation object
 
    Parameters
    ----------
    seed_num : [float], optional
        A seed number in order to be able to replicate results, by default random.random()
    
    Returns
    -------
    object
        An object that contains all simulation records
    """
    if seed_num == None:
        seed_num = random.random()
    model = build_model(
        lambda_a, lambda_o, mu, num_of_servers, system_capacity, parking_capacity
    )
    node = build_custom_node(threshold)
    ciw.seed(seed_num)
    simulation = ciw.Simulation(model, node_class=node, tracker=tracker)
    simulation.simulate_until_max_time(runtime)
    return simulation


def get_pi(Q):
    pi_dictionary = Q.statetracker.state_probabilities()
    return pi_dictionary


def extract_times_from_records(simulation_records, warm_up_time):
    """Get the required times that we are interested in out of ciw's records
    
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
    waiting = [
        r.waiting_time
        for r in simulation_records
        if r.arrival_date > warm_up_time and r.node == 2
    ]
    serving = [
        r.service_time
        for r in simulation_records
        if r.arrival_date > warm_up_time and r.node == 2
    ]
    blocking = [
        r.time_blocked
        for r in simulation_records
        if r.arrival_date > warm_up_time and r.node == 1
    ]
    return waiting, serving, blocking


def get_list_of_results(results):
    """Modify the outputs even further so that it is output in a different more convenient format for some graphs 
    
    Parameters
    ----------
    results : list
        A list of named tuples for each iteration
    
    Returns
    -------
    list, list, list
        Three lists that include all waits, services and blocks of all runs of all individuals
    """
    all_waits = [w.waiting_times for w in results]
    all_services = [s.service_times for s in results]
    all_blocks = [b.blocking_times for b in results]
    return all_waits, all_services, all_blocks


def get_multiple_runs_results(
    lambda_a,
    lambda_o,
    mu,
    num_of_servers,
    threshold,
    seed_num=None,
    warm_up_time=100,
    num_of_trials=10,
    runtime=1440,
    output_type="tuple",
    system_capacity=float("inf"),
    parking_capacity=float("inf"),
):
    """Get waiting, service and blocking times for multiple runs 
    
    Parameters
    ----------
    warm_up_time : int, optional
        Time to start collecting results, by default 100
    num_of_trials : int, optional
        Number of trials to run the model, by default 10
    output_type : str, optional
        The results' output type (either tuple or list)], by default "tuple"
    
    Returns
    -------
    list
        A list of records where each record consists of the waiting, service and blocking times of one trial. Alternatively if the output_type = "list" then returns theree lists with all waiting, service and blocking times

    """
    if seed_num == None:
        seed_num = random.random()
    records = collections.namedtuple(
        "records", "waiting_times service_times blocking_times"
    )
    results = []
    for trial in range(num_of_trials):
        simulation = simulate_model(
            lambda_a,
            lambda_o,
            mu,
            num_of_servers,
            threshold,
            seed_num + trial,
            runtime,
            system_capacity,
            parking_capacity,
        )
        sim_results = simulation.get_all_records()
        ext = extract_times_from_records(sim_results, warm_up_time)
        results.append(records(ext[0], ext[1], ext[2]))

    if output_type == "list":
        all_waits, all_services, all_blocks = get_list_of_results(results)
        return [all_waits, all_services, all_blocks]

    return results


def get_mean_blocking_difference_between_two_hospitals(
    prop_1,
    lambda_a,
    lambda_o_1,
    lambda_o_2,
    mu_1,
    mu_2,
    num_of_servers_1,
    num_of_servers_2,
    threshold_1,
    threshold_2,
    seed_num_1,
    seed_num_2,
    num_of_trials,
    warm_up_time,
    runtime,
):
    """Given a predefined proportion of the ambulance's arrival rate calculate the mean difference between blocking times of two hospitals with given set of parameters. Note that all parameters that end in "_1" correspond to the first hospital while "_2" to the second.
    
    Parameters
    ----------
    prop_1 : float
        Proportion of ambulance's arrival rate that will be distributed to hospital 1
    lambda_a : float
        Total ambulance arrival rate
    
    Returns
    -------
    float
        The difference between the mean blocking time of the two hospitals
    """
    lambda_a_1 = prop_1 * lambda_a
    lambda_a_2 = (1 - prop_1) * lambda_a

    res_1 = get_multiple_runs_results(
        lambda_a=lambda_a_1,
        lambda_o=lambda_o_1,
        mu=mu_1,
        num_of_servers=num_of_servers_1,
        threshold=threshold_1,
        seed_num=seed_num_1,
        warm_up_time=warm_up_time,
        num_of_trials=num_of_trials,
        output_type="tuple",
        runtime=runtime,
    )
    res_2 = get_multiple_runs_results(
        lambda_a=lambda_a_2,
        lambda_o=lambda_o_2,
        mu=mu_2,
        num_of_servers=num_of_servers_2,
        threshold=threshold_2,
        seed_num=seed_num_2,
        warm_up_time=warm_up_time,
        num_of_trials=num_of_trials,
        output_type="tuple",
        runtime=runtime,
    )

    hospital_1_blockages = [
        np.nanmean(b.blocking_times) if len(b.blocking_times) != 0 else 0 for b in res_1
    ]
    hospital_2_blockages = [
        np.nanmean(b.blocking_times) if len(b.blocking_times) != 0 else 0 for b in res_2
    ]
    diff = np.mean(hospital_1_blockages) - np.mean(hospital_2_blockages)

    return diff


def calculate_optimal_ambulance_distribution(
    lambda_a,
    lambda_o_1,
    lambda_o_2,
    mu_1,
    mu_2,
    num_of_servers_1,
    num_of_servers_2,
    threshold_1,
    threshold_2,
    seed_num_1,
    seed_num_2,
    num_of_trials,
    warm_up_time,
    runtime,
):
    """Obtains the optimal distribution of ambulances such that the blocking times of the ambulances in the two hospitals are identical and thus optimal(minimised). 
    
    The brentq function is used which is an algorithm created to find the root of a function that combines root bracketing, bisection, and inverse quadratic interpolation. In this specific example the root to be found is the difference between the blocking times of two hospitals. In essence the brentq algorith atempts to find the value of "prop_1" where the "diff" is zero (see function: get_mean_blocking_difference_between_two_hospitals).
    
    Returns
    -------
    float
        The optimal proportion where the hospitals have identical blocking times
    """
    optimal_prop = scipy.optimize.brentq(
        get_mean_blocking_difference_between_two_hospitals,
        a=0.01,
        b=0.99,
        args=(
            lambda_a,
            lambda_o_1,
            lambda_o_2,
            mu_1,
            mu_2,
            num_of_servers_1,
            num_of_servers_2,
            threshold_1,
            threshold_2,
            seed_num_1,
            seed_num_2,
            num_of_trials,
            warm_up_time,
            runtime,
        ),
    )
    return optimal_prop
