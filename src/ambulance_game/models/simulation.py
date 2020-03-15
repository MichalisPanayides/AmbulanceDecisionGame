import numpy as np
import random
import ciw
import collections


def build_model(lambda_a, lambda_o, mu, total_capacity):
    """ Builds the required ciw network

    Parameters
    ----------
    lambda_a : [float]
        [Arrival rate of ambulance patients]
    lambda_o : [float]
        [Arrival rate of other patients]
    mu : [float]
        [Service rate of hospital]
    total_capacity : [integer]
        [The total capacity of the hospital]
    """
    model = ciw.create_network(
        arrival_distributions=[
            ciw.dists.Exponential(lambda_a),
            ciw.dists.Exponential(lambda_o),
        ],
        service_distributions=[ciw.dists.Deterministic(0), ciw.dists.Exponential(mu)],
        routing=[[0.0, 1.0], [0.0, 0.0]],
        number_of_servers=[float("inf"), total_capacity],
    )
    return model


def build_custom_node(threshold=8):
    """Build a custome node to replace the default ciw.Node
    
    Parameters
    ----------
    threshold : [int], optional
        [The capacity threshold to be used by the method, by default 7]
    
    Returns
    -------
    [class]
        [A custom node class that inherits from ciw.Node]
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
                self.number_of_individuals > threshold and self.id_number == 2
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


def simulate_model(lambda_a, lambda_o, mu, total_capacity, threshold, seed_num=None):
    """Simulating the model and returning the simulation object
 
    Parameters
    ----------
    seed_num : [float], optional
        [A seed number in order to be able to replicate results], by default random.random()
    
    Returns
    -------
    [object]
        [An object that contains all simulation records]
    """
    if seed_num == None:
        seed_num = random.random()
    model = build_model(lambda_a, lambda_o, mu, total_capacity)
    node = build_custom_node(threshold)
    ciw.seed(seed_num)
    simulation = ciw.Simulation(model, node_class=node)
    simulation.simulate_until_max_time(1440)
    return simulation


def extract_times_from_records(simulation_records, warm_up_time):
    """Get the required times that we are interested in out of ciw's records
    
    Parameters
    ----------
    simulation_records : [list]
        [A list of all simulated records]
    warm_up_time : [int]
        [The time we start collecting results]
    
    Returns
    -------
    [list, list, list]
        [Three lists that contain waiting, service and blocking times]
    """
    waiting = [r.waiting_time for r in simulation_records if r.arrival_date > warm_up_time]
    serving = [r.service_time for r in simulation_records if r.arrival_date > warm_up_time]
    blocking = [r.time_blocked for r in simulation_records if r.arrival_date > warm_up_time]
    return waiting, serving, blocking
    

def get_list_of_results(results):
    """Modify the outputs even further so that it is output in a different more convenient format for some graphs 
    
    Parameters
    ----------
    results : [list]
        [A list of named tuples for each iteration]
    
    Returns
    -------
    [list, list, list]
        [Three lists that include all waits, services and blocks of all runs of all individuals]
    """
    all_waits = [w.waiting_times for w in results]
    all_services = [s.service_times for s in results]
    all_blocks = [b.blocking_times for b in results]
    return all_waits, all_services, all_blocks


def get_multiple_runs_results(lambda_a, lambda_o, mu, total_capacity, threshold, seed_num=None, warm_up_time=100, num_of_trials=10, output_type="tuple"):
    """[summary]
    
    Parameters
    ----------
    warm_up_time : int, optional
        [Time to start collecting results], by default 100
    num_of_trials : int, optional
        [Number of trials to run the model], by default 10
    output_type : str, optional
        [The results' output type (either tuple or list)], by default "tuple"
    
    Returns
    -------
    [type]
        [description]
    """
    if seed_num == None:
        seed_num = random.random()
    records = collections.namedtuple('records', 'waiting_times service_times blocking_times')
    results = []
    for trial in range(num_of_trials):
        simulation = simulate_model(lambda_a, lambda_o, mu, total_capacity, threshold, seed_num + trial)
        sim_results = simulation.get_all_records()
        ext = extract_times_from_records(sim_results, warm_up_time)
        results.append(records(ext[0], ext[1], ext[2]))
        
    if output_type == "list":
        all_waits, all_services, all_blocks = get_list_of_results(results)
        return [all_waits, all_services, all_blocks]
    
    return results