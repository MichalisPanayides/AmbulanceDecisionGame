import numpy as np
import random
import ciw
from collections import Counter



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
        arrival_distributions = [ciw.dists.Exponential(lambda_a),
                                 ciw.dists.Exponential(lambda_o)],
        service_distributions = [ciw.dists.Deterministic(0),
                                 ciw.dists.Exponential(mu)],
        routing=[[0.0, 1.0],
                 [0.0, 0.0]],
        number_of_servers=[float('inf'), total_capacity]
    )
    return model


def build_custom_node(threshold=7):
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
                next_individual.server.next_end_service_date = float('Inf')
            blockage = (next_node.number_of_individuals >= threshold and self.id_number == 1)
            if (next_node.number_of_individuals < next_node.node_capacity) and not blockage:
                self.release(next_individual_index, next_node)
            else:
                self.block_individual(next_individual, next_node)
    return CustomNode


def simulate_model(lambda_a, lambda_o, mu, total_capacity, threshold, seed_num = random.random()):    
    """[Simulate the model]
    
    Parameters
    ----------
    seed_num : [float], optional
        [A seed number in order to be able to replicate results], by default random.random()
    
    Returns
    -------
    [object]
        [An object that contains all simulation records]
    """    
    model = build_model(lambda_a, lambda_o, mu, total_capacity)
    node = build_custom_node(threshold)
    ciw.seed(seed_num)
    Simulation = ciw.Simulation(model, node_class=node)
    Simulation.simulate_until_max_time(1440)
    return Simulation


print(type(build_model(1,1,3,1)))
print(type(build_custom_node(1)))