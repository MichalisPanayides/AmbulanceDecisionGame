"""
Code to create a queue object
"""

import ciw
import numpy as np

from .simulation import (
    simulate_model,
    get_simulated_state_probabilities,
    get_multiple_runs_results_from_simulations,
)
from .markov import (
    build_states,
    get_transition_matrix,
    visualise_markov_chain,
    get_symbolic_transition_matrix,
    get_markov_state_probabilities,
    get_steady_state_algebraically,
    get_mean_blocking_time_using_markov_state_probabilities,
    get_mean_waiting_time_using_markov_state_probabilities,
    proportion_within_target_using_markov_state_probabilities,
)


class Queue:
    """
    A class for a queueing system with two waiting zones; one that serves as a
    parking space and the other is wehre service takes place.
    """

    def __init__(
        self,
        lambda_1,
        lambda_2,
        mu,
        num_of_servers,
        threshold,
        system_capacity=float("inf"),
        buffer_capacity=float("inf"),
    ):
        self.parameters = {
            "lambda_1": lambda_1,
            "lambda_2": lambda_2,
            "mu": mu,
            "num_of_servers": num_of_servers,
            "threshold": threshold,
            "system_capacity": system_capacity,
            "buffer_capacity": buffer_capacity,
        }

        self.simulation = None
        self.simulation_state_probabilities = None
        self.simulation_main_results = None

        self.all_states = None
        self.markov_state_probabilities = None
        self.markov_main_results = None

    def simulate(self, num_of_trials=1, seed_num=0, runtime=1440):
        """
        Simulate the queueing system
        """
        self.simulation = simulate_model(
            lambda_1=self.parameters["lambda_1"],
            lambda_2=self.parameters["lambda_2"],
            mu=self.parameters["mu"],
            num_of_servers=self.parameters["num_of_servers"],
            threshold=self.parameters["threshold"],
            system_capacity=self.parameters["system_capacity"],
            buffer_capacity=self.parameters["buffer_capacity"],
            num_of_trials=num_of_trials,
            seed_num=seed_num,
            runtime=runtime,
        )

        if isinstance(self.simulation, list):
            self.simulation_state_probabilities = []
            for sim in self.simulation:
                self.simulation_state_probabilities.append(
                    get_simulated_state_probabilities(sim)
                )
        elif isinstance(self.simulation, ciw.Simulation):
            self.simulation_state_probabilities = get_simulated_state_probabilities(
                self.simulation
            )

    def simulation_main_performance_measures(
        self,
        target=1,
        warm_up_time=100,
        class_type=None,
    ):
        """
        Get the waiting time the blolcking time and the proportion of
        individuals within a predefined target.

        Note that this method is simulates the system again so it might give
        different results than the simulate method if no seed_num is given.
        """
        results = get_multiple_runs_results_from_simulations(
            simulations=self.simulation,
            target=target,
            class_type=class_type,
            warm_up_time=warm_up_time,
        )
        waiting_times = [np.mean(w.waiting_times) for w in results]
        blocking_times = [np.mean(w.blocking_times) for w in results]
        proportion_within_target = [p.proportion_within_target for p in results]

        self.simulation_main_results = {
            "waiting_times": waiting_times,
            "blocking_times": blocking_times,
            "proportion_within_target": proportion_within_target,
        }

    def markov_chain(self):
        """
        Gets all necessary pieces that form the markov chain model
        """
        if float("inf") in [
            self.parameters["system_capacity"],
            self.parameters["buffer_capacity"],
        ]:
            raise NotImplementedError(
                "Markov chain is not implemented for infinite system or buffer capacity"
            )

        self.all_states = build_states(
            threshold=self.parameters["threshold"],
            system_capacity=self.parameters["system_capacity"],
            buffer_capacity=self.parameters["buffer_capacity"],
        )

        transition_matrix = get_transition_matrix(
            lambda_1=self.parameters["lambda_1"],
            lambda_2=self.parameters["lambda_2"],
            mu=self.parameters["mu"],
            num_of_servers=self.parameters["num_of_servers"],
            threshold=self.parameters["threshold"],
            system_capacity=self.parameters["system_capacity"],
            buffer_capacity=self.parameters["buffer_capacity"],
        )

        self.markov_state_probabilities = get_markov_state_probabilities(
            get_steady_state_algebraically(transition_matrix), self.all_states
        )

    def visualise_markov_chain(self):
        """
        Visualise the markov chain model as a graph from networkx
        """
        return visualise_markov_chain(
            num_of_servers=self.parameters["num_of_servers"],
            threshold=self.parameters["threshold"],
            system_capacity=self.parameters["system_capacity"],
            buffer_capacity=self.parameters["buffer_capacity"],
        )

    def get_transition_matrix(self):
        """
        Get the transition matrix
        """
        return get_transition_matrix(
            lambda_1=self.parameters["lambda_1"],
            lambda_2=self.parameters["lambda_2"],
            mu=self.parameters["mu"],
            num_of_servers=self.parameters["num_of_servers"],
            threshold=self.parameters["threshold"],
            system_capacity=self.parameters["system_capacity"],
            buffer_capacity=self.parameters["buffer_capacity"],
        )

    def get_symbolic_transition_matrix(self):
        """
        Get the symbolic version of the transition matrix
        """
        return get_symbolic_transition_matrix(
            num_of_servers=self.parameters["num_of_servers"],
            threshold=self.parameters["threshold"],
            system_capacity=self.parameters["system_capacity"],
            buffer_capacity=self.parameters["buffer_capacity"],
        )

    def markov_main_performance_measures(self, class_type=None, target=float("inf")):
        """
        Get the main performance measures using the markov chain state ptobabilities
        """
        blocking_time = get_mean_blocking_time_using_markov_state_probabilities(
            lambda_2=self.parameters["lambda_2"],
            lambda_1=self.parameters["lambda_1"],
            mu=self.parameters["mu"],
            num_of_servers=self.parameters["num_of_servers"],
            threshold=self.parameters["threshold"],
            system_capacity=self.parameters["system_capacity"],
            buffer_capacity=self.parameters["buffer_capacity"],
        )

        waiting_time = get_mean_waiting_time_using_markov_state_probabilities(
            lambda_2=self.parameters["lambda_2"],
            lambda_1=self.parameters["lambda_1"],
            mu=self.parameters["mu"],
            num_of_servers=self.parameters["num_of_servers"],
            threshold=self.parameters["threshold"],
            system_capacity=self.parameters["system_capacity"],
            buffer_capacity=self.parameters["buffer_capacity"],
            class_type=class_type,
        )

        proportion_within_target = (
            proportion_within_target_using_markov_state_probabilities(
                lambda_2=self.parameters["lambda_2"],
                lambda_1=self.parameters["lambda_1"],
                mu=self.parameters["mu"],
                num_of_servers=self.parameters["num_of_servers"],
                threshold=self.parameters["threshold"],
                system_capacity=self.parameters["system_capacity"],
                buffer_capacity=self.parameters["buffer_capacity"],
                class_type=class_type,
                target=target,
            )
        )

        self.markov_main_results = {
            "waiting_time": waiting_time,
            "blocking_time": blocking_time,
            "proportion_within_target": proportion_within_target,
        }
