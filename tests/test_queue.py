"""
Test the Queue class
"""

import unittest

import ciw
import networkx as nx
import numpy as np
import sympy as sym

from ambulance_game.queue import Queue


class TestQueue(unittest.TestCase):
    """
    Tests for the Queue class
    """

    def test_finite_capacity_init(self):
        """
        Tests that the Queue class can be initialized with finite capacity
        """
        Q = Queue(1, 1, 2, 2, 1, system_capacity=10, buffer_capacity=10)
        self.assertEqual(Q.parameters["lambda_1"], 1)
        self.assertEqual(Q.parameters["lambda_2"], 1)
        self.assertEqual(Q.parameters["mu"], 2)
        self.assertEqual(Q.parameters["num_of_servers"], 2)
        self.assertEqual(Q.parameters["threshold"], 1)
        self.assertEqual(Q.parameters["system_capacity"], 10)
        self.assertEqual(Q.parameters["buffer_capacity"], 10)

        assert Q.simulation is None
        assert Q.simulation_state_probabilities is None
        assert Q.simulation_main_results is None

        assert Q.all_states is None
        assert Q.markov_state_probabilities is None
        assert Q.markov_main_results is None

    def test_inf_capacity_init(self):
        """
        Test that the Queue class can be initialized with infinite capacity
        """
        Q = Queue(1, 1, 2, 2, 1)
        self.assertEqual(Q.parameters["lambda_1"], 1)
        self.assertEqual(Q.parameters["lambda_2"], 1)
        self.assertEqual(Q.parameters["mu"], 2)
        self.assertEqual(Q.parameters["num_of_servers"], 2)
        self.assertEqual(Q.parameters["threshold"], 1)
        self.assertEqual(Q.parameters["system_capacity"], float("inf"))
        self.assertEqual(Q.parameters["buffer_capacity"], float("inf"))

    def test_simulate_one_trial(self):
        """
        Test that the Queue class can simulate one trial
        """
        Q = Queue(1, 1, 2, 2, 1, system_capacity=10, buffer_capacity=10)
        Q.simulate(runtime=100, seed_num=0)
        self.assertTrue(isinstance(Q.simulation, ciw.Simulation))

        expeceted_probs = np.array(
            [
                [
                    2.64308222e-01,
                    2.41130001e-01,
                    3.90850533e-02,
                    7.85035814e-03,
                    6.35350098e-03,
                    1.21306115e-03,
                ],
                [
                    np.nan,
                    1.20800636e-01,
                    3.34784609e-02,
                    4.99538834e-03,
                    3.74648572e-05,
                    np.nan,
                ],
                [
                    np.nan,
                    8.43296880e-02,
                    2.68301724e-02,
                    1.95603107e-03,
                    4.53555034e-04,
                    np.nan,
                ],
                [
                    np.nan,
                    5.84135272e-02,
                    3.01553006e-02,
                    8.32715781e-03,
                    6.35521059e-03,
                    np.nan,
                ],
                [np.nan, 2.18935256e-02, 4.11064706e-03, np.nan, np.nan, np.nan],
                [np.nan, 1.42699504e-02, 4.51499236e-03, np.nan, np.nan, np.nan],
                [np.nan, 2.57975386e-03, np.nan, np.nan, np.nan, np.nan],
                [np.nan, 5.13540531e-03, np.nan, np.nan, np.nan, np.nan],
                [np.nan, 1.14229365e-02, np.nan, np.nan, np.nan, np.nan],
            ]
        )

        for i, expected_i in zip(Q.simulation_state_probabilities, expeceted_probs):
            for j, expected_j in zip(i, expected_i):
                if not np.isnan(j) and not np.isnan(expected_j):
                    self.assertAlmostEqual(j, expected_j)

    def test_simulate_multiple_trials(self):
        """
        Test that simulation for multiple trials works as expected
        """
        Q = Queue(2, 1, 2, 3, 4, system_capacity=8, buffer_capacity=6)
        Q.simulate(runtime=100, seed_num=0, num_of_trials=3)
        self.assertTrue(isinstance(Q.simulation, list))
        for sim in Q.simulation:
            self.assertTrue(isinstance(sim, ciw.Simulation))

        expeceted_probs = [
            np.array(
                [
                    [
                        0.19345697,
                        0.35856713,
                        0.23898119,
                        0.1141801,
                        0.04629752,
                        0.02284553,
                        0.00266527,
                        0.00214411,
                    ],
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        0.00981092,
                        0.00410901,
                        0.00303729,
                        0.00072347,
                    ],
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        0.00215636,
                        0.00102515,
                        np.nan,
                        np.nan,
                    ],
                ]
            ),
            np.array(
                [
                    [
                        0.19345697,
                        0.35856713,
                        0.23898119,
                        0.1141801,
                        0.04629752,
                        0.02284553,
                        0.00266527,
                        0.00214411,
                    ],
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        0.00981092,
                        0.00410901,
                        0.00303729,
                        0.00072347,
                    ],
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        0.00215636,
                        0.00102515,
                        np.nan,
                        np.nan,
                    ],
                ]
            ),
            np.array(
                [
                    [
                        0.19345697,
                        0.35856713,
                        0.23898119,
                        0.1141801,
                        0.04629752,
                        0.02284553,
                        0.00266527,
                        0.00214411,
                    ],
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        0.00981092,
                        0.00410901,
                        0.00303729,
                        0.00072347,
                    ],
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        0.00215636,
                        0.00102515,
                        np.nan,
                        np.nan,
                    ],
                ]
            ),
        ]

        for k, expected_k in zip(Q.simulation_state_probabilities, expeceted_probs):
            for i, expected_i in zip(k, expected_k):
                for j, expected_j in zip(i, expected_i):
                    if not np.isnan(j) and not np.isnan(expected_j):
                        self.assertAlmostEqual(j, expected_j)

    def test_simulation_main_performance_measures_example(self):
        """
        Test that getting the main performance measures form the simulation
        works as expected
        """
        Q = Queue(2, 1, 2, 3, 4, system_capacity=8, buffer_capacity=6)
        Q.simulation_main_performance_measures(seed_num=0)
        self.assertEqual(
            Q.simulation_main_results["waiting_times"][0],
            0.05471029486451625,
        )
        self.assertEqual(
            Q.simulation_main_results["blocking_times"][0],
            0.045058211431798834,
        )
        self.assertEqual(
            Q.simulation_main_results["proportion_within_target"][0],
            0.8495534692008244,
        )

    def test_markov_chain_example(self):
        """
        Test that when the markov_chain() method is ran Q.all_states and
        Q.markov_state_probabilities are populated correctly
        """
        Q = Queue(2, 1, 3, 4, 3, system_capacity=4, buffer_capacity=2)
        Q.markov_chain()
        self.assertEqual(
            Q.all_states,
            [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (0, 4), (1, 4), (2, 4)],
        )

        expected_state_probs = np.array(
            [
                [
                    3.67265630e-01,
                    3.67265630e-01,
                    1.83632815e-01,
                    6.12109383e-02,
                    9.41706743e-03,
                ],
                [np.nan, np.nan, np.nan, 7.84755619e-03, 1.93170614e-03],
                [np.nan, np.nan, np.nan, 1.08658470e-03, 3.42072962e-04],
            ]
        )

        for row, expected_row in zip(
            Q.markov_state_probabilities, expected_state_probs
        ):
            for prob, expected_prob in zip(row, expected_row):
                if not np.isnan(prob) and not np.isnan(expected_prob):
                    self.assertAlmostEqual(prob, expected_prob)

    def test_markov_chain_error_for_inf_capacities(self):
        """
        Test that when infinit capacity is used in the markov chain method
        for either the system or the buffer, an error is raised
        """
        Q = Queue(2, 1, 3, 4, 3, system_capacity=np.inf, buffer_capacity=2)
        self.assertRaises(NotImplementedError, Q.markov_chain)

        Queue(2, 1, 3, 4, 3, system_capacity=4, buffer_capacity=np.inf)
        self.assertRaises(NotImplementedError, Q.markov_chain)

        Queue(2, 1, 3, 4, 3, system_capacity=np.inf, buffer_capacity=np.inf)
        self.assertRaises(NotImplementedError, Q.markov_chain)

    def test_visualise_markov_chain(self):
        """
        Make sure that the visualise_markov_chain() method outputs a networkx
        DiGraph with the same states as Q.all_states
        """
        Q = Queue(2, 1, 3, 4, 3, system_capacity=4, buffer_capacity=2)
        Q.markov_chain()
        graph = Q.visualise_markov_chain()

        self.assertTrue(isinstance(graph, nx.DiGraph))
        for state in graph.nodes:
            self.assertTrue(state in Q.all_states)

    def test_get_transition_matrix_example(self):
        """
        Test that the get_transition_matrix() method returns the correct
        transition matrix
        """
        Q = Queue(0.5, 0.5, 3, 2, 3, system_capacity=3, buffer_capacity=1)
        Q.markov_chain()
        expected_transition_matrix = np.array(
            [
                [-1.0, 1.0, 0.0, 0.0, 0.0],
                [3.0, -4.0, 1.0, 0.0, 0.0],
                [0.0, 6.0, -7.0, 1.0, 0.0],
                [0.0, 0.0, 6.0, -6.5, 0.5],
                [0.0, 0.0, 0.0, 6.0, -6.0],
            ]
        )
        self.assertTrue(
            np.allclose(Q.get_transition_matrix(), expected_transition_matrix)
        )

    def test_get_symbolic_transition_matrix(self):
        """
        Test that the symbolic transition matrix is correctly generated
        """
        Q = Queue(0.5, 0.5, 3, 2, 1, system_capacity=2, buffer_capacity=1)

        Lambda = sym.Symbol("Lambda")
        lambda_1 = sym.Symbol("lambda_1")
        lambda_2 = sym.Symbol("lambda_2")
        mu = sym.Symbol("mu")

        calculated_transition_matrix = Q.get_symbolic_transition_matrix()
        self.assertTrue(isinstance(calculated_transition_matrix, sym.Matrix))

        expected_transition_matrix = sym.Matrix(
            [
                [-Lambda, Lambda, 0, 0, 0],
                [mu, -lambda_1 - lambda_2 - mu, lambda_2, lambda_1, 0],
                [0, mu, -lambda_1 - mu, 0, lambda_1],
                [0, 2 * mu, 0, -lambda_2 - 2 * mu, lambda_2],
                [0, 0, 2 * mu, 0, -2 * mu],
            ]
        )

        for calculated, expected in zip(
            calculated_transition_matrix, expected_transition_matrix
        ):
            self.assertEqual(calculated, expected)

    def test_markov_main_performance_measures_example(self):
        """
        Test that the main performance measures are correct when the
        markov_main_performance_measures() method is ran
        """
        Q = Queue(2, 2, 3, 2, 2, system_capacity=3, buffer_capacity=1)
        Q.markov_chain()
        Q.markov_main_performance_measures(target=1)

        self.assertAlmostEqual(
            Q.markov_main_results["waiting_time"], 0.029824561403508774
        )
        self.assertAlmostEqual(
            Q.markov_main_results["blocking_time"], 0.08243727598566308
        )
        self.assertAlmostEqual(
            Q.markov_main_results["proportion_within_target"], 0.9417472329452903
        )
