import csv
import itertools
import pathlib

import nashpy as nash
import numpy as np

import ambulance_game as abg


def looks_degenerate(A, B):
    """
    Check if the game looks degenerate i.e. if on a given column of the payoff
    matrix of A or a given row of the payoff matrix of B there are any duplicate
    maximum values
    """
    for col in A.transpose():
        max_value = np.max(col)
        max_duplicates = np.sum(col == max_value)
        if max_duplicates > 1:
            return True
    for row in B:
        max_value = np.max(row)
        max_duplicates = np.sum(row == max_value)
        if max_duplicates > 1:
            return True
    return False


def get_path_of_experiment():
    """Get the name of the directory for the current experiment"""
    path = str(pathlib.Path.cwd())
    parent_dir_index = path.rfind("\\")
    dirname = path[parent_dir_index + 1 :]
    target_path = pathlib.Path("../../data") / dirname
    return target_path


def get_parameters():
    """
    Get the values of the parameters for this experiment
    """
    keys = [
        "alpha",
        "buffer_capacity_1",
        "buffer_capacity_2",
        "lambda_1_1",
        "lambda_1_2",
        "lambda_2",
        "mu_1",
        "mu_2",
        "num_of_servers_1",
        "num_of_servers_2",
        "system_capacity_1",
        "system_capacity_2",
        "target",
    ]
    target_path = get_path_of_experiment()
    path = pathlib.Path(target_path) / "main.csv"
    with open(path, "r") as file:
        reader = csv.reader(file)
        values = tuple(reader)[0]

    parameters = {}
    for key, value in zip(keys, values):
        parameters[key] = float(value) if "." in value else int(value)
    return parameters


def get_matrices():
    """
    Get the generated matrices for this experiment
    """
    target_path = get_path_of_experiment()
    matrices = np.load(target_path / "main.npz")
    R = matrices["routing_matrix"]
    A = matrices["payoff_matrix_A"]
    B = matrices["payoff_matrix_B"]
    return R, A, B


def get_lemke_howson_outcome():
    """
    Get all unique outputs of the Lemke Howson algorithm for all dropped labels
    """
    _, A, B = get_matrices()
    game = nash.Game(A, B)
    all_equilibs = tuple(game.lemke_howson_enumeration())
    unique_equilibs = None
    for current_equilib in all_equilibs:
        if unique_equilibs is None:
            unique_equilibs = (current_equilib,)
            continue
        for check_equilib in unique_equilibs:
            if np.allclose(
                current_equilib[0], check_equilib[0], atol=1e-2
            ) and np.allclose(current_equilib[1], check_equilib[1], atol=1e-2):
                include = False
                break
            if include:
                unique_equilibs += (current_equilib,)
    return unique_equilibs


def get_fictitious_play_outcome(iterations=1000, repetitions=10):
    """
    Get the outputs of the fictitious play algorithm for a number of iterations.
    """
    _, A, B = get_matrices()
    game = nash.Game(A, B)
    all_outcomes = None
    for _ in range(repetitions):
        current_rep = tuple(game.fictitious_play(iterations))[-1]
        current_rep = tuple(i / np.sum(i) for i in current_rep)
        if all_outcomes is None:
            all_outcomes = (current_rep,)
            continue
        include = True
        for check_rep in all_outcomes:
            if np.allclose(current_rep[0], check_rep[0], atol=1e-1) and np.allclose(
                current_rep[1], check_rep[1], atol=1e-1
            ):
                include = False
                break
        if include:
            all_outcomes += (current_rep,)
    return all_outcomes


def get_stochastic_fictitious_play_outcome(iterations=10000, repetitions=10):
    """
    Get the outputs of the stochastic fictitious play algorithm for a number of iterations.
    """
    _, A, B = get_matrices()
    game = nash.Game(A, B)
    all_outcomes = None
    for _ in range(repetitions):
        current_rep = tuple(game.stochastic_fictitious_play(iterations))[-1][-1]
        if all_outcomes == None:
            all_outcomes = (current_rep,)
        include = True
        for check_rep in all_outcomes:
            if np.allclose(current_rep[0], check_rep[0], atol=1e-1) and np.allclose(
                current_rep[1], check_rep[1], atol=1e-1
            ):
                include = False
                break
        if include:
            all_outcomes += (current_rep,)
    return all_outcomes


def get_performance_measure_for_given_strategies(
    strategy_A, strategy_B, routing, parameters, performance_measure_function
):
    """
    For a given set of strategies get the sum of a given performance measure of the two players
    """
    prop_1 = routing[strategy_A, strategy_B]
    lambda_2_1 = parameters["lambda_2"] * prop_1
    lambda_2_2 = parameters["lambda_2"] * (1 - prop_1)

    performance_measure_1 = performance_measure_function(
        lambda_2=lambda_2_1,
        lambda_1=parameters["lambda_1_1"],
        mu=parameters["mu_1"],
        num_of_servers=int(parameters["num_of_servers_1"]),
        threshold=strategy_A + 1,
        system_capacity=int(parameters["system_capacity_1"]),
        buffer_capacity=int(parameters["buffer_capacity_1"]),
    )

    performance_measure_2 = performance_measure_function(
        lambda_2=lambda_2_2,
        lambda_1=parameters["lambda_1_2"],
        mu=parameters["mu_2"],
        num_of_servers=int(parameters["num_of_servers_2"]),
        threshold=strategy_B + 1,
        system_capacity=int(parameters["system_capacity_2"]),
        buffer_capacity=int(parameters["buffer_capacity_2"]),
    )

    if (
        performance_measure_function
        == abg.markov.get_accepting_proportion_of_class_2_individuals
    ):
        return 2 - performance_measure_1 - performance_measure_2
    return performance_measure_1 + performance_measure_2


def build_performance_values_array(routing, parameters, performance_measure_function):
    """
    Get all the values for the current investigated performance measure
    """
    all_performance_values = np.zeros(routing.shape)
    for strategy_A, strategy_B in itertools.product(
        range(routing.shape[0]), range(routing.shape[1])
    ):
        all_performance_values[
            strategy_A, strategy_B
        ] = get_performance_measure_for_given_strategies(
            strategy_A=strategy_A,
            strategy_B=strategy_B,
            routing=routing,
            parameters=parameters,
            performance_measure_function=performance_measure_function,
        )
    return all_performance_values


def find_worst_nash_equilibrium_measure(
    all_nash_equilibrias,
    performance_values_array,
):
    """
    Get the maximum value of the performance measure out of all possible
    equilibria
    """
    max_performance_measure = None
    for row_strategies, col_strategies in all_nash_equilibrias:
        current_performance_measure = (
            row_strategies @ performance_values_array @ col_strategies
        )
        if (
            max_performance_measure is None
            or current_performance_measure > max_performance_measure
        ):
            max_performance_measure = current_performance_measure
    return max_performance_measure


def get_price_of_anarchy(performance_measure_function):
    """
    Get the price of anarchy for the performance measure function given. Possible
    performance_measure_functions:
        - Mean Blocking Time
        - Mean Waiting Time
        - Proportion of lost class 2 individuals
    """
    parameters = get_parameters()
    routing, A, B = get_matrices()
    #     if looks_degenerate(A, B):
    #         equilibria = get_fictitious_play_outcome()
    #     else:
    #         equilibria = get_lemke_howson_outcome()
    equilibria = get_lemke_howson_outcome()

    performance_values_array = build_performance_values_array(
        routing=routing,
        parameters=parameters,
        performance_measure_function=performance_measure_function,
    )
    minimum_value = np.min(performance_values_array)
    worst_equilib_value = find_worst_nash_equilibrium_measure(
        all_nash_equilibrias=equilibria,
        performance_values_array=performance_values_array,
    )
    price_of_anarchy = worst_equilib_value / minimum_value
    return price_of_anarchy
