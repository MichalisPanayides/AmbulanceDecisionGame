import numpy as np
import matplotlib.pyplot as plt

from .simulation.simulation import (
    simulate_model,
    get_multiple_runs_results,
    get_simulated_state_probabilities,
    get_average_simulated_state_probabilities,
)

from .markov.markov import (
    build_states,
    get_transition_matrix,
    get_steady_state_algebraically,
    get_markov_state_probabilities,
)


def get_heatmaps_0(
    lambda_a,
    lambda_o,
    mu,
    num_of_servers,
    threshold,
    system_capacity,
    parking_capacity,
    seed_num=None,
    runtime=1440,
    num_of_trials=1,
):

    simulation_object = simulate_model(
        lambda_a,
        lambda_o,
        mu,
        num_of_servers,
        threshold,
        seed_num,
        runtime,
        system_capacity,
        parking_capacity,
    )

    all_states = build_states(threshold, system_capacity, parking_capacity)
    transition_matrix = get_transition_matrix(
        lambda_a,
        lambda_o,
        mu,
        num_of_servers,
        threshold,
        system_capacity,
        parking_capacity,
    )
    pi = get_steady_state_algebraically(
        transition_matrix, algebraic_function=np.linalg.lstsq
    )

    sim_state_probabilities_array = get_simulated_state_probabilities(
        simulation_object=simulation_object,
        output=np.ndarray,
        system_capacity=system_capacity,
        parking_capacity=parking_capacity,
    )
    markov_state_probabilities_array = get_markov_state_probabilities(
        pi=pi,
        all_states=all_states,
        output=np.ndarray,
        system_capacity=system_capacity,
        parking_capacity=parking_capacity,
    )
    diff_states_probabilities_array = (
        sim_state_probabilities_array - markov_state_probabilities_array
    )

    plt.figure(figsize=(20, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(sim_state_probabilities_array, cmap="viridis")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(markov_state_probabilities_array, cmap="viridis")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(diff_states_probabilities_array, cmap="viridis")
    plt.colorbar()


def get_heatmaps(
    lambda_a,
    lambda_o,
    mu,
    num_of_servers,
    threshold,
    system_capacity,
    parking_capacity,
    seed_num=None,
    runtime=1440,
    num_of_trials=10,
):

    all_states = build_states(threshold, system_capacity, parking_capacity)
    transition_matrix = get_transition_matrix(
        lambda_a,
        lambda_o,
        mu,
        num_of_servers,
        threshold,
        system_capacity,
        parking_capacity,
    )
    pi = get_steady_state_algebraically(
        transition_matrix, algebraic_function=np.linalg.lstsq
    )

    sim_state_probabilities_array = get_average_simulated_state_probabilities(
        lambda_a,
        lambda_o,
        mu,
        num_of_servers,
        threshold,
        system_capacity,
        parking_capacity,
        seed_num=seed_num,
        runtime=runtime,
        num_of_trials=num_of_trials,
        output=np.ndarray,
    )
    markov_state_probabilities_array = get_markov_state_probabilities(
        pi=pi,
        all_states=all_states,
        output=np.ndarray,
        system_capacity=system_capacity,
        parking_capacity=parking_capacity,
    )
    diff_states_probabilities_array = (
        sim_state_probabilities_array - markov_state_probabilities_array
    )

    plt.figure(figsize=(20, 20))
    grid = plt.GridSpec(2, 4)
    # plt.subplot(1, 3, 1)
    plt.subplot(grid[0,0:2])
    plt.imshow(sim_state_probabilities_array, cmap="viridis")
    plt.title("Simulatioin state probabilities")
    plt.xlabel("Patients in Hospital")
    plt.ylabel("Patients blocked")
    plt.colorbar()

    # plt.subplot(1, 3, 2)
    plt.subplot(grid[0,2:4])
    plt.imshow(markov_state_probabilities_array, cmap="viridis")
    plt.title("Markov chain state probabilities")
    plt.xlabel("Patients in Hospital")
    plt.ylabel("Patients blocked")
    plt.colorbar()

    # plt.subplot(1, 3, 3)
    plt.subplot(grid[1,1:3])
    plt.imshow(diff_states_probabilities_array, cmap="viridis")
    plt.title("Simulation and Markov chain state probability differences")
    plt.xlabel("Patients in Hospital")
    plt.ylabel("Patients blocked")
    plt.colorbar()
