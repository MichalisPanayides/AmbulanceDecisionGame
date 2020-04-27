import numpy as np
import matplotlib.pyplot as plt

from .simulation.simulation import (
    simulate_model,
    get_multiple_runs_results,
    get_sim_pi_array,
)

from .markov.markov import (
    build_states,
    get_transition_matrix,
    get_steady_state_algebraically,
    get_mar_pi_array,
)


def get_heatmaps(lambda_a, lambda_o, mu, num_of_servers, threshold, system_capacity, parking_capacity, seed_num=None, runtime=1440, num_of_trials=1):

    simulation_object = simulate_model(lambda_a, lambda_o, mu, num_of_servers, threshold, seed_num, runtime, system_capacity, parking_capacity)
    
    all_states = build_states(threshold, system_capacity, parking_capacity)
    transition_matrix = get_transition_matrix(lambda_a, lambda_o, mu, num_of_servers, threshold, system_capacity, parking_capacity)
    pi = get_steady_state_algebraically(transition_matrix, algebraic_function=np.linalg.lstsq)
    
    parking_capacity = max([state[0] for state in all_states])
    system_capacity = max([state[1] for state in all_states])

    sim_state_probabilities_array = get_sim_pi_array(simulation_object, system_capacity, parking_capacity)
    markov_state_probabilities_array = get_mar_pi_array(pi, all_states, system_capacity, parking_capacity)
    diff_states_probabilities_array = sim_state_probabilities_array - markov_state_probabilities_array
    
    plt.figure(figsize=(20,5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(sim_state_probabilities_array, cmap="viridis")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(markov_state_probabilities_array, cmap='viridis')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(diff_states_probabilities_array, cmap='viridis')
    plt.colorbar()