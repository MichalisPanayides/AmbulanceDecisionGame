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
    mean_waiting_time_formula,
    get_mean_waiting_time_markov,
)


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
    linear_positioning=False,
):
    """Get heatmaps plot that compare the state probabilities of the simulation and markov state probabilities. In total three heatmaps are generated; one for the simulation state probabilities, one for the markov state probabilities and one for the difference between the two

    Parameters
    ----------
    lambda_a : float
    lambda_o : float
    mu : float
    num_of_servers : int
    threshold : int
    system_capacity : int
    parking_capacity : int
    seed_num : float, optional
    runtime : int, optional
    num_of_trials : int, optional
    linear_positioning : boolean, optional
        To distinguish between the two position formats of the heatmaps, by default False
    """
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

    if not linear_positioning:
        grid = plt.GridSpec(2, 4)
        plt.subplot(grid[0, 0:2])
    else:
        plt.subplot(1, 3, 1)
    plt.imshow(sim_state_probabilities_array, cmap="cividis")
    plt.title("Simulation state probabilities")
    plt.xlabel("Patients in Hospital")
    plt.ylabel("Patients blocked")
    plt.colorbar()

    if not linear_positioning:
        plt.subplot(grid[0, 2:4])
    else:
        plt.subplot(1, 3, 2)

    plt.imshow(markov_state_probabilities_array, cmap="cividis")
    plt.title("Markov chain state probabilities")
    plt.xlabel("Patients in Hospital")
    plt.ylabel("Patients blocked")
    plt.colorbar()

    if not linear_positioning:
        plt.subplot(grid[1, 1:3])
    else:
        plt.subplot(1, 3, 3)
    plt.imshow(diff_states_probabilities_array, cmap="viridis")
    plt.title("Simulation and Markov chain state probability differences")
    plt.xlabel("Patients in Hospital")
    plt.ylabel("Patients blocked")
    plt.colorbar()


def get_mean_waiting_time_from_simulation_state_probabilities(
    lambda_a,
    lambda_o,
    mu,
    num_of_servers,
    threshold,
    system_capacity,
    parking_capacity,
    seed_num,
    runtime=1440,
    num_of_trials=10,
    output="both",
):
    """An alternative approach to obtaining the mean waiting time from the simulation. This function gets the mean waiting time from the simulation state probabilities. This is mainly used in comparing the simulation results with the markov ones.

    Parameters
    ----------
    lambda_a : float
    lambda_o : float
    mu : float
    num_of_servers : int
    threshold : int
    system_capacity : int
    parking_capacity : int
    seed_num : float
    num_of_trials : int
    output : str, optional
        A string to identify wheteher to get the waiting time of other patients, ambulance patients or the overall of both, by default "both"

    Returns
    -------
    float
        The waiting time in the system of the given patient type
    """
    state_probabilities = get_average_simulated_state_probabilities(
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
    )
    all_states = [
        (u, v)
        for v in range(state_probabilities.shape[1])
        for u in range(state_probabilities.shape[0])
        if state_probabilities[u, v] > 0
    ]

    if output == "both":
        mean_waiting_time_other = mean_waiting_time_formula(
            all_states,
            state_probabilities,
            "others",
            lambda_a,
            lambda_o,
            mu,
            num_of_servers,
            threshold,
            system_capacity,
            parking_capacity,
        )
        mean_waiting_time_ambulance = mean_waiting_time_formula(
            all_states,
            state_probabilities,
            "ambulance",
            lambda_a,
            lambda_o,
            mu,
            num_of_servers,
            threshold,
            system_capacity,
            parking_capacity,
        )
        ambulance_rate = lambda_a / (lambda_a + lambda_o)
        others_rate = lambda_o / (lambda_a + lambda_o)
        return (
            mean_waiting_time_ambulance * ambulance_rate
            + mean_waiting_time_other * others_rate
        )  # TODO: fix this

    mean_waiting_time = mean_waiting_time_formula(
        all_states,
        state_probabilities,
        output,
        lambda_a,
        lambda_o,
        mu,
        num_of_servers,
        threshold,
        system_capacity,
        parking_capacity,
    )
    return mean_waiting_time


def get_plot_comparing_times(
    lambda_a,
    lambda_o,
    mu,
    num_of_servers,
    threshold,
    num_of_trials,
    seed_num,
    runtime,
    system_capacity,
    parking_capacity,
    output="both",
    plot_over="lambda_a",
    max_parameter_value=1,
    accuracy=None,
):
    """Get a plot to compare the simulated waiting times and the markov chain mean waiting times for different values of a given parameter.

    Parameters
    ----------
    lambda_a : float
    lambda_o : float
    mu : float
    num_of_servers : int
    threshold : int
    num_of_trials : int
    seed_num : float
    runtime : int
    system_capacity : int
    parking_capacity : int
    output : str, optional
        A string to identify wheteher to get the waiting time of other patients, ambulance patients or the overall of both, by default "both"
    plot_over : str, optional
        A string with the name of the variable to plot over, by default "lambda_a"
    max_parameter_value : float, optional
        The maximum value of the parameter to plot over, by default 1
    accuracy : int, optional
        The number of iterations between the minimum and maximum number of the parameter, by default None

    Plots
    -------
    matplotlib object
        A plot of the mean waiting time from markov and simualtion state probabiliteis as well as the distributions of the waiting time from the simulation over different values of the given parameter.

    Returns
    -------
    tuple
        The x-axis of the graph
    list
        A list of all mean waiting times of the simulation (from state probabilities)
    list
        A list of all mean waiting times of the markov model
    list
        A list of lists of all mean waiting times of the simulation (simulated) for all trials
    """
    all_times_sim = []
    all_mean_times_sim = []
    all_mean_times_markov = []

    if accuracy == None or accuracy <= 1:
        accuracy = 5

    starting_value = locals()[plot_over]
    range_space = np.linspace(starting_value, max_parameter_value, accuracy)

    for parameter in range_space:
        if plot_over == "lambda_a":
            lambda_a = parameter
        elif plot_over == "lambda_o":
            lambda_o = parameter
        elif plot_over == "mu":
            mu = parameter
        elif plot_over == "num_of_servers":
            num_of_servers = int(parameter)
        elif plot_over == "threshold":
            threshold = int(parameter)
        elif plot_over == "system_capacity":
            system_capacity = int(parameter)
        elif plot_over == "parking_capacity":
            parking_capacity = int(parameter)

        times = get_multiple_runs_results(
            lambda_a,
            lambda_o,
            mu,
            num_of_servers,
            threshold,
            num_of_trials=num_of_trials,
            seed_num=seed_num,
            runtime=runtime,
            system_capacity=system_capacity,
            parking_capacity=parking_capacity,
            patient_type=output,
        )
        simulation_waiting_times = [np.mean(w.waiting_times) for w in times]
        mean_waiting_time_sim = get_mean_waiting_time_from_simulation_state_probabilities(
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
            output=output,
        )
        mean_waiting_time_markov = get_mean_waiting_time_markov(
            lambda_a,
            lambda_o,
            mu,
            num_of_servers,
            threshold,
            system_capacity,
            parking_capacity,
            output=output,
        )

        all_times_sim.append(simulation_waiting_times)
        all_mean_times_sim.append(mean_waiting_time_sim)
        all_mean_times_markov.append(mean_waiting_time_markov)

    diff = (range_space[1] - range_space[0]) / 2
    plt.figure(figsize=(20, 10))
    plt.plot(range_space, all_mean_times_sim, label="Simulation State probabilities")
    plt.plot(range_space, all_mean_times_markov, label="Markov State probabilities")
    plt.violinplot(
        all_times_sim,
        positions=range_space,
        widths=diff,
        showmeans=True,
        showmedians=False,
    )
    title = (
        "lambda_a="
        + str(lambda_a)
        + ", lambda_o="
        + str(lambda_o)
        + ", mu="
        + str(mu)
        + ", C="
        + str(num_of_servers)
        + ", T="
        + str(threshold)
        + ", N="
        + str(system_capacity)
        + ", M="
        + str(parking_capacity)
    )
    plt.title(title)
    plt.xlabel(plot_over)
    plt.ylabel("Waiting time")
    plt.legend()
    return range_space, all_mean_times_sim, all_mean_times_markov, all_times_sim
