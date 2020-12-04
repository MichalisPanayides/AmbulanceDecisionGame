import matplotlib.pyplot as plt
import numpy as np

from .markov.blocking import (
    get_mean_blocking_time_using_markov_state_probabilities,
    mean_blocking_time_formula_using_direct_approach,
)
from .markov.markov import (
    build_states,
    get_markov_state_probabilities,
    get_steady_state_algebraically,
    get_transition_matrix,
)
from .markov.utils import is_accepting_state
from .markov.waiting import (
    get_mean_waiting_time_using_markov_state_probabilities,
    mean_waiting_time_formula_using_closed_form_approach,
    mean_waiting_time_formula_using_direct_approach,
    mean_waiting_time_formula_using_recursive_approach,
    overall_waiting_time_formula,
)
from .simulation.simulation import (
    get_average_simulated_state_probabilities,
    get_multiple_runs_results,
    get_simulated_state_probabilities,
    simulate_model,
)


def get_heatmaps(
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
    linear_positioning=False,
):
    """Get heatmaps plot that compare the state probabilities of the simulation
    and Markov state probabilities. In total three heatmaps are generated; one for
    the simulation state probabilities, one for the Markov state probabilities and
    one for the difference between the two.

    Parameters
    ----------
    lambda_2 : float
    lambda_1 : float
    mu : float
    num_of_servers : int
    threshold : int
    system_capacity : int
    buffer_capacity : int
    seed_num : float, optional
    runtime : int, optional
    num_of_trials : int, optional
    linear_positioning : Boolean, optional
        To distinguish between the two position formats of the heatmaps, by default False
    """
    all_states = build_states(
        threshold=threshold,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
    )
    transition_matrix = get_transition_matrix(
        lambda_2=lambda_2,
        lambda_1=lambda_1,
        mu=mu,
        num_of_servers=num_of_servers,
        threshold=threshold,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
    )
    pi = get_steady_state_algebraically(
        Q=transition_matrix, algebraic_function=np.linalg.lstsq
    )

    sim_state_probabilities_array = get_average_simulated_state_probabilities(
        lambda_2=lambda_2,
        lambda_1=lambda_1,
        mu=mu,
        num_of_servers=num_of_servers,
        threshold=threshold,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
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
        buffer_capacity=buffer_capacity,
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
    plt.title("Simulation state probabilities", fontsize=11, fontweight="bold")
    plt.xlabel("Individuals in service area", fontsize=11, fontweight="bold")
    plt.ylabel("Individuals in buffer centre", fontsize=11, fontweight="bold")
    plt.colorbar()

    if not linear_positioning:
        plt.subplot(grid[0, 2:4])
    else:
        plt.subplot(1, 3, 2)

    plt.imshow(markov_state_probabilities_array, cmap="cividis")
    plt.title("Markov chain state probabilities", fontsize=11, fontweight="bold")
    plt.xlabel("Individuals in service area", fontsize=11, fontweight="bold")
    plt.ylabel("Individuals in buffer centre", fontsize=11, fontweight="bold")
    plt.colorbar()

    if not linear_positioning:
        plt.subplot(grid[1, 1:3])
    else:
        plt.subplot(1, 3, 3)
    plt.imshow(diff_states_probabilities_array, cmap="viridis")
    plt.title(
        "Simulation and Markov chain state probability differences",
        fontsize=11,
        fontweight="bold",
    )
    plt.xlabel("Individuals in service area", fontsize=11, fontweight="bold")
    plt.ylabel("Individuals in buffer centre", fontsize=11, fontweight="bold")
    plt.colorbar()

    return (
        sim_state_probabilities_array,
        markov_state_probabilities_array,
        diff_states_probabilities_array,
    )


def get_mean_waiting_time_from_simulation_state_probabilities(
    lambda_2,
    lambda_1,
    mu,
    num_of_servers,
    threshold,
    system_capacity,
    buffer_capacity,
    seed_num,
    runtime=1440,
    num_of_trials=10,
    class_type=None,
    waiting_formula=mean_waiting_time_formula_using_closed_form_approach,
):
    """An alternative approach to obtaining the mean waiting time from the simulation.
    This function gets the mean waiting time from the simulation state probabilities.
    This is mainly used in comparing the simulation results with the Markov ones.

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
    class_type : int, optional
        Takes values (0, 1, None) to identify whether to get the waiting time of
        class 1 individuals, class 2 individuals or the overall of both,
        by default None

    Returns
    -------
    float
        The waiting time in the system of the given class
    """
    state_probabilities = get_average_simulated_state_probabilities(
        lambda_2=lambda_2,
        lambda_1=lambda_1,
        mu=mu,
        num_of_servers=num_of_servers,
        threshold=threshold,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
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

    if class_type is None:
        get_mean_waiting_time = overall_waiting_time_formula
    else:
        get_mean_waiting_time = waiting_formula

    mean_waiting_time = get_mean_waiting_time(
        all_states=all_states,
        pi=state_probabilities,
        class_type=class_type,
        lambda_2=lambda_2,
        lambda_1=lambda_1,
        mu=mu,
        num_of_servers=num_of_servers,
        threshold=threshold,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
        waiting_formula=waiting_formula,
    )

    return mean_waiting_time


def get_mean_blocking_time_from_simulation_state_probabilities(
    lambda_2,
    lambda_1,
    mu,
    num_of_servers,
    threshold,
    system_capacity,
    buffer_capacity,
    seed_num=None,
    num_of_trials=10,
    runtime=2000,
    blocking_formula=mean_blocking_time_formula_using_direct_approach,
):
    """An alternative approach to obtaining the mean blocking time from the simulation.
    This function gets the mean blocking time from the simulation's state probabilities.
    This is mainly used in comparing the simulation results with the Markov ones.

    Parameters
    ----------
    lambda_2 : float
    lambda_1 : float
    mu : float
    num_of_servers : int
    threshold : int
    system_capacity : int
    buffer_capacity : int
    seed_num : float, optional
    num_of_trials : int, optional
    runtime : int, optional

    Returns
    -------
    float
        The mean blocking time
    """
    state_probabilities = get_average_simulated_state_probabilities(
        lambda_2=lambda_2,
        lambda_1=lambda_1,
        mu=mu,
        num_of_servers=num_of_servers,
        threshold=threshold,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
        seed_num=seed_num,
        num_of_trials=num_of_trials,
        runtime=runtime,
    )
    all_states = [
        (u, v)
        for v in range(state_probabilities.shape[1])
        for u in range(state_probabilities.shape[0])
        if state_probabilities[u, v] > 0
    ]
    mean_blocking_time = blocking_formula(
        all_states=all_states,
        pi=state_probabilities,
        lambda_1=lambda_1,
        mu=mu,
        num_of_servers=num_of_servers,
        threshold=threshold,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
    )
    return mean_blocking_time


def get_plot_comparing_times(
    lambda_2,
    lambda_1,
    mu,
    num_of_servers,
    threshold,
    num_of_trials,
    seed_num,
    runtime,
    system_capacity,
    buffer_capacity,
    times_to_compare,
    warm_up_time=0,
    class_type=None,
    plot_over="lambda_2",
    max_parameter_value=1,
    accuracy=None,
):
    """Get a plot to compare the simulated waiting or blocking times and the Markov
    chain mean waiting or blocking times for different values of a given parameter.

    Parameters
    ----------
    lambda_2 : float
    lambda_1 : float
    mu : float
    num_of_servers : int
    threshold : int
    num_of_trials : int
    seed_num : float
    runtime : int
    system_capacity : int
    buffer_capacity : int
    times_to_compare : str
    class_type : int, optional
        Takes values (0, 1, None) to identify whether to get the waiting time of
        class 1 individuals, class 2 individuals or the overall of both,
        by default None
    plot_over : str, optional
        A string with the name of the variable to plot over, by default "lambda_2"
    max_parameter_value : float, optional
        The maximum value of the parameter to plot over, by default 1
    accuracy : int, optional
        The number of iterations between the minimum and maximum number of the parameter, by default None

    Plots
    -------
    matplotlib object
        A plot of the mean waiting time from Markov and simulation state probabilities
        as well as the distributions of the waiting time from the simulation over
        different values of the given parameter.

    Returns
    -------
    tuple
        The x-axis of the graph
    list
        A list of all mean waiting times of the simulation (from state probabilities)
    list
        A list of all mean waiting times of the Markov model
    list
        A list of lists of all mean waiting times of the simulation (simulated)
        for all trials
    """
    all_times_sim = []
    all_mean_times_sim = []
    all_mean_times_markov = []
    if accuracy == None or accuracy <= 1:
        accuracy = 5

    starting_value = locals()[plot_over]
    range_space = np.linspace(starting_value, max_parameter_value, accuracy)

    for parameter in range_space:
        if plot_over == "lambda_2":
            lambda_2 = parameter
        elif plot_over == "lambda_1":
            lambda_1 = parameter
        elif plot_over == "mu":
            mu = parameter
        elif plot_over == "num_of_servers":
            num_of_servers = int(parameter)
        elif plot_over == "threshold":
            threshold = int(parameter)
        elif plot_over == "system_capacity":
            system_capacity = int(parameter)
        elif plot_over == "buffer_capacity":
            buffer_capacity = int(parameter)

        times = get_multiple_runs_results(
            lambda_2=lambda_2,
            lambda_1=lambda_1,
            mu=mu,
            num_of_servers=num_of_servers,
            threshold=threshold,
            num_of_trials=num_of_trials,
            seed_num=seed_num,
            runtime=runtime,
            warm_up_time=warm_up_time,
            system_capacity=system_capacity,
            buffer_capacity=buffer_capacity,
            class_type=class_type,
        )
        # TODO: Get rid of times_to_compare variable
        if times_to_compare == "waiting":
            simulation_times = [np.mean(w.waiting_times) for w in times]
            mean_time_sim = get_mean_waiting_time_from_simulation_state_probabilities(
                lambda_2=lambda_2,
                lambda_1=lambda_1,
                mu=mu,
                num_of_servers=num_of_servers,
                threshold=threshold,
                system_capacity=system_capacity,
                buffer_capacity=buffer_capacity,
                seed_num=seed_num,
                runtime=runtime,
                num_of_trials=num_of_trials,
                class_type=class_type,
            )
            mean_time_markov = get_mean_waiting_time_using_markov_state_probabilities(
                lambda_2=lambda_2,
                lambda_1=lambda_1,
                mu=mu,
                num_of_servers=num_of_servers,
                threshold=threshold,
                system_capacity=system_capacity,
                buffer_capacity=buffer_capacity,
                class_type=class_type,
            )
        elif times_to_compare == "blocking":
            if class_type == 0:
                raise Exception("Blocking does not occur for class 1 individuals")
            simulation_times = [np.mean(b.blocking_times) for b in times]
            mean_time_sim = get_mean_blocking_time_from_simulation_state_probabilities(
                lambda_2=lambda_2,
                lambda_1=lambda_1,
                mu=mu,
                num_of_servers=num_of_servers,
                threshold=threshold,
                system_capacity=system_capacity,
                buffer_capacity=buffer_capacity,
                num_of_trials=num_of_trials,
                seed_num=seed_num,
                runtime=runtime,
            )
            mean_time_markov = get_mean_blocking_time_using_markov_state_probabilities(
                lambda_2=lambda_2,
                lambda_1=lambda_1,
                mu=mu,
                num_of_servers=num_of_servers,
                threshold=threshold,
                system_capacity=system_capacity,
                buffer_capacity=buffer_capacity,
            )

        all_times_sim.append(simulation_times)
        all_mean_times_sim.append(mean_time_sim)
        all_mean_times_markov.append(mean_time_markov)

    diff = (range_space[1] - range_space[0]) / 2
    plt.figure(figsize=(20, 10))
    plt.plot(
        range_space,
        all_mean_times_sim,
        label="Simulation State probabilities",
        ls="solid",
        lw=1.5,
    )
    plt.plot(
        range_space,
        all_mean_times_markov,
        label="Markov State probabilities",
        ls="solid",
        lw=1.5,
    )
    plt.violinplot(
        all_times_sim,
        positions=range_space,
        widths=diff,
        showmeans=True,
        showmedians=False,
    )
    title = (
        r"$\lambda_2=$"
        + str(lambda_2)
        + r", $\lambda_1=$"
        + str(lambda_1)
        + r", $\mu=$"
        + str(mu)
        + ", C="
        + str(num_of_servers)
        + ", T="
        + str(threshold)
        + ", N="
        + str(system_capacity)
        + ", M="
        + str(buffer_capacity)
    )
    plt.title(title, fontsize=18)
    plt.xlabel(plot_over, fontsize=15, fontweight="bold")
    plt.ylabel("Waiting time", fontsize=15, fontweight="bold")
    plt.legend()
    return range_space, all_mean_times_sim, all_mean_times_markov, all_times_sim
