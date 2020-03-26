import numpy as np
import csv
import matplotlib.pyplot as plt
import timeit
import pandas as pd
import seaborn as sbr
import random

from .models.simulation import (
    simulate_model,
    get_multiple_runs_results,
)

# Plot 1: Mean waiting Time vs Threshold

def get_waiting_times(individuals):
    """Extracts waiting times from results to be used for the plot
    
    Parameters
    ----------
    individuals : [object]
        [An object with all indivduals that enetered the system]
    
    Returns
    -------
    [list, list, list]
        [Three lists that store the waiting times of patients from the ambulance, other patients and patients still in system]]
    """
    ambulance_patients_times = []
    other_patients_times = []
    patients_still_in_system = []

    for ind in individuals:
        if ind.data_records[0].node == 1 and len(ind.data_records) == 2:
            ambulance_patients_times.append(
                ind.data_records[0].waiting_time + ind.data_records[1].waiting_time
            )
        elif ind.data_records[0].node == 2 and len(ind.data_records) == 1:
            other_patients_times.append(ind.data_records[0].waiting_time)
        else:
            patients_still_in_system.append(ind)
    return [ambulance_patients_times, other_patients_times, patients_still_in_system]


def get_blocking_times(individuals):
    """Extracts blocking times from results to be used for the plot
    
    Parameters
    ----------
    individuals : [object]
        [An object with all indivduals that enetered the system]
    
    Returns
    -------
    [list, list, list]
        [Three lists that store the blocking times of patients from the ambulance, other patients and patients still in system]]
    """
    ambulance_patients_times = []
    other_patients_times = []
    patients_still_in_system = []

    for ind in individuals:
        if ind.data_records[0].node == 1 and len(ind.data_records) == 2:
            ambulance_patients_times.append(
                ind.data_records[0].time_blocked + ind.data_records[1].time_blocked
            )
        elif ind.data_records[0].node == 2 and len(ind.data_records) == 1:
            other_patients_times.append(ind.data_records[0].time_blocked)
        else:
            patients_still_in_system.append(ind)
    return [ambulance_patients_times, other_patients_times, patients_still_in_system]


def get_both_times(individuals):
    """Extracts waiting times and blocking times from results to be used for the plot
    
    Parameters
    ----------
    individuals : object
        An object with all indivduals that enetered the system
    
    Returns
    -------
    list, list, list
        Three lists that store the waiting and blocking times of patients from the ambulance, other patients and patients still in system
    """
    ambulance_patients_times = []
    other_patients_times = []
    patients_still_in_system = []

    for ind in individuals:
        if ind.data_records[0].node == 1 and len(ind.data_records) == 2:
            ambulance_patients_times.append(
                ind.data_records[0].time_blocked
                + ind.data_records[1].time_blocked
                + ind.data_records[0].waiting_time
                + ind.data_records[1].waiting_time
            )
        elif ind.data_records[0].node == 2 and len(ind.data_records) == 1:
            other_patients_times.append(
                ind.data_records[0].waiting_time + ind.data_records[0].time_blocked
            )
        else:
            patients_still_in_system.append(ind)
    return [ambulance_patients_times, other_patients_times, patients_still_in_system]


def get_times_for_patients(
    lambda_a,
    lambda_o,
    mu,
    num_of_servers,
    threshold,
    seed_num,
    measurement_type,
    runtime,
):
    """Determines the appropriate times to be used set by the user
    
    Parameters
    ----------
    lambda_a : [float]
    lambda_o : [float]
    mu : [float]
    num_of_servers : [int]
    threshold : [int]
    seed_num : [float]
    measurement_type : [string]
    runtime: [float]
    
    Returns
    -------
    [list, list, list]
        [Three lists that store the times of patients from the ambulance, other patients and patients still in system]
    """
    individuals = simulate_model(
        lambda_a, lambda_o, mu, num_of_servers, threshold, seed_num, runtime
    ).get_all_individuals()

    if measurement_type == "w":
        times = get_waiting_times(individuals)
    elif measurement_type == "b":
        times = get_blocking_times(individuals)
    else:
        times = get_both_times(individuals)

    return [times[0], times[1], times[2]]


def get_plot_for_different_thresholds_labels(measurement_type):
    """A function to get necessary labels for the waiting times of different thresholds
    """
    if measurement_type == "w":
        title = "Waiting times over different thresholds"
        y_axis_label = "Waiting Time"
    elif measurement_type == "b":
        title = "Blocking times over different thresholds"
        y_axis_label = "Blocking Time"
    else:
        title = "Waiting and blocking times over different thresholds"
        y_axis_label = "Waiting and Blocking Time"

    x_axis_label = "Capacity Threshold"
    return (x_axis_label, y_axis_label, title)


def make_plot_for_different_thresholds(
    lambda_a,
    lambda_o,
    mu,
    num_of_servers,
    num_of_trials,
    seed_num=None,
    measurement_type=None,
    runtime=1440,
):
    """Makes a plot of the mean/waiting time vs different thresholds
    
    Parameters
    ----------
    lambda_a : [float]
    lambda_o : [float]
    mu : [float]
    num_of_servers : [int]
    seed_num : [float], optional
        [The ciw.seed value to be used by ciw], by default None
    measurement_type : [string], optional
        [Defines whether to use blocking, waiting time or both], by default None
    plot_function : [function], optional
        [The function to be used for the plot i.e either plot of the means or sums of times], by default np.mean
    
    Returns
    -------
    [matplotlib object]
        [The plot of mean waiting/blocking time for different thresholds]
    """
    all_ambulance_patients_mean_times = []
    all_other_patients_mean_times = []
    all_total_mean_times = []
    for threshold in range(1, num_of_servers + 1):
        current_ambulance_patients_mean_times = []
        current_other_patients_mean_times = []
        current_total_mean_times = []
        for _ in range(num_of_trials):
            times = get_times_for_patients(
                lambda_a,
                lambda_o,
                mu,
                num_of_servers,
                threshold,
                seed_num,
                measurement_type,
                runtime,
            )
            current_ambulance_patients_mean_times.append(np.mean(times[0]))
            current_other_patients_mean_times.append(np.mean(times[1]))
            current_total_mean_times.append(np.mean(times[0] + times[1]))
        all_ambulance_patients_mean_times.append(
            np.mean(current_ambulance_patients_mean_times)
        )
        all_other_patients_mean_times.append(np.mean(current_other_patients_mean_times))
        all_total_mean_times.append(np.mean(current_total_mean_times))

    x_axis = [thres for thres in range(1, num_of_servers + 1)]
    x_axis_label, y_axis_label, title = get_plot_for_different_thresholds_labels(
        measurement_type
    )
    plt.figure(figsize=(23, 10))
    diff_threshold_plot = plt.plot(
        x_axis,
        all_ambulance_patients_mean_times,
        ":",
        x_axis,
        all_other_patients_mean_times,
        ":",
        x_axis,
        all_total_mean_times,
        "-",
    )
    plt.title(title)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.legend(
        ["Ambulance Patients", "Other Patients", "All times"], fontsize="x-large"
    )

    return diff_threshold_plot


# Plot 2: Proportion of people within target

def get_target_proportions_of_current_trial(individuals, target):
    """Get the proportion waiting times within the target for a given trial of a threshold
    
    Parameters
    ----------
    individuals : object
        A ciw object that contains all individuals records
    
    Returns
    -------
    int 
        all ambulance patients that finished the simulation
    int
        all ambulance patients whose waiting times where within the target
    int
        all other patients that finished the simulation
    int
        all other patients whose waiting times where within the target
    """
    ambulance_waits, ambulance_target_waits = 0, 0
    other_waits, other_target_waits = 0, 0
    for individual in individuals:
        ind_class = len(individual.data_records) - 1
        rec = individual.data_records[-1]
        if rec.node == 2 and ind_class == 0:
            other_waits += 1
            if rec.waiting_time < target:
                other_target_waits += 1
        elif rec.node == 2 and ind_class == 1:
            ambulance_waits += 1
            if rec.waiting_time < target:
                ambulance_target_waits += 1

    return ambulance_waits, ambulance_target_waits, other_waits, other_target_waits


def get_mean_waits_of_current_threshold(
    lambda_a, lambda_o, mu, num_of_servers, threshold, seed_num, num_of_trials, runtime, target
):
    """Calculates the mean proportion of times that satisfy the target of all trials for the current threshold iteration
    
    Returns
    -------
    float, float, float
        The mean waiting times for ambulance patients, other patients and all patients for a given threshold
    """
    current_ambulance_proportions = []
    current_other_proportions = []
    current_combined_proportions = []

    if seed_num == None:
        seed_num = random.random()

    for trial in range(num_of_trials):
        individuals = simulate_model(
            lambda_a, lambda_o, mu, num_of_servers, threshold, seed_num + trial, runtime
        ).get_all_individuals()
        (
            ambulance_waits,
            ambulance_target_waits,
            other_waits,
            other_target_waits,
        ) = get_target_proportions_of_current_trial(individuals, target)

        current_ambulance_proportions.append(
            (ambulance_target_waits / ambulance_waits) if ambulance_waits != 0 else 1
        )
        current_other_proportions.append(
            (other_target_waits / other_waits) if other_waits != 0 else 1
        )
        current_combined_proportions.append(
            (ambulance_target_waits + other_target_waits)
            / (ambulance_waits + other_waits)
            if (ambulance_waits + other_waits) != 0
            else 1
        )

    return (
        np.mean(current_ambulance_proportions),
        np.mean(current_other_proportions),
        np.mean(current_combined_proportions),
    )


def make_proportion_plot(
    lambda_a,
    lambda_o,
    mu,
    num_of_servers,
    num_of_trials,
    seed_num,
    target,
    runtime=1440,
):
    """Builds a plot that shows the proportion of individuals that satisfy the desired waiting time target. The plot shows the proportions of ambulance patients, other patients and the combined proportion of the two, that satisfy the target.
    
    Parameters
    ----------
    num_of_trials : int
        The number of trials to run the simulation to average out uncertainty
    target : int
        The target time to compare the waiting times with (Proprtion found based on this target)
    runtime : int, optional
        The runtime to run the simulation, by default 1440
    
    Returns
    -------
    matplotlib object
        Plot of proportions within target for ambulance, others and all patients
    """
    ambulance_proportions = []
    other_proportions = []
    all_proportions = []
    for threshold in range(num_of_servers + 1):
        mean_ambulance, mean_other, mean_combined = get_mean_waits_of_current_threshold(
            lambda_a,
            lambda_o,
            mu,
            num_of_servers,
            threshold,
            seed_num,
            num_of_trials,
            runtime,
            target,
        )
        ambulance_proportions.append(mean_ambulance)
        other_proportions.append(mean_other)
        all_proportions.append(mean_combined)

    plt.figure(figsize=(23, 10))
    proportion_plot = plt.plot(
        ambulance_proportions, ":", other_proportions, ":", all_proportions, "-"
    )
    plt.title(
        "Proportion of individuals within target for different capacity thresholds"
    )
    plt.xlabel("Capacity Threshold")
    plt.ylabel("Proportion of Individuals within target")
    plt.legend(
        ["Ambulance Patients", "Other Patient", "All Patients"], fontsize="x-large"
    )

    return proportion_plot


# Plot 3: Arrival rate vs waiting/blocking time between two Hospitals

def update_hospitals_lists(
    hospital_times_1, hospital_times_2, times_1, times_2, measurement_type
):
    """Update the two lists that are going to be used for plotting
    
    Parameters
    ----------
    hospital_times_1 : [list]
        [Times of the first hospital that we want to update]
    hospital_times_2 : [list]
        [Times of the second hospital that we want to update]
    times_1 : [list]
        [A list of named tuples that holds the records of hospital 1]
    times_2 : [list]
        [A list of named tuples that holds the records of hospital 2]
    measurement_type : [string]
    
    Returns
    -------
    [list, list]
        [description]
    """
    if measurement_type == "w":
        hospital_times_1.append(np.mean([np.mean(w.waiting_times) for w in times_1]))
        hospital_times_2.append(np.mean([np.mean(w.waiting_times) for w in times_2]))
    else:
        hospital_times_1.append(np.mean([np.mean(b.blocking_times) for b in times_1]))
        hospital_times_2.append(np.mean([np.mean(b.blocking_times) for b in times_2]))
    return hospital_times_1, hospital_times_2


def get_two_hospital_plot_labels(measurement_type):
    """A function to get necessary labels for the two hospitals plot
    """
    if measurement_type == "w":
        title = "Waiting times of two hospitals over different distribution of patients"
        y_axis_label = "Waiting Time"
    else:
        title = (
            "Blocking times of two hospitals over different distribution of patients"
        )
        y_axis_label = "Blocking Time"
    x_axis_label = "Hospital 1 arrival proportion"
    return (x_axis_label, y_axis_label, title)


def make_plot_two_hospitals_arrival_split(
    lambda_a,
    lambda_o_1,
    lambda_o_2,
    mu_1,
    mu_2,
    num_of_servers_1,
    num_of_servers_2,
    threshold_1,
    threshold_2,
    measurement_type="b",
    seed_num_1=None,
    seed_num_2=None,
    warm_up_time=100,
    trials=1,
    accuracy=10,
    runtime=1440,
):
    """Make a plot of the waiting/blocking time between two hospitals that have a joint arrival rate of ambulance patients. In other words plots the waiting/blocking times of patients based on how the ambulance patients are distributed among hospitals
    
    Parameters
    ----------
    lambda_a : [float]
    lambda_o_1 : [float]
    lambda_o_2 : [float]
    mu_1 : [float]
    mu_2 : [float]
    num_of_servers_1 : [int]
    num_of_servers_2 : [int]
    threshold_1 : [int]
    threshold_2 : [int]
    measurement_type : [string], optional, by default "b"
    seed_num_1 : [float], optional, by default None
    seed_num_2 : [float], optional, by default None
    warm_up_time : int, optional
    trials : int, optional
        [The number of trials to get results from], by default 1
    
    Returns
    -------
    [matplotlib object]
        [proportion of arrivals to hospital 1 vs waiting times for both hospitals]
    """
    hospital_times_1 = []
    hospital_times_2 = []
    all_arrival_rates = np.linspace(0, lambda_a, accuracy + 1)
    for arrival_rate_1 in all_arrival_rates[1:-1]:
        arrival_rate_2 = lambda_a - arrival_rate_1
        times_1 = get_multiple_runs_results(
            arrival_rate_1,
            lambda_o_1,
            mu_1,
            num_of_servers_1,
            threshold_1,
            seed_num_1,
            warm_up_time,
            trials,
            runtime,
        )
        times_2 = get_multiple_runs_results(
            arrival_rate_2,
            lambda_o_2,
            mu_2,
            num_of_servers_2,
            threshold_2,
            seed_num_2,
            warm_up_time,
            trials,
            runtime,
        )
        hospital_times_1, hospital_times_2 = update_hospitals_lists(
            hospital_times_1, hospital_times_2, times_1, times_2, measurement_type
        )

    x_axis_label, y_axis_label, title = get_two_hospital_plot_labels(measurement_type)
    x_labels = all_arrival_rates[1:-1] / all_arrival_rates[-1]
    plt.figure(figsize=(23, 10))
    waiting_time_plot = plt.plot(x_labels, hospital_times_1, ":")
    plt.plot(x_labels, hospital_times_2, ":")
    plt.legend(["Hospital 1", "Hospital 2"], fontsize="x-large")
    plt.title(title)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)

    return waiting_time_plot


# Plot 4: Waiting/Blocking time distribution VS warm-up time

def get_times_and_labels(records, measurement_type):
    """Identifies the required times (waiting or blocking) and plot lebels (Function is used in Plot 5 as well)
    
    Parameters
    ----------
    records : list
        A list of named tuples that contains the results of multiple runs of the simulation
    measurement_type : string
        A string to distinguish between times to be used
    
    Returns
    -------
    list
        The mean waiting/blocking times of each trial
    string
        plot title
    string
        y-axis label
    """
    if measurement_type == "w":
        mean_time = [np.mean(w.waiting_times) for w in records]
        title = "Distributions of waiting times over runtimes"
        y_axis_label = "Waiting Times"
    else:
        mean_time = [np.mean(b.blocking_times) for b in records]
        title = "Distributions of blocking times over runtimes"
        y_axis_label = "Blocking Times"
    return mean_time, title, y_axis_label


def make_plot_of_confidence_intervals_over_warm_up_time(
    lambda_a,
    lambda_o,
    mu,
    num_of_servers,
    threshold,
    num_of_trials,
    min_w=0,
    max_w=500,
    seed_num=None,
    measurement_type=None,
    runtime=1440,
):
    """Make a plot of the distributions of times (waiting or blocking) over values of warm-up times
    
    Parameters
    ----------
    min_w : int, optional
        The minimum value of warm-up time of the range to be included, by default 720(12 hours)
    max_w : int, optional
        The maximum value of warm-up time of the range to be included, by default 2880(2 Days)
    measurement_type : string, optional
        A string to distinguish between times to be plotted, by default None
    
    Returns
    -------
    matplotlib object
        A plot of the distributions of waiting/blocking times for different values of the Simulation's runtime
    """
    mean_time = []
    x_axis = []
    warm_up_range = np.linspace(min_w, max_w, 20)
    for warm_up_time in warm_up_range:
        res = get_multiple_runs_results(
            lambda_a,
            lambda_o,
            mu,
            num_of_servers,
            threshold,
            seed_num,
            warm_up_time,
            num_of_trials,
            runtime,
        )
        current_mean_time, title, y_axis_label = get_times_and_labels(
            res, measurement_type
        )
        mean_time.append(current_mean_time)
        x_axis.append(round(warm_up_time))

    plt.figure(figsize=(23, 10))
    plot = plt.boxplot(mean_time, labels=x_axis, showfliers=False)
    plt.title(title)
    plt.xlabel("Warm-up time")
    plt.ylabel(y_axis_label)

    return plot


# Plot 5: Waiting/Blocking time confidence interavls VS runtime

def make_plot_of_confidence_intervals_over_runtime(
    lambda_a,
    lambda_o,
    mu,
    num_of_servers,
    threshold,
    warm_up_time=100,
    num_of_trials=10,
    min_r=720,
    max_r=2880,
    seed_num=None,
    measurement_type=None,
):
    """Make a plot of the distributions of times (waiting or blocking) over values of runtime
    
    Parameters
    ----------
    min_r : int, optional
        The minimum value of runtime of the range to be included, by default 720(12 hours)
    max_r : int, optional
        The maximum value of runtime of the range to be included, by default 2880(2 Days)
    measurement_type : string, optional
        A string to distinguish between times to be plotted, by default None
    
    Returns
    -------
    matplotlib object
        A plot of the distributions of waiting/blocking times for different values of the Simulation's runtime
    """
    mean_time = []
    x_axis = []
    runtime_range = np.linspace(min_r, max_r, 20)
    for runtime in runtime_range:
        res = get_multiple_runs_results(
            lambda_a,
            lambda_o,
            mu,
            num_of_servers,
            threshold,
            seed_num,
            warm_up_time,
            num_of_trials,
            runtime,
        )
        current_mean_time, title, y_axis_label = get_times_and_labels(
            res, measurement_type
        )
        mean_time.append(current_mean_time)
        x_axis.append(round(runtime))

    plt.figure(figsize=(23, 10))
    plot = plt.boxplot(mean_time, labels=x_axis, showfliers=False)
    plt.title(title)
    plt.xlabel("Simulation runtime")
    plt.ylabel(y_axis_label)

    return plot
