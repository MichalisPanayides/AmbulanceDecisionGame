import numpy as np
import csv
import matplotlib.pyplot as plt
import timeit
import pandas as pd
import seaborn as sbr

from .models.simulation import(
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
            ambulance_patients_times.append(ind.data_records[0].waiting_time + ind.data_records[1].waiting_time)
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
            ambulance_patients_times.append(ind.data_records[0].time_blocked + ind.data_records[1].time_blocked)
        elif ind.data_records[0].node == 2 and len(ind.data_records) == 1:
            other_patients_times.append(ind.data_records[0].time_blocked)
        else:
            patients_still_in_system.append(ind)
    return [ambulance_patients_times, other_patients_times, patients_still_in_system]


def get_both_times(individuals):
    """Extracts waiting times and blocking times from results to be used for the plot
    
    Parameters
    ----------
    individuals : [object]
        [An object with all indivduals that enetered the system]
    
    Returns
    -------
    [list, list, list]
        [Three lists that store the waiting and blocking times of patients from the ambulance, other patients and patients still in system]]
    """
    ambulance_patients_times = []
    other_patients_times = []
    patients_still_in_system = []
    
    for ind in individuals:
        if ind.data_records[0].node == 1 and len(ind.data_records) == 2:
            ambulance_patients_times.append(ind.data_records[0].time_blocked + ind.data_records[1].time_blocked + 
                                            ind.data_records[0].waiting_time + ind.data_records[1].waiting_time)
        elif ind.data_records[0].node == 2 and len(ind.data_records) == 1:
            other_patients_times.append(ind.data_records[0].waiting_time + ind.data_records[0].time_blocked)
        else:
            patients_still_in_system.append(ind)
    return [ambulance_patients_times, other_patients_times, patients_still_in_system]


def get_times_for_patients(lambda_a, lambda_o, mu, total_capacity, threshold, seed_num=None, measurement_type=None):
    """Determines the appropriate times to be used set by the user
    
    Parameters
    ----------
    lambda_a : [float]
    lambda_o : [float]
    mu : [float]
    total_capacity : [int]
    threshold : [int]
    seed_num : [float], optional
    measurement_type : [string], optional
    
    Returns
    -------
    [list, list, list]
        [Three lists that store the times of patients from the ambulance, other patients and patients still in system]
    """
    individuals = simulate_model(lambda_a, lambda_o, mu, total_capacity, threshold, seed_num).get_all_individuals()

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
    return(x_axis_label, y_axis_label, title)


def make_plot_for_different_thresholds(lambda_a, lambda_o, mu, total_capacity, num_of_trials, seed_num=None, measurement_type=None):
    """Makes a plot of the mean/waiting time vs different thresholds
    
    Parameters
    ----------
    lambda_a : [float]
    lambda_o : [float]
    mu : [float]
    total_capacity : [int]
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
    for threshold in range(1, total_capacity+1):
        current_ambulance_patients_mean_times = []
        current_other_patients_mean_times = []
        current_total_mean_times = []
        for _ in range(num_of_trials):
            times = get_times_for_patients(lambda_a, lambda_o, mu, total_capacity, threshold, seed_num, measurement_type)
            current_ambulance_patients_mean_times.append(np.mean(times[0]))
            current_other_patients_mean_times.append(np.mean(times[1]))
            current_total_mean_times.append(np.mean(times[0] + times[1]))
        all_ambulance_patients_mean_times.append(np.mean(current_ambulance_patients_mean_times))
        all_other_patients_mean_times.append(np.mean(current_other_patients_mean_times))
        all_total_mean_times.append(np.mean(current_total_mean_times))

    x_axis = [thres for thres in range(1, total_capacity + 1)]
    x_axis_label, y_axis_label, title = get_plot_for_different_thresholds_labels(measurement_type)
    plt.figure(figsize=(23,10))
    diff_threshold_plot = plt.plot(x_axis, all_ambulance_patients_mean_times, ':', x_axis, all_other_patients_mean_times, ':', x_axis, all_total_mean_times, '-')
    plt.title(title)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.legend(['Ambulance Patients', 'Other Patients', 'All times'])

    return diff_threshold_plot


# Plot 2: Proportion of people within target
def make_proportion_plot(lambda_a, lambda_o, mu, total_capacity, num_of_trials, seed_num, target):
    all_proportions = []
    for threshold in range(total_capacity + 1):
        current_proportions = []
        for _ in range(num_of_trials):
            records = simulate_model(lambda_a, lambda_o, mu, total_capacity, threshold, seed_num).get_all_records()
            waits, target_waits = 0, 0
            for rec in records:
                if rec.node == 2:
                    waits += 1
                    if rec.waiting_time <= target:
                        target_waits += 1
            current_proportions.append(target_waits / waits)
        all_proportions.append(np.mean(current_proportions))

    plt.figure(figsize=(23,10))
    proportion_plot = plt.plot(all_proportions)
    plt.title("Proportion of individuals within target for different capacity thresholds")
    plt.xlabel("Capacity Threshold")
    plt.ylabel("Proportion of Individuals within target")

    return proportion_plot


# Plot 3: Arrival rate vs waiting/blocking time between two Hospitals

def update_hospitals_lists(hospital_times_1, hospital_times_2, times_1, times_2, measurement_type):
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
        title = "Blocking times of two hospitals over different distribution of patients"
        y_axis_label = "Blocking Time"
    x_axis_label = "Hospital 1 arrival proportion"
    return(x_axis_label, y_axis_label, title)

def make_plot_two_hospitals_arrival_split(lambda_a, lambda_o_1, lambda_o_2, mu_1, mu_2, total_capacity_1, total_capacity_2, threshold_1, threshold_2, measurement_type="b", seed_num_1=None, seed_num_2=None, warm_up_time=100, trials=1, accuracy=10):
    """Make a plot of the waiting/blocking time between two hospitals that have a joint arrival rate of ambulance patients. In other words plots the waiting/blocking times of patients based on how the ambulance patients are distributed among hospitals
    
    Parameters
    ----------
    lambda_a : [float]
    lambda_o_1 : [float]
    lambda_o_2 : [float]
    mu_1 : [float]
    mu_2 : [float]
    total_capacity_1 : [int]
    total_capacity_2 : [int]
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
        times_1 = get_multiple_runs_results(arrival_rate_1, lambda_o_1, mu_1, total_capacity_1, threshold_1, seed_num_1, warm_up_time, trials)
        times_2 = get_multiple_runs_results(arrival_rate_2, lambda_o_2, mu_2, total_capacity_2, threshold_2, seed_num_2, warm_up_time, trials)
        hospital_times_1, hospital_times_2 = update_hospitals_lists(hospital_times_1, hospital_times_2, times_1, times_2, measurement_type)
    

    x_axis_label, y_axis_label, title = get_two_hospital_plot_labels(measurement_type)
    x_labels = all_arrival_rates[1:-1] / all_arrival_rates[-1]
    plt.figure(figsize=(23,10))
    waiting_time_plot = plt.plot(x_labels, hospital_times_1, ':')
    plt.plot(x_labels, hospital_times_2,  ':')
    plt.legend(['Hospital 1', 'Hospital 2'])
    plt.title(title)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)

    return waiting_time_plot

