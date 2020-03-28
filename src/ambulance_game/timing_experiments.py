import numpy as np
import csv
import matplotlib.pyplot as plt
import timeit
import pandas as pd
import seaborn as sbr


###############################
### First Timing Experiment ###
###############################


def get_trial_duration(
    lambda_a, lambda_o, mu, num_of_servers, threshold, trials, repeat_trial
):
    """Use the timeit library to get the duration of a trial for a specified amount of repetitions
    
    Parameters
    ----------
    trials : [int]
        [Number of trials to run timeit on]
    repeat_trial : [int]
        [Number of repetitions per trial to run timeit on]
    
    Returns
    -------
    [list]
        [A list of floats that represent the time required for the specified number of trials to for each repetition]
    """
    parameters = [lambda_a, lambda_o, mu, num_of_servers, threshold]
    timeit_string = "simulate_model("
    for par in range(len(parameters)):
        timeit_string += str(parameters[par]) + ","
    timeit_string += ")"
    duration_object = timeit.Timer(
        timeit_string, "from ambulance_game.models.simulation import simulate_model"
    )
    duration_of_trial = duration_object.repeat(repeat=repeat_trial, number=trials)
    return duration_of_trial


def save_to_file(duration, filename):
    """Takes the repetitions of durations of one trial and saves them to a file
    
    Parameters
    ----------
    duration : [list of floats]
        [A list of value to be saved to a file]
    filename : str, optional
        [The filename where the files will be saved], by default "Custom_Trials.csv"
    """
    with open(filename, "a", newline="") as times_file:
        csv_object = csv.writer(times_file)
        csv_object.writerow(duration)


def time_for_different_number_of_trials(
    lambda_a,
    lambda_o,
    mu,
    num_of_servers,
    threshold,
    num_of_trials,
    repeat_trial=1,
    method="Simulation",
    filename=None,
):
    """A function to claculate the duration of the simualtion for given queueing parameters, number of trials and repetitions (to be edited to accepet an analytical solution as well instead of simulation)
    
    Parameters
    ----------
    num_of_trials : [int]
        [The number of trials to run the duration on]
    repeat_trial : int, optional
        [A number to indicate how many repetitions to make for each trial], by default 1
    method : str, optional
        [An arguement to identify which approach to use (currently only supports 'Simulation')], by default 'Simulation'
    filename : str, optional
        [A string that contains the filename that the recorded times will be saved to], dy default None

    Returns
    -------
    [list]
        [A list that includes all durations and repetitions]
    """
    if method == "Simulation":
        write_to_file = filename != None
        if write_to_file:
            open(filename, "w").close()
        times = []
        for trials in range(num_of_trials):
            duration = get_trial_duration(
                lambda_a, lambda_o, mu, num_of_servers, threshold, trials, repeat_trial
            )
            times.append(duration)
            if write_to_file:
                save_to_file(duration, filename)
    return times


def old_import_trials_duration(path):
    """ Needs to be deleted
    """
    times = []
    with open(path, "r") as textfile:
        string = textfile.read()
    string = list(string.split("\n"))
    for record in string[:-1]:
        string_list = record[1:-1].split(",")
        times.append([float(s) for s in string_list])
    return times


def import_trials_duration(path):
    """ Import some already existing local csv of times
    
    Parameters
    ----------
    path : [string]
        [the path that the file is located in]
    
    Returns
    -------
    [list]
        [a list of times to be plotted or viewed]
    """
    with open(path, "r") as times_file:
        csv_object = csv.reader(times_file)
        times = [[float(time) for time in row] for row in csv_object]
        return times


def get_duration_distribution_plot(times, kind=None):
    """ Produce the distribution of the simulation's duration for different number of trials
    
    Parameters
    ----------
    times : [list]
        [A list of durations of the simulation]
    kind : [string], optional
        [A keyword to identify the type of distribution plot ("violin": violinplot, "box": boxplot, otherwise: scatterplot)], by default None
    
    Returns
    -------
    [matplotlib object]
        [A distribution plot of the number of trials vs duration]
    """
    times_to_plot = []
    step = int(np.ceil(len(times) / 30))
    index = []
    for i in range(1, len(times), step):
        times_to_plot.append(times[i])
        for _ in range(len(times[i])):
            index.append(i)

    times_to_plot = np.matrix(times_to_plot).flatten().transpose()
    times_df = pd.DataFrame({"Number of trials": index})
    times_df["Time in seconds"] = times_to_plot
    if kind == "violin":
        plt.figure(figsize=(23, 10))
        time_plot = sbr.violinplot(
            x="Number of trials", y="Time in seconds", kind="box", data=times_df
        )
    elif kind == "box":
        time_plot = sbr.catplot(
            x="Number of trials",
            y="Time in seconds",
            kind="box",
            data=times_df,
            aspect=2.5,
        )
    else:
        time_plot = sbr.catplot(
            x="Number of trials", y="Time in seconds", data=times_df, aspect=2.5
        )

    return time_plot


def get_duration_all_lines_plot(times):
    """Produce a plot of the all repetitions of the duration of the simulation vs the number of trials
    
    Parameters
    ----------
    times : [list]
        [A list of durations of the simulation]
    
    Returns
    -------
    [type]
        [description]
    """
    plt.figure(figsize=(23, 10))
    time_plot = plt.plot(times)
    time_plot = plt.xlabel("Number of Trials")
    time_plot = plt.ylabel("Time in seconds")
    return time_plot


def get_duration_mean_plot(times):
    """Produce a plot of the mean duration of the simulation vs the number of trials
    
    Parameters
    ----------
    times : [list]
        [A list of durations of the simulation]
    
    Returns
    -------
    [matplotlib object]
        [plot of time (in seconds) vs trials]
    """
    plt.figure(figsize=(23, 10))
    time_plot = plt.plot([np.nanmean(t) for t in times])
    time_plot = plt.xlabel("Number of Trials")
    time_plot = plt.ylabel("Time in seconds")
    return time_plot


################################
### Second Timing Experiment ###
################################


def update_aggregated_list(times, plot_labels, aggregated_times, mean_times, list_pos):
    """Update the aggregated list so that it is plotted in the required way
    
    Parameters
    ----------
    times : [int]
        [The current iteration number]
    plot_labels : [list]
        [The x-axis labels to use for plotting]
    aggregated_times : [list]
        [Cumulative list of times]
    mean_times : [list]
        [List of the current trial's mean wiating/service/blocking times]
    list_pos : [int]
        [tracks position of aggregated_times list]
    
    Returns
    -------
    [list, list, int]
    """
    plot_labels.append(times)
    aggregated_times[list_pos] += mean_times
    list_pos += 1
    return plot_labels, aggregated_times, list_pos


def get_distributions_over_time(all_times):
    """Creates a cumulative list to show the distribution of number times as the number of trials increases. Thus a list is created where all values are aggregated in the form of lists. The lists format should look something like:
    [[x_1], [x_1, x_2], [x_1, x_2, x_3], [x_1, x_2, x_3, x_4], ... ]
    where x_i is the mean waiting/service/blocking time of the i_th trial
    
    Parameters
    ----------
    all_times : [list]
        [List of times to be plotted]
    
    Returns
    -------
    [list, list]
        [aggregates_times is a cumulative list where the i_th entry is a list that contains the same elements as the {i-1}_th entry including the mean of times of an additional trial]
        [plot_labels is a list that stores the x-axis labels to be used in the plot]
    """
    mean_times = []
    plot_labels = []
    list_pos = 0
    step = int(np.ceil(len(all_times) / 20))
    aggregated_times = [[] for _ in range(int(np.ceil(len(all_times) / step)))]
    for times in range(len(all_times)):
        mean_times.append(np.nanmean(all_times[times]))
        if times % step == 0:
            plot_labels, aggregated_times, list_pos = update_aggregated_list(
                times, plot_labels, aggregated_times, mean_times, list_pos
            )

    return [aggregated_times, plot_labels]


def get_plot_of_confidence_intervals_labels(time_type):
    """Get graph labels
    
    Parameters
    ----------
    time_type : [string]
        [A string to distinguish between times in order to adjust labels]
    
    Returns
    -------
    [string, string, string]
        [three strings that represent the graph's title, x-axis label and y-axis label]
    """
    if time_type == "w":
        title = "Mean Waiting time over number of iterations"
    elif time_type == "s":
        title = "Mean Service time over number of iterations"
    elif time_type == "b":
        title = "Mean time Blocked over number of iterations"
    else:
        title = " "

    x_axis = "Number of trials"
    y_axis = "Means of times"

    return title, x_axis, y_axis


def make_plot_of_confidence_intervals_over_iterations(all_times, time_type="b"):
    """Make a plot of waiting times confidence intervals over number of trials ran
    
    Parameters
    ----------
    all_times : [list]
        [The list output from the get_multiple_results() function that contains all time records]
    time_type : [string], optional
        [A letter to distinguish between which data to grab and plot (blocking times: "b", service times: "s", waiting times: "w")], by default "b"
    
    Returns
    -------
    [matplotlib object]
        [A plot of confidence intervals of times over number of trials]
    """
    if time_type == "w":
        all_times = all_times[0]
    elif time_type == "s":
        all_times = all_times[1]
    else:
        all_times = all_times[2]

    aggregated_times, plot_labels = get_distributions_over_time(all_times)
    title, x_axis_label, y_axis_label = get_plot_of_confidence_intervals_labels(
        time_type
    )

    plt.figure(figsize=(23, 10))
    plot = plt.boxplot(aggregated_times, whis=1, showfliers=False, labels=plot_labels)
    plt.title(title)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)

    return plot
