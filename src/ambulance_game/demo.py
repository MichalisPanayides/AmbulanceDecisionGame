import ambulance_game as abg
import matplotlib.pyplot as plt
import csv

# Queueing Parameters
lambda_a = 0.15       # Ambulance: 6 arrivals per hour
lambda_o = 0.2      # Others: 3 arrivals per hour
mu = 0.05            # Service time: 3 services per hour
total_capacity = 8   # Number of servers: 8 servers
threshold = 4        # Threshold of accepting ambulance patients
seed_num = None
warm_up_time = 100
output_type = "list"    

# Timing Experiment parameters
num_of_trials = 10      # Number of trials to be considered
repetition = 5           # Repetition of each trial
method = "Simulation"     # Method to be used (only Simulation available)


# Model Plots Parameters
target = 4
lambda_o_1 = 0.08
lambda_o_2 =0.08
mu_1 = 0.03
mu_2 = 0.03
total_capacity_1 = 6
total_capacity_2 = 6
threshold_1 = 3
threshold_2 = 3
seed_num_1 = None
seed_num_2 = None


# abg.make_proportion_plot(lambda_a, lambda_o, mu, total_capacity, num_of_trials, seed_num, target)
# plt.show()

# records = abg.models.simulate_model(lambda_a, lambda_o, mu, total_capacity, threshold, seed_num).get_all_records()
# print(records)

# abg.make_plot_two_hospitals_arrival_split(lambda_a, lambda_o_1, lambda_o_2, mu_1, mu_2, total_capacity_1, total_capacity_2, threshold_1, threshold_2, "b", seed_num_1, seed_num_2, warm_up_time, num_of_trials, accuracy=50)
# plt.show()

# abg.make_plot_two_hospitals_arrival_split(lambda_a, lambda_o_1, lambda_o_2, mu_1, mu_2, total_capacity_1, total_capacity_2, threshold_1, threshold_2, "w", seed_num_1, seed_num_2, warm_up_time, num_of_trials, accuracy=20)
# plt.show()

# abg.make_plot_two_hospitals_arrival_split(lambda_a, lambda_o_1, lambda_o_2, mu_1, mu_2, total_capacity_1, total_capacity_2, threshold_1, threshold_2, "b", seed_num_1, seed_num_2, warm_up_time, num_of_trials, accuracy=20)
# plt.show()

abg.make_plot_for_different_thresholds(lambda_a, lambda_o, mu, total_capacity, num_of_trials, seed_num, measurement_type="w")
plt.show()
# abg.make_plot_for_different_thresholds(lambda_a, lambda_o, mu, total_capacity, seed_num, measurement_type="b")
# plt.show()
# abg.make_plot_for_different_thresholds(lambda_a, lambda_o, mu, total_capacity, seed_num, measurement_type="both")
# plt.show()


# times = abg.time_for_different_number_of_trials(lambda_a, lambda_o, mu, total_capacity, threshold, num_of_trials, repetition, method)
# abg.get_all_lines_plot(times)
# plt.show()

# times = import_trials_duration("Custom_trials.csv")  
# abg.get_all_lines_plot(times)
# plt.show()

# abg.get_mean_plot(times)
# plt.show()

# abg.get_distribution_plot(times)
# plt.show()

# times = abg.models.get_multiple_runs_results(lambda_a, lambda_o, mu, total_capacity, threshold, seed_num, warm_up_time, num_of_trials, output_type)

# abg.make_plot_of_confidence_intervals(all_times=times, time_type="w", filename="Trials_100.csv")
#plt.show()

# print(times)

