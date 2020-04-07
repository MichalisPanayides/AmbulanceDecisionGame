import ambulance_game as abg
import matplotlib.pyplot as plt
import numpy as np

# Queueing Parameters
lambda_a = 0.09  # Ambulance: 6 arrivals per hour
lambda_o = 0.09  # Others: 3 arrivals per hour
mu = 0.013  # Service time: 3 services per hour
num_of_servers = 15  # Number of servers: 8 servers
threshold = 12  # Threshold of accepting ambulance patients
seed_num = None
warm_up_time = 100
output_type = "list"
runtime = 1440

# Timing Experiment parameters
num_of_trials = 50  # Number of trials to be considered
repetition = 5  # Repetition of each trial
method = "Simulation"  # Method to be used (only Simulation available)
measurement_type = "w"

# Model Plots Parameters
max_threshold = num_of_servers + 1
target = 4
lambda_o_1 = 0.08
lambda_o_2 = 0.08
mu_1 = 0.03
mu_2 = 0.03
num_of_servers_1 = 6
num_of_servers_2 = 6
threshold_1 = 3
threshold_2 = 3
seed_num_1 = None
seed_num_2 = None


abg.make_plot_for_proportion_within_target(lambda_a, lambda_o, mu, num_of_servers, num_of_trials, seed_num, target, runtime=1440, max_threshold=max_threshold)
plt.show()