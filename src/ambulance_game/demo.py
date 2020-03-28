import ambulance_game as abg
import matplotlib.pyplot as plt
import numpy as np

# Queueing Parameters
lambda_a = 0.1  # Ambulance: 6 arrivals per hour
lambda_o = 0.1  # Others: 3 arrivals per hour
mu = 0.05  # Service time: 3 services per hour
num_of_servers = 8  # Number of servers: 8 servers
threshold = 4  # Threshold of accepting ambulance patients
seed_num = None
warm_up_time = 100
output_type = "list"
runtime = 1440

# Timing Experiment parameters
num_of_trials = 10  # Number of trials to be considered
repetition = 5  # Repetition of each trial
method = "Simulation"  # Method to be used (only Simulation available)


# Model Plots Parameters
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


records = abg.models.simulate_model(lambda_a=lambda_a, lambda_o=lambda_o, mu=mu, num_of_servers=num_of_servers, threshold=threshold, seed_num=seed_num, runtime=runtime).get_all_records()
blocks = [b.time_blocked for b in records]
waits = [w.waiting_time for w in records]
print("Mean blocking time: ", np.mean(blocks))
print("Mean waiting time: ", np.mean(waits))