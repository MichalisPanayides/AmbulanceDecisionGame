"""
Code to generate the example heatmaps
"""
import ambulance_game as abg
import matplotlib.pyplot as plt

lambda_2 = 0.1
lambda_1 = 0.1
mu = 0.1
seed_num = 10
num_of_trials = 100
runtime = 10000

num_of_servers = 4
threshold = 3
system_capacity = 5
buffer_capacity = 3

plt.figure(figsize=(20, 10))
abg.get_heatmaps(
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
plt.savefig("main.pdf")
plt.close()
