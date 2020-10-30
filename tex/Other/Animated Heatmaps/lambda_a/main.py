"""
Code to generate the animated heatmaps
"""
import matplotlib.pyplot as plt
import numpy as np
import ambulance_game as abg


# Parameters
lambda_2_space = np.linspace(0.1, 0.6, 20)
lambda_1 = 0.1
mu = 0.1
num_of_servers = 5
threshold = num_of_servers
system_capacity = 20
buffer_capacity = 20
runtime = 10000
num_of_trials = 10
seed_num = 0

fig = 0
for lambda_2 in lambda_2_space:
    plt.figure(figsize=(20, 10))
    abg.get_heatmaps(
        lambda_2,
        lambda_1,
        mu,
        num_of_servers,
        threshold,
        system_capacity,
        buffer_capacity,
        runtime=runtime,
        num_of_trials=num_of_trials,
        seed_num=seed_num,
    )
    fig += 1
    plt.savefig("main_" + str(fig) + ".png")
    plt.close()
