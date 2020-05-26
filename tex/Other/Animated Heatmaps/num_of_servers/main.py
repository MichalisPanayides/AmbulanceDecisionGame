"""
Code to generate the animated heatmaps
"""
import matplotlib.pyplot as plt
import numpy as np
import ambulance_game as abg


# Parameters
lambda_a = 0.4
lambda_o = 0.4
mu = 0.1
num_of_servers_space = np.linspace(1, 20, 20, dtype=int)
threshold = 10
system_capacity = 20
parking_capacity = 20

runtime = 10000
num_of_trials = 10
seed_num = 0

fig = 0
for num_of_servers in num_of_servers_space:
    plt.figure(figsize=(20, 10))
    abg.get_heatmaps(
        lambda_a,
        lambda_o,
        mu,
        int(num_of_servers),
        threshold,
        system_capacity,
        parking_capacity,
        runtime=runtime,
        num_of_trials=num_of_trials,
        seed_num=seed_num,
    )
    fig += 1
    plt.savefig("main_" + str(fig) + ".png")
    plt.close()
