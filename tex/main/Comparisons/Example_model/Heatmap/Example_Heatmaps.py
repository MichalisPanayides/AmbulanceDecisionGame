"""
Code to generate the example heatmaps
"""
import ambulance_game as abg
import matplotlib.pyplot as plt

lambda_a = 0.1
lambda_o = 0.1
mu = 0.1

num_of_servers = 4
threshold = 3
system_capacity = 5
parking_capacity = 3

abg.get_heatmaps(lambda_a, lambda_o, mu, num_of_servers, threshold, system_capacity, parking_capacity, None, 1440, 10)
plt.savefig("Example_Heatmaps.pdf")
plt.close()

