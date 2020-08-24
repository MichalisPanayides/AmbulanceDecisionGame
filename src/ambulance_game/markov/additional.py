import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

from .markov import (
    build_states,
    visualise_ambulance_markov_chain,
)


def convert_networkxx_figure_to_tikz(num_of_servers, threshold, system_capacity, parking_capacity):

    visualise_ambulance_markov_chain(num_of_servers=num_of_servers, threshold=threshold, system_capacity=system_capacity, parking_capacity=parking_capacity)
    tikzplotlib.save("example.tex")


def generate_code_for_tikz_figure(num_of_servers, threshold, system_capacity, parking_capacity):
    string = "\\begin{figure}[h]" + "\n" + "\\centering" + "\n" + "\\begin{tikzpicture}[-, node distance = 1cm, auto]" + "\n" + "\\node[state] (u0v0) {(0,0)};" + "\n"
    service_rate = 0

    for v in range(1, min(threshold + 1, system_capacity + 1)):
        service_rate = (service_rate + 1) if service_rate < num_of_servers else service_rate

        string += "\\node[state, right=of u0v" + str(v-1) + "] (u0v"+str(v)+") {("+str(0) + "," + str(v) + ")};" + "\n"
        string += "\\draw(u0v"+str(v-1)+") edge[bend left] node {\\( \\Lambda \\)} (u0v"+str(v)+");" + "\n"
        string += "\\draw(u0v"+str(v)+") edge[bend left] node {\\(" + str(service_rate) + "\\mu \\)} (u0v"+str(v-1)+");" + "\n"


    return string