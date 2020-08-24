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
        string += "\\draw[->](u0v"+str(v-1)+") edge[bend left] node {\\( \\Lambda \\)} (u0v"+str(v)+");" + "\n"
        string += "\\draw[->](u0v"+str(v)+") edge[bend left] node {\\(" + str(service_rate) + "\\mu \\)} (u0v"+str(v-1)+");" + "\n"

    for u in range(1, parking_capacity + 1):
        string += "\\node[state, below=of u" + str(u-1) + "v" + str(v) + "] (u"+str(u)+"v"+str(v)+") {("+str(u) + "," + str(v) + ")};" + "\n"
        
        string += "\\draw[->](u"+str(u-1)+"v"+str(v)+") edge[bend left] node {\\( \\lambda^A \\)} (u"+str(u)+"v"+str(v)+");" + "\n"
        string += "\\draw[->](u"+str(u)+"v"+str(v)+") edge[bend left] node {\\(" + str(service_rate) + "\\mu \\)} (u"+str(u-1)+"v"+str(v)+");" + "\n"

    for v in range(threshold + 1, system_capacity + 1):
        service_rate = (service_rate + 1) if service_rate < num_of_servers else service_rate

        for u in range(parking_capacity + 1):
            string += "\\node[state, right=of u" + str(u) +"v" + str(v-1) + "] (u"+str(u)+"v"+str(v)+") {("+str(u) + "," + str(v) + ")};" + "\n"
            
            string += "\\draw[->](u"+str(u)+"v"+str(v-1)+") edge[bend left] node {\\( \\lambda^o \\)} (u"+str(u)+"v"+str(v)+");" + "\n"
            string += "\\draw[->](u"+str(u)+"v"+str(v)+") edge[bend left] node {\\(" + str(service_rate) + "\\mu \\)} (u"+str(u)+"v"+str(v-1)+");" + "\n"

            if u != 0:
                string += "\\draw[->](u"+str(u-1)+"v"+str(v)+") edge node {\\( \\lambda^A \\)} (u"+str(u)+"v"+str(v)+");" + "\n"

    string += "\\end{tikzpicture}" + "\n" + "\\caption{Markov chain model with " + str(num_of_servers) + " servers}" + "\n" + "\\label{Exmple_model-" + str(num_of_servers) + str(threshold) + str(system_capacity) + str(parking_capacity) +"}" + "\n" + "\\end{figure}"
    
    return string





# \begin{figure}[h]
#     \centering
#     \begin{tikzpicture}[-, node distance = 1cm, auto]
#         \node[state] (empty) {(0,0)};
#         \node[state, right=of empty] (one) {(0,1)};
#         \node[state, right=of one] (two) {(0,2)};
#         \node[state, right=of two] (three) {(0,3)};
#         \node[state, right=of three] (four) {(0,4)};
#         \node[state, right=of four] (five) {(0,5)};

#         \node[state, below=of three] (three_one) {(1,3)};
#         \node[state, below=of three_one] (three_two) {(2,3)};
#         \node[state, below=of four] (four_one) {(1,4)};
#         \node[state, below=of four_one] (four_two) {(2,4)};
#         \node[state, below=of five] (five_one) {(1,5)};
#         \node[state, below=of five_one] (five_two) {(2,5)};

#         \draw[every loop]
#             (empty) edge[bend left] node {\( \Lambda \)} (one)
#             (one) edge[bend left] node {\( \mu \)} (empty)
#             (one) edge[bend left] node {\( \Lambda \)} (two)
#             (two) edge[bend left] node {\( 2 \mu \)} (one)
#             (two) edge[bend left] node {\( \Lambda \)} (three)
#             (three) edge[bend left] node {\( 3 \mu \)} (two)
#             (three) edge[bend left] node {\( \lambda^o \)} (four)
#             (four) edge[bend left] node {\( 4 \mu \)} (three)
#             (four) edge[bend left] node {\( \lambda^o \)} (five)
#             (five) edge[bend left] node {\( 4 \mu \)} (four)
#             (three) edge[bend left] node {\( \lambda^A \)} (three_one)
#             (three_one) edge[bend left] node {\( 3 \mu \)} (three)
#             (three_one) edge[bend left] node {\( \lambda^o \)} (four_one)
#             (four_one) edge[bend left] node {\( 4 \mu \)} (three_one)
#             (four_one) edge[bend left] node {\( \lambda^o \)} (five_one)
#             (five_one) edge[bend left] node {\( 4 \mu \)} (four_one)
#             (four) edge node {\( \lambda^A \)} (four_one)
#             % (four_one) edge[bend left] node {\( \mu \)} (four)
#             (five) edge node {\( \lambda^A \)} (five_one)
#             % (five_one) edge[bend left] node {\( \mu \)} (five)
#             (three_one) edge[bend left] node {\( \lambda^A \)} (three_two)
#             (three_two) edge[bend left] node {\( 3 \mu \)} (three_one)
#             (four_one) edge node {\( \lambda^A \)} (four_two)
#             % (four_two) edge[bend left] node {\( \mu \)} (four_one)
#             (five_one) edge node {\( \lambda^A \)} (five_two)
#             % (five_two) edge[bend left] node {\( \mu \)} (five_one)
#             (three_two) edge[bend left] node {\( \lambda^o \)} (four_two)
#             (four_two) edge[bend left] node {\( 4 \mu \)} (three_two)
#             (four_two) edge[bend left] node {\( \lambda^o \)} (five_two)
#             (five_two) edge[bend left] node {\( 4 \mu \)} (four_two)
#             ;       
#     \end{tikzpicture}
#     \caption{Markov chains: number of servers=4} 
#     \label{Model_mini}
# \end{figure}
