import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

from .markov import (
    build_states,
    visualise_ambulance_markov_chain,
)


def convert_networkxx_figure_to_tikz(
    num_of_servers, threshold, system_capacity, parking_capacity
):
    """TODO: Build a string of latex code that generates the tikz picture of the networkxx model as constructed by the networkxx library.
    """

    visualise_ambulance_markov_chain(
        num_of_servers=num_of_servers,
        threshold=threshold,
        system_capacity=system_capacity,
        parking_capacity=parking_capacity,
    )
    tikzplotlib.save("example.tex")


def generate_code_for_tikz_figure(
    num_of_servers, threshold, system_capacity, parking_capacity
):
    """Builds a string of latex code that generates the tikz picture of the Markov chain with the given parameters: number of servers (C), threshold (T), system capacity (N) and parking capacity (M).

    The function works using three loops:
        - First loop to build nodes and edges of states (0,0) - (0,T)
        - Second loop to build nodes and edges of states (0,T) - (M,T)
        - Third loop to build nodes and edges of the remaining states (the remainig rectangle of states) 

    Parameters
    ----------
    num_of_servers : int
    threshold : int
    system_capacity : int
    parking_capacity : int

    Returns
    -------
    string
        A string containing the full latex code to build a tikz figure of the Markov chain
    """
    tikz_code = (
        "\\begin{figure}[h]"
        + "\n"
        + "\\centering"
        + "\n"
        + "\\begin{tikzpicture}[-, node distance = 1cm, auto]"
        + "\n"
        + "\\node[state] (u0v0) {(0,0)};"
        + "\n"
    )
    service_rate = 0

    for v in range(1, min(threshold + 1, system_capacity + 1)):
        service_rate = (
            (service_rate + 1) if service_rate < num_of_servers else service_rate
        )

        tikz_code += (
            "\\node[state, right=of u0v"
            + str(v - 1)
            + "] (u0v"
            + str(v)
            + ") {("
            + str(0)
            + ","
            + str(v)
            + ")};"
            + "\n"
        )
        tikz_code += (
            "\\draw[->](u0v"
            + str(v - 1)
            + ") edge[bend left] node {\\( \\Lambda \\)} (u0v"
            + str(v)
            + ");"
            + "\n"
        )
        tikz_code += (
            "\\draw[->](u0v"
            + str(v)
            + ") edge[bend left] node {\\("
            + str(service_rate)
            + "\\mu \\)} (u0v"
            + str(v - 1)
            + ");"
            + "\n"
        )

    for u in range(1, parking_capacity + 1):
        tikz_code += (
            "\\node[state, below=of u"
            + str(u - 1)
            + "v"
            + str(v)
            + "] (u"
            + str(u)
            + "v"
            + str(v)
            + ") {("
            + str(u)
            + ","
            + str(v)
            + ")};"
            + "\n"
        )

        tikz_code += (
            "\\draw[->](u"
            + str(u - 1)
            + "v"
            + str(v)
            + ") edge[bend left] node {\\( \\lambda^A \\)} (u"
            + str(u)
            + "v"
            + str(v)
            + ");"
            + "\n"
        )
        tikz_code += (
            "\\draw[->](u"
            + str(u)
            + "v"
            + str(v)
            + ") edge[bend left] node {\\("
            + str(service_rate)
            + "\\mu \\)} (u"
            + str(u - 1)
            + "v"
            + str(v)
            + ");"
            + "\n"
        )

    for v in range(threshold + 1, system_capacity + 1):
        service_rate = (
            (service_rate + 1) if service_rate < num_of_servers else service_rate
        )

        for u in range(parking_capacity + 1):
            tikz_code += (
                "\\node[state, right=of u"
                + str(u)
                + "v"
                + str(v - 1)
                + "] (u"
                + str(u)
                + "v"
                + str(v)
                + ") {("
                + str(u)
                + ","
                + str(v)
                + ")};"
                + "\n"
            )

            tikz_code += (
                "\\draw[->](u"
                + str(u)
                + "v"
                + str(v - 1)
                + ") edge[bend left] node {\\( \\lambda^o \\)} (u"
                + str(u)
                + "v"
                + str(v)
                + ");"
                + "\n"
            )
            tikz_code += (
                "\\draw[->](u"
                + str(u)
                + "v"
                + str(v)
                + ") edge[bend left] node {\\("
                + str(service_rate)
                + "\\mu \\)} (u"
                + str(u)
                + "v"
                + str(v - 1)
                + ");"
                + "\n"
            )

            if u != 0:
                tikz_code += (
                    "\\draw[->](u"
                    + str(u - 1)
                    + "v"
                    + str(v)
                    + ") edge node {\\( \\lambda^A \\)} (u"
                    + str(u)
                    + "v"
                    + str(v)
                    + ");"
                    + "\n"
                )

    tikz_code += (
        "\\end{tikzpicture}"
        + "\n"
        + "\\caption{Markov chain model with "
        + str(num_of_servers)
        + " servers}"
        + "\n"
        + "\\label{Exmple_model-"
        + str(num_of_servers)
        + str(threshold)
        + str(system_capacity)
        + str(parking_capacity)
        + "}"
        + "\n"
        + "\\end{figure}"
    )

    tikz_code = tikz_code.replace("1\\mu", "\\mu")

    return tikz_code
