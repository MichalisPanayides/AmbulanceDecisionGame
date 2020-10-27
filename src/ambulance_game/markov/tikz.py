import numpy as np
from .graphical import (
    reset_L_and_R_in_array,
    find_next_permutation_over,
    find_next_permutation_over_L_and_R,
    generate_next_permutation_of_edges,
    check_permutation_is_valid,
)


def generate_code_for_tikz_figure(
    num_of_servers, threshold, system_capacity, buffer_capacity
):
    """Builds a string of latex code that generates the tikz picture of the Markov
    chain with the given parameters: number of servers (C), threshold (T),
    system capacity (N) and buffer capacity (M).

    The function works using three loops:
        - First loop to build nodes and edges of states (0,0) - (0,T)
        - Second loop to build nodes and edges of states (0,T) - (M,T)
        - Third loop to build nodes and edges of the remaining states
          (the remaining rectangle of states)

    Parameters
    ----------
    num_of_servers : int
    threshold : int
    system_capacity : int
    buffer_capacity : int

    Returns
    -------
    string
        A string containing the full latex code to build a tikz figure of the Markov
        chain
    """
    tikz_code = (
        "\\begin{tikzpicture}[-, node distance = 1cm, auto]"
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

    for u in range(1, buffer_capacity + 1):
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
            + ") edge[bend left] node {\\( \\lambda_2 \\)} (u"
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

        for u in range(buffer_capacity + 1):
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
                + ") edge[bend left] node {\\( \\lambda_1 \\)} (u"
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
                    + ") edge node {\\( \\lambda_2 \\)} (u"
                    + str(u)
                    + "v"
                    + str(v)
                    + ");"
                    + "\n"
                )

    tikz_code += "\\end{tikzpicture}"

    tikz_code = tikz_code.replace("1\\mu", "\\mu")

    return tikz_code


def build_body_of_tikz_spanning_tree(
    num_of_servers, threshold, system_capacity, buffer_capacity
):
    """Builds the main body of the tikz code"""
    main_body = (
        "\n\n\\begin{tikzpicture}[-, node distance = 1cm, auto]"
        + "\n"
        + "\\node[state] (u0v0) {(0,0)};"
        + "\n"
    )
    service_rate = 0

    for v in range(1, min(threshold + 1, system_capacity + 1)):
        service_rate = (
            (service_rate + 1) if service_rate < num_of_servers else service_rate
        )

        main_body += (
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

        main_body += (
            "\\draw[->](u0v"
            + str(v)
            + ") edge node {\\("
            + str(service_rate)
            + "\\mu \\)} (u0v"
            + str(v - 1)
            + ");"
            + "\n"
        )

    for u in range(1, buffer_capacity + 1):
        main_body += (
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

        main_body += (
            "\\draw[->](u"
            + str(u)
            + "v"
            + str(v)
            + ") edge node {\\("
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

        for u in range(buffer_capacity + 1):
            main_body += (
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

        main_body += (
            "\\draw[->](u"
            + str(u)
            + "v"
            + str(v)
            + ") edge node {\\("
            + str(service_rate)
            + "\\mu \\)} (u"
            + str(u)
            + "v"
            + str(v - 1)
            + ");"
            + "\n"
        )

    return main_body


def get_tikz_code_for_permutation(
    edges, num_of_servers, threshold, system_capacity, buffer_capacity
):
    """Given a specific valid permutation of edges that corresponds to a spanning
    tree of a Markov chain, generate tikz code to build that spanning tree.
    The function generates the appropriate string based on the elements of the
    edges array."""

    tikz_code = ""

    pos = 0
    service_rate = 1
    for u in range(buffer_capacity):
        service_rate = (
            num_of_servers if (threshold + 1) > num_of_servers else (threshold + 1)
        )
        for v in range(threshold + 1, system_capacity + 1):
            service_rate = (
                (service_rate + 1) if service_rate < num_of_servers else service_rate
            )
            if edges[pos] == "L":
                tikz_code += (
                    "\\draw[->](u"
                    + str(u)
                    + "v"
                    + str(v)
                    + ") edge node {\\("
                    + str(service_rate)
                    + "\\mu \\)} (u"
                    + str(u)
                    + "v"
                    + str(v - 1)
                    + ");"
                    + "\n"
                )
            elif edges[pos] == "D":
                tikz_code += (
                    "\\draw[->](u"
                    + str(u)
                    + "v"
                    + str(v)
                    + ") edge node {\\(\\lambda_2 \\)} (u"
                    + str(u + 1)
                    + "v"
                    + str(v)
                    + ");"
                    + "\n"
                )
            elif edges[pos] == "R":
                tikz_code += (
                    "\\draw[->](u"
                    + str(u)
                    + "v"
                    + str(v)
                    + ") edge node {\\(\\lambda_1 \\)} (u"
                    + str(u)
                    + "v"
                    + str(v + 1)
                    + ");"
                    + "\n"
                )
            pos += 1

    return tikz_code


def generate_code_for_tikz_spanning_trees_rooted_at_00(
    num_of_servers, threshold, system_capacity, buffer_capacity
):
    """Builds a string of latex code that generates tikz pictures of all spanning
    trees of the Markov chain that are rooted at node (0,0). The function considers
    the Markov chain with the given parameters and performs the following steps:
        - FOR a specific combination of edges (e.g. 2 x down_edges, 3 x right_edges and 2 x left_edges):
            - Initialise an array with the corresponding values i.e. ["L","L","R","R","R","D","D"]
            - WHILE more trees exist with these specific values:
                - if the array can be translated into a valid spanning tree (no cycles):
                    - Generate tikz code for that array
                - Generate the next permutation
                  i.e. ["L","L","R","R","R","D","D"] -> ["L","R","L","R","R","D","D"]
                - if no more permutations can be generated exit the while loop
            - Move to next combination of edges until all combinations are considered
        - Add a permutation with only left_edges ["L", "L", ..., "L"]
    Parameters
    ----------
    num_of_servers : int
    threshold : int
    system_capacity : int
    buffer_capacity : int


    Yields
    -------
    str
        a string of latex_code that will generate a specific spanning tree
    """

    spanning_tree_counter = 1
    for down_edges in np.linspace(
        buffer_capacity * (system_capacity - threshold),
        1,
        buffer_capacity * (system_capacity - threshold),
        dtype=int,
    ):
        for right_edges in range(
            buffer_capacity * (system_capacity - threshold) - down_edges + 1
        ):
            edges_index = [
                "D"
                if (i >= buffer_capacity * (system_capacity - threshold) - down_edges)
                else "N"
                for i in range(buffer_capacity * (system_capacity - threshold))
            ]
            left_edges = (
                buffer_capacity * (system_capacity - threshold)
                - down_edges
                - right_edges
            )

            for pos in range(left_edges):
                edges_index[pos] = "L"

            for pos in range(left_edges, left_edges + right_edges):
                edges_index[pos] = "R"

            more_trees_exist = True
            while more_trees_exist:
                is_valid = check_permutation_is_valid(edges_index, buffer_capacity)
                if is_valid:
                    spanning_tree_counter += 1

                    tikz_code = build_body_of_tikz_spanning_tree(
                        num_of_servers, threshold, system_capacity, buffer_capacity
                    )

                    tikz_code += get_tikz_code_for_permutation(
                        edges_index,
                        num_of_servers,
                        threshold,
                        system_capacity,
                        buffer_capacity,
                    )
                    tikz_code += "\\end{tikzpicture}"
                    tikz_code = tikz_code.replace("1\\mu", "\\mu")
                    yield tikz_code

                edges_index = generate_next_permutation_of_edges(
                    edges=edges_index,
                    downs=down_edges,
                    lefts=left_edges,
                    rights=right_edges,
                )

                if edges_index == []:
                    more_trees_exist = False

    edges_index = ["L" for _ in range(buffer_capacity * (system_capacity - threshold))]
    tikz_code = build_body_of_tikz_spanning_tree(
        num_of_servers, threshold, system_capacity, buffer_capacity
    )
    tikz_code += get_tikz_code_for_permutation(
        edges_index,
        num_of_servers,
        threshold,
        system_capacity,
        buffer_capacity,
    )
    tikz_code += "\\end{tikzpicture}"
    tikz_code = tikz_code.replace("1\\mu", "\\mu")
    yield tikz_code
