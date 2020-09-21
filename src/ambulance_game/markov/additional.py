import math
import numpy as np


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

    tikz_code += "\\end{tikzpicture}"

    tikz_code = tikz_code.replace("1\\mu", "\\mu")

    return tikz_code


def reset_L_and_R_in_array(edges, lefts):
    """
    Take an array and re-sorts the values in such a way such that:
    - All "D" values remain in the exact same position
    - In the remaining spaces, "L" and "R" are sorted starting from the left with all "L"

    Example
    -----------
    Input: [D, R, R, D, L, L, L]
    Output: [D, L, L, D, L, R, R]
    """

    L_count = 0
    for pos, element in enumerate(edges):
        reset_this_entry = element == "L" or element == "R"
        if reset_this_entry and L_count < lefts:
            edges[pos] = "L"
            L_count += 1
        elif reset_this_entry:
            edges[pos] = "R"
    return edges


def find_next_permutation_over(edges, direction, rights=0, permute_over="D"):
    """Finds the next permutation of an array (edges) by permuting a specific element of the array (direction) over another specified element of the array (permute_over).
    [X, X, Y, Y]->[X, Y, X, Y]->[X, Y, Y, X] -> [Y, X, X, Y] ... -> [Y, Y, X, X]

    This function is used in the following cases:
        - If the array consists only of elements "L" and "D" (direction="L"):
            - The rightmost "L" value will be replaced with the "D" value that is exactly after it.
            - If there is no "D" after the last "L" (meaning "L" is already in the last position):
                1. Turn all consecutive rightmost "L" into "D"
                2. Find an "L" value with a "D" in the next position.
                3. Replace that "L" with "D"
                4. Turn (the same amount as in (1)) elements after it into "L"

        - If the array conssists only of elements "R" and "D" (direction="R"):
            - Same as case of "L" and "D"

        - If the array conssists of elements "L", "R" and "D" (direction="LR"):
            - Treats all "L" and "R" values as the same element
            - Performs the same opperations as above with (L+R vs D)

        - If the array conssists only of elements "L" and "R" (direction="L", permute_over="R"):
            - Performs the same opperations as above with (L vs R)

    Example 1 (direction = "L")
    ----------
    Input: [L, L, D, L, D]
    Output: [L, L, D, D, L]

    Example 2 (direction = "R")
    ----------
    Input: [R, D, D, D, R]
    Output: [D, R, R, D, D]

    Example 3 (direction = "LR")
    ----------
    Input: [L, L, R, D, D]
    Output: [L, L, D, R, D]

    Example 4 (direction = "L", permute_over = "R")
    ----------
    Input: [L, L, R]
    Output: [L, R, L]

    Parameters
    ----------
    edges : array
    direction : str
        Indicating whether to consider "L" or "R" direction or both ("LR")
    """
    if direction == "LR":
        for pos, element in enumerate(edges[:-1]):
            if (element == "L" or element == "R") and edges[pos + 1] == permute_over:
                target_position = pos

        pos_last_D = len(edges) - edges[::-1].index(permute_over) - 1
        edges_to_be_swapped = len(edges) - pos_last_D
        edges[target_position] = permute_over

        direction = "L"
        for i in range(edges_to_be_swapped):
            edges[-1 - i] = permute_over
        for i in range(edges_to_be_swapped):
            if i >= edges_to_be_swapped - rights:
                direction = "R"
            edges[target_position + 1 + i] = direction

    else:
        for pos, element in enumerate(edges[:-1]):
            if element == direction and edges[pos + 1] == permute_over:
                target_position = pos

        pos_last_D = len(edges) - edges[::-1].index(permute_over) - 1
        edges_to_be_swapped = len(edges) - pos_last_D
        edges[target_position] = permute_over
        for i in range(edges_to_be_swapped):
            edges[-1 - i] = permute_over
        for i in range(edges_to_be_swapped):
            edges[target_position + 1 + i] = direction
    return edges


def find_next_permutation_over_L_and_R(edges):
    """This function deals with permutations of "L" and "R" while not changing positions to any other element. In essence, it only changes the positions of "L" and "R" elements in an orderly manner.

    Example
    ----------
    Input: [L, R, D, D, R]
    Output: [R, L, D, D, R]
    """
    only_LR_edges = []
    only_LR_positions = []
    for pos, element in enumerate(edges):
        if element == "L" or element == "R":
            only_LR_edges.append(element)
            only_LR_positions.append(pos)

    only_LR_edges = find_next_permutation_over(
        edges=only_LR_edges, direction="L", permute_over="R"
    )

    for pos, pos_LR in enumerate(only_LR_positions):
        edges[pos_LR] = only_LR_edges[pos]

    return edges


def generate_next_permutation_of_edges(edges, downs, lefts, rights):
    """Given an array of with elements "L", "R" and "D" finds the next permutation of the elements in an orderly manner such that all possible combinations considered at the end.

    Parameters
    ----------
    edges : array
        The current permutatioin of the edges
    downs : int
        Number of down-edges that exist in the array
    lefts : int
        Number of left-edges that exist in the array
    rights : int
        Number of right-edges that exist in the array

    Returns
    -------
    array
        Next permutation of the edges array
    """
    if "L" in edges and "R" in edges:
        all_L_positions = []
        all_R_positions = []
        for pos, element in enumerate(edges):
            if element == "L":
                all_L_positions.append(pos)
            elif element == "R":
                all_R_positions.append(pos)

        if max(all_R_positions) > min(all_L_positions):
            edges = find_next_permutation_over_L_and_R(edges=edges)
        else:
            edges = reset_L_and_R_in_array(edges=edges, lefts=lefts)
            pos_last_D = len(edges) - edges[::-1].index("D") - 1
            if pos_last_D == (downs - 1):
                return []
            else:
                edges = find_next_permutation_over(
                    edges=edges, direction="LR", rights=rights
                )

    elif "L" in edges:
        pos_last_D = len(edges) - edges[::-1].index("D") - 1
        if pos_last_D == (downs - 1):
            return []
        edges = find_next_permutation_over(edges=edges, direction="L", rights=rights)

    elif "R" in edges:
        pos_last_D = len(edges) - edges[::-1].index("D") - 1
        if pos_last_D == (downs - 1):
            return []
        edges = find_next_permutation_over(edges=edges, direction="R", rights=rights)

    else:
        edges = []

    return edges


def check_permutation_is_valid(edges, parking_capacity):
    """Check that the given array is a valid spanning tree of the graph.
    Specifically, a given array is not a valid spanning tree if:
        - Any element that corresponds to a node of the final column is "R" (nodes of last column cannot have a right edge)
        - If there exist an "L" value exaclty after an "R" value (would make a cycle between two nodes)"""

    start = (len(edges) / parking_capacity) - 1
    for pos in np.linspace(start, len(edges) - 1, parking_capacity, dtype=int):
        if edges[pos] == "R":
            return False

    for pos, element in enumerate(edges[:-1]):
        if element == "R" and edges[pos + 1] == "L":
            return False

    return True


def build_body_of_tikz_spanning_tree(
    num_of_servers, threshold, system_capacity, parking_capacity
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

    for u in range(1, parking_capacity + 1):
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

        for u in range(parking_capacity + 1):
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
    edges, num_of_servers, threshold, system_capacity, parking_capacity
):
    """Given a specific valid permutation of edges that corresponds to a spanning tree of a Markov chain, generate tikz code to build that spanning tree. The function generates the appropriate string based on the elements of the edges array."""

    tikz_code = ""

    pos = 0
    service_rate = 1
    for u in range(parking_capacity):
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
                    + ") edge node {\\(\\lambda^A \\)} (u"
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
                    + ") edge node {\\(\\lambda^o \\)} (u"
                    + str(u)
                    + "v"
                    + str(v + 1)
                    + ");"
                    + "\n"
                )
            pos += 1

    return tikz_code


def generate_code_for_tikz_spanning_trees_rooted_at_00(
    num_of_servers, threshold, system_capacity, parking_capacity
):
    """Builds a string of latex code that generates tikz pictures of all spaning trees of the Markov chain that are rooted at node (0,0). The function considers the markov chain with the given paramaters and performs the following steps:
        - FOR a specific combination of edges (e.g. 2 x down_edges, 3 x right_edges and 2 x left_edges):
            - Initialise an array with the corresponding values i.e. ["L","L","R","R","R","D","D"]
            - WHILE more trees exist with these specific values:
                - if the array can be translated into a valid spanning tree (no cycles):
                    - Generate tikz code for that array
                - Generate the next permutation i.e. ["L","L","R","R","R","D","D"] -> ["L","R","L","R","R","D","D"]
                - if no more permutations can be generated exit the while loop
            - Move to next combination of edges until all combinations are considered
        - Add a permutation with only left_edges ["L", "L", ..., "L"]
    Parameters
    ----------
    num_of_servers : int
    threshold : int
    system_capacity : int
    parking_capacity : int


    Yields
    -------
    str
        a string of latex_code that will generate a specific spanning tree
    """

    spanning_tree_counter = 1
    for down_edges in np.linspace(
        parking_capacity * (system_capacity - threshold),
        1,
        parking_capacity * (system_capacity - threshold),
        dtype=int,
    ):
        for right_edges in range(
            parking_capacity * (system_capacity - threshold) - down_edges + 1
        ):
            edges_index = [
                "D"
                if (i >= parking_capacity * (system_capacity - threshold) - down_edges)
                else "N"
                for i in range(parking_capacity * (system_capacity - threshold))
            ]
            left_edges = (
                parking_capacity * (system_capacity - threshold)
                - down_edges
                - right_edges
            )

            for pos in range(left_edges):
                edges_index[pos] = "L"

            for pos in range(left_edges, left_edges + right_edges):
                edges_index[pos] = "R"

            more_trees_exist = True
            while more_trees_exist:
                is_valid = check_permutation_is_valid(edges_index, parking_capacity)
                if is_valid:
                    spanning_tree_counter += 1

                    tikz_code = build_body_of_tikz_spanning_tree(
                        num_of_servers, threshold, system_capacity, parking_capacity
                    )

                    tikz_code += get_tikz_code_for_permutation(
                        edges_index,
                        num_of_servers,
                        threshold,
                        system_capacity,
                        parking_capacity,
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

    edges_index = ["L" for _ in range(parking_capacity * (system_capacity - threshold))]
    tikz_code = build_body_of_tikz_spanning_tree(
        num_of_servers, threshold, system_capacity, parking_capacity
    )
    tikz_code += get_tikz_code_for_permutation(
        edges_index,
        num_of_servers,
        threshold,
        system_capacity,
        parking_capacity,
    )
    tikz_code += "\\end{tikzpicture}"
    tikz_code = tikz_code.replace("1\\mu", "\\mu")
    yield tikz_code


def get_rate_of_state_00_graphically(
    lambda_a, lambda_o, mu, num_of_servers, threshold, system_capacity, parking_capacity
):
    """Calculates the unnormalised rate of state (0,0) using the same permutation
    algorithm used in function generate_code_for_tikz_spanning_trees_rooted_at_00().
    The function considers the markov chain with the given paramaters and performs
    the following steps:
        - FOR a specific combination of edges (e.g. 2 x down_edges, 3 x right_edges and 2 x left_edges):
            - Initialise an array with the corresponding values i.e. ["L","L","R","R","R","D","D"]
            - WHILE more trees exist with these specific values:
                - if the array can be translated into a valid spanning tree (no cycles):
                    - +1 to the number of spanning trees
                - Generate the next permutation i.e. ["L","L","R","R","R","D","D"] -> ["L","R","L","R","R","D","D"]
                - if no more permutations can be generated exit the while loop
                - Add to the total P00_rate the term with the number of all possible spannign
                    trees multiplied by lambda_a raised to the power of the down edges, multiplied
                    by lambda_o raised to the power of the right edges, multiplied by mu raised
                    to the power of the left edges:
                    e.g num_of_spanning_trees * (λ_α^2) * (λ_ο^3) * (μ^2)
            - Move to next combination of edges until all combinations are considered
        - Add to P00_rate the term: μ^(N-T)
        - Muliplry P00_rate by the term: μ^(N*M)

    TODO: fix function for case of num_of_servers > 1
    
    Parameters
    ----------
    lambda_a : float
    lambda_o : float
    mu : float
    num_of_servers : int
    threshold : int
    system_capacity : int
    parking_capacity : int

    Returns
    -------
        The unnormalised rate of state (0,0) (P_00)
    """

    if num_of_servers != 1:
        return "Unable to calculate for cases where number of servers is not 1"

    P00_rate = 0
    for down_edges in np.linspace(
        parking_capacity * (system_capacity - threshold),
        1,
        parking_capacity * (system_capacity - threshold),
        dtype=int,
    ):
        for right_edges in range(
            parking_capacity * (system_capacity - threshold) - down_edges + 1
        ):
            spanning_tree_counter = 0
            edges_index = [
                "D"
                if (i >= parking_capacity * (system_capacity - threshold) - down_edges)
                else "N"
                for i in range(parking_capacity * (system_capacity - threshold))
            ]
            left_edges = (
                parking_capacity * (system_capacity - threshold)
                - down_edges
                - right_edges
            )

            for pos in range(left_edges):
                edges_index[pos] = "L"

            for pos in range(left_edges, left_edges + right_edges):
                edges_index[pos] = "R"

            more_trees_exist = True
            while more_trees_exist:
                is_valid = check_permutation_is_valid(edges_index, parking_capacity)
                if is_valid:
                    spanning_tree_counter += 1
                edges_index = generate_next_permutation_of_edges(
                    edges=edges_index,
                    downs=down_edges,
                    lefts=left_edges,
                    rights=right_edges,
                )
                if edges_index == []:
                    more_trees_exist = False

            P00_rate += (
                spanning_tree_counter * lambda_a ** down_edges
                + lambda_o ** right_edges
                + mu ** left_edges
            )

    P00_rate += mu ** (system_capacity - threshold)
    P00_rate *= mu ** (system_capacity * parking_capacity)

    return P00_rate
