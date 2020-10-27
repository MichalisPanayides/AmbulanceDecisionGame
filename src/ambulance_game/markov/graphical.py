import numpy as np


def reset_L_and_R_in_array(edges, lefts):
    """
    Take an array and re-sorts the values in such a way such that:
    - All "D" values remain in the exact same position
    - In the remaining spaces, "L" and "R" are sorted starting from the left with
    all "L"

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
    """Finds the next permutation of an array (edges) by permuting a specific
    element of the array (direction) over another specified element of the array
    (permute_over).
    [X, X, Y, Y]->[X, Y, X, Y]->[X, Y, Y, X] -> [Y, X, X, Y] ... -> [Y, Y, X, X]

    This function is used in the following cases:
        - If the array consists only of elements "L" and "D" (direction="L"):
            - The rightmost "L" value will be replaced with the "D" value that is
            exactly after it.
            - If there is no "D" after the last "L" (meaning "L" is already in the
            last position):
                1. Turn all consecutive rightmost "L" into "D"
                2. Find an "L" value with a "D" in the next position.
                3. Replace that "L" with "D"
                4. Turn (the same amount as in (1)) elements after it into "L"

        - If the array consists only of elements "R" and "D" (direction="R"):
            - Same as case of "L" and "D"

        - If the array consists of elements "L", "R" and "D" (direction="LR"):
            - Treats all "L" and "R" values as the same element
            - Performs the same operations as above with (L+R vs D)

        - If the array consists only of elements "L" and "R"
        (direction="L", permute_over="R"):
            - Performs the same operations as above with (L vs R)

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
    """This function deals with permutations of "L" and "R" while not changing
    positions to any other element. In essence, it only changes the positions of
    "L" and "R" elements in an orderly manner.

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
    """Given an array of with elements "L", "R" and "D" finds the next permutation
    of the elements in an orderly manner such that all possible combinations
    considered at the end.

    Parameters
    ----------
    edges : array
        The current permutation of the edges
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


def check_permutation_is_valid(edges, buffer_capacity):
    """Check that the given array is a valid spanning tree of the graph.
    Specifically, a given array is not a valid spanning tree if:
        - Any element that corresponds to a node of the final column is "R"
        (nodes of last column cannot have a right edge)
        - If there exist an "L" value exactly after an "R" value
        (would make a cycle between two nodes)"""

    start = (len(edges) / buffer_capacity) - 1
    for pos in np.linspace(start, len(edges) - 1, buffer_capacity, dtype=int):
        if edges[pos] == "R":
            return False

    for pos, element in enumerate(edges[:-1]):
        if element == "R" and edges[pos + 1] == "L":
            return False

    return True


def get_rate_of_state_00_graphically(
    lambda_2, lambda_1, mu, num_of_servers, threshold, system_capacity, buffer_capacity
):
    """Calculates the unnormalized rate of state (0,0) using the same permutation
    algorithm used in function generate_code_for_tikz_spanning_trees_rooted_at_00().
    The function considers the Markov chain with the given parameters and performs
    the following steps:
        - FOR a specific combination of edges
        (e.g. 2 x down_edges, 3 x right_edges and 2 x left_edges):
            - Initialise an array with the corresponding values
            i.e. ["L","L","R","R","R","D","D"]
            - WHILE more trees exist with these specific values:
                - if the array can be translated into a valid spanning tree
                (no cycles):
                    - +1 to the number of spanning trees
                - Generate the next permutation
                i.e. ["L","L","R","R","R","D","D"] -> ["L","R","L","R","R","D","D"]
                - if no more permutations can be generated exit the while loop
                - Add to the total P00_rate the term with the number of all
                possible spanning
                    trees multiplied by lambda_2 raised to the power of the down
                    edges, multiplied by lambda_1 raised to the power of the right
                    edges, multiplied by mu raised to the power of the left edges:
                    e.g num_of_spanning_trees * (λ_α^2) * (λ_ο^3) * (μ^2)
            - Move to next combination of edges until all combinations are considered
        - Add to P00_rate the term: μ^(N-T)
        - Multiply P00_rate by the term: μ^(N*M)

    TODO: fix function for case of num_of_servers > 1

    Parameters
    ----------
    lambda_2 : float
    lambda_1 : float
    mu : float
    num_of_servers : int
    threshold : int
    system_capacity : int
    buffer_capacity : int

    Returns
    -------
        The unnormalized rate of state (0,0) (P_00)
    """

    if num_of_servers != 1:
        return "Unable to calculate for cases where number of servers is not 1"

    P00_rate = 0
    for down_edges in np.linspace(
        buffer_capacity * (system_capacity - threshold),
        1,
        buffer_capacity * (system_capacity - threshold),
        dtype=int,
    ):
        for right_edges in range(
            buffer_capacity * (system_capacity - threshold) - down_edges + 1
        ):
            spanning_tree_counter = 0
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
                edges_index = generate_next_permutation_of_edges(
                    edges=edges_index,
                    downs=down_edges,
                    lefts=left_edges,
                    rights=right_edges,
                )
                if edges_index == []:
                    more_trees_exist = False

            P00_rate += (
                spanning_tree_counter
                * lambda_2 ** down_edges
                * lambda_1 ** right_edges
                * mu ** left_edges
            )

    P00_rate += mu ** (system_capacity - threshold)
    P00_rate *= mu ** (system_capacity * buffer_capacity)

    return P00_rate


def get_all_permutations(D, R, L):
    """Given some number of "D"s, some number of "R"s and some number of "L"s,
    this function gets the total number of permutations of an array that consists
    of these "D","R" and "L".

    This can be calculated by: (D+R+L)! / (D! * R! * L!)

    Parameters
    ----------
    D : int
        number of "D"s in the array
    R : int
        number of "R"s in the array
    L : int
        number of "L"s in the array

    Returns
    -------
    int
        total number of permutations
    """
    return np.math.factorial(D + R + L) // (
        np.math.factorial(D) * np.math.factorial(R) * np.math.factorial(L)
    )


def get_permutations_ending_in_R(D, R, L):
    """Given some number of "D"s, some number of "R"s and some number of "L"s,
    this function gets the total number of permutations of an array that consists
    of these "D","R" and "L" and ends in "R".

    This can be calculated by: (D+R+L)! / (D! * (R-1)! * L!)

    Parameters
    ----------
    D : int
        number of "D"s in the array
    R : int
        number of "R"s in the array
    L : int
        number of "L"s in the array

    Returns
    -------
    int
        total number of permutations ending in R
    """
    if R > 0:
        return np.math.factorial(D + R + L - 1) // (
            np.math.factorial(D) * np.math.factorial(R - 1) * np.math.factorial(L)
        )
    else:
        return 0


def get_permutations_ending_in_D_where_any_RL_exists(D, R, L):
    """Given some number of "D"s, some number of "R"s and some number of "L"s,
    this function gets the total number of permutations of an array that consists
    of these "D","R" and "L", ends in "D" and has at least one "R" followed by an
    "L" somewhere.

    This can be calculated by:
        Σ_(i=1)^(min(R,L)) (-1)^(i+1) [(D+R+L-i-1)! / ((D-1)! * (R-i)! * (L-i)! * (i)!)]

    Parameters
    ----------
    D : int
        number of "D"s in the array
    R : int
        number of "R"s in the array
    L : int
        number of "L"s in the array

    Returns
    -------
    int
        total number of permutations ending in "D" with at least one "R" followed
        by an "L"
    """
    max_RL = min(R, L)
    if max_RL > 0:
        sign = -1
        perms = 0
        for num_RL in np.linspace(1, max_RL, max_RL):
            sign *= -1
            perms += (
                sign
                * np.math.factorial(R + D + L - num_RL - 1)
                // (
                    np.math.factorial(D - 1)
                    * np.math.factorial(R - num_RL)
                    * np.math.factorial(L - num_RL)
                    * np.math.factorial(num_RL)
                )
            )
        return perms
    else:
        return 0


def get_permutations_ending_in_L_where_any_RL_exists(D, R, L):
    """Given some number of "D"s, some number of "R"s and some number of "L"s,
    this function gets the total number of permutations of an array that consists
    of these "D","R" and "L", ends in "L" and has at least one "R" followed by an
    "L" somewhere.

    This can be calculated by:
        Σ_(i=1)^(min(R,L-1)) (-1)^(i+1) [(D+R+L-i-1)! / ((D)! * (R-i)! * (L-i-1)! * (i)!)]

    Parameters
    ----------
    D : int
        number of "D"s in the array
    R : int
        number of "R"s in the array
    L : int
        number of "L"s in the array

    Returns
    -------
    int
        total number of permutations ending in "L" with at least one "R" followed
        by an "L" (excluding final "L")
    """
    max_RL = min(R, L - 1)
    if max_RL > 0:
        sign = -1
        perms = 0
        for num_RL in np.linspace(1, max_RL, max_RL):
            sign *= -1
            perms += (
                sign
                * np.math.factorial(R + D + L - num_RL - 1)
                // (
                    np.math.factorial(D)
                    * np.math.factorial(R - num_RL)
                    * np.math.factorial(L - 1 - num_RL)
                    * np.math.factorial(num_RL)
                )
            )
        return perms
    else:
        return 0


def get_permutations_ending_in_RL_where_RL_exists_only_at_the_end(D, R, L):
    """Given some number of "D"s, some number of "R"s and some number of "L"s,
    this function gets the total number of permutations of an array that consists
    of these "D","R" and "L" ends in ["R","L"] and has no "R"'s followed by an
    "L" anywhere else.

    This can be calculated by:
        Σ_(i=1)^(min(R,L)) (-1)^(i+1) [(D+R+L-i-1)! / ((D)! * (R-i)! * (L-i)! * (i-1)!)]

    Parameters
    ----------
    D : int
        number of "D"s in the array
    R : int
        number of "R"s in the array
    L : int
        number of "L"s in the array

    Returns
    -------
    int
        total number of permutations ending in ["R", "L"] with no "R" followed by
        an "L" anywhere else
    """
    max_RL = min(R, L)
    if max_RL > 0:
        sign = -1
        perms = 0
        for num_RL in np.linspace(1, max_RL, max_RL):
            sign *= -1
            num_RL_perms = (
                sign
                * np.math.factorial(R + D + L - num_RL - 1)
                // (
                    np.math.factorial(D)
                    * np.math.factorial(R - num_RL)
                    * np.math.factorial(L - num_RL)
                    * np.math.factorial(num_RL - 1)
                )
            )
            perms += num_RL_perms
        return perms
    else:
        return 0


def get_coefficient(D, R, L):
    """Get the coefficient of the term (lambda_2 ^ D) * (lambda_1 ^ R) * (mu ^ L)
    by using only the values of D, R and L. The function finds all valid spanning
    trees by permuting around the number of D's, R's and L's. The function finds
    all permutations where there is no "R" at the end and there is no "R" followed
    by an "L" anywhere. This is done in the following way:
    - Find the total number of permutations
    - Subtract the permutations ending in "R"
    - Subtract the permutations that end in "D" AND have an "RL" somewhere
    - Subtract the permutations that end in "L" AND have an "RL" somewhere
    - Subtract the permutations that end in "RL" AND do not have an "RL" anywhere else

    Parameters
    ----------
    D : int
        number of "D"s in the array
    R : int
        number of "R"s in the array
    L : int
        number of "L"s in the array

    Returns
    -------
    int
        The coefficient of the term: (lambda_2 ^ D) * (lambda_1 ^ R) * (mu ^ L)
    """
    all_permutations = get_all_permutations(D, R, L)
    permutations_ending_in_R = get_permutations_ending_in_R(D, R, L)
    permutations_ending_in_D_where_any_RL_exists = (
        get_permutations_ending_in_D_where_any_RL_exists(D, R, L)
    )
    permutations_ending_in_L_where_any_RL_exists = (
        get_permutations_ending_in_L_where_any_RL_exists(D, R, L)
    )
    permutations_ending_in_RL_where_RL_exists_only_at_the_end = (
        get_permutations_ending_in_RL_where_RL_exists_only_at_the_end(D, R, L)
    )

    coefficient = all_permutations
    coefficient -= permutations_ending_in_R
    coefficient -= permutations_ending_in_D_where_any_RL_exists
    coefficient -= permutations_ending_in_L_where_any_RL_exists
    coefficient -= permutations_ending_in_RL_where_RL_exists_only_at_the_end

    return coefficient
