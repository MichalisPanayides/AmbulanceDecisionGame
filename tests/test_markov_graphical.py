import numpy as np

from ambulance_game.markov.graphical import (
    reset_L_and_R_in_array,
    find_next_permutation_over,
    find_next_permutation_over_L_and_R,
    generate_next_permutation_of_edges,
    check_permutation_is_valid,
    get_rate_of_state_00_graphically,
    get_all_permutations,
    get_permutations_ending_in_R,
    get_permutations_ending_in_D_where_any_RL_exists,
    get_permutations_ending_in_L_where_any_RL_exists,
    get_permutations_ending_in_RL_where_RL_exists_only_at_the_end,
    get_coefficient,
)

from hypothesis import (
    given,
    settings,
)
from hypothesis.strategies import (
    floats,
    integers,
    booleans,
)


def test_reset_L_and_R_in_array():
    """Test to ensure that the function takes an array with elements "D",
    "L" and "R" and while leaving all "D" elements in the same place, moves
    all "L"s to the leftmost positions and leaves the "R"s to the righmost
    positions.
    """
    array_to_reset = ["R", "D", "D", "R", "D", "L", "L"]
    reset_array = reset_L_and_R_in_array(array_to_reset, 2)
    assert reset_array == ["L", "D", "D", "L", "D", "R", "R"]

    array_to_reset = ["R", "R", "L", "L", "L"]
    reset_array = reset_L_and_R_in_array(array_to_reset, 3)
    assert reset_array == ["L", "L", "L", "R", "R"]


def test_find_next_permutation_over():
    """Test to ensure that function works as expected in all four different cases that it is used.
    - When the array has only "L" and "D" elements
    - When the array has only "R" and "D" elements
    - When the array has "L", "R" and "D" elements
    - When the array has only "L" and "R" elements
    """

    array_to_permute = ["L", "L", "D", "L", "D"]
    permuted_array = find_next_permutation_over(edges=array_to_permute, direction="L")
    assert permuted_array == ["L", "L", "D", "D", "L"]

    array_to_permute = ["R", "R", "D", "D", "R"]
    permuted_array = find_next_permutation_over(edges=array_to_permute, direction="R")
    assert permuted_array == ["R", "D", "R", "R", "D"]

    array_to_permute = ["L", "L", "R", "D", "D"]
    permuted_array = find_next_permutation_over(
        edges=array_to_permute, direction="LR", rights=1
    )
    assert permuted_array == ["L", "L", "D", "R", "D"]

    array_to_permute = ["L", "L", "R"]
    permuted_array = find_next_permutation_over(
        edges=array_to_permute, direction="L", permute_over="R"
    )
    assert permuted_array == ["L", "R", "L"]


def test_find_next_permutation_over_L_and_R():
    """Test to ensure that new permutation of arrays between L and R are
    visited in a circular manner and that no permutation is left behind.
    """
    array_to_permute = ["L", "D", "L", "L", "R", "R"]
    permutation_1 = find_next_permutation_over_L_and_R(edges=array_to_permute)
    assert permutation_1 == ["L", "D", "L", "R", "L", "R"]

    permutation_2 = find_next_permutation_over_L_and_R(edges=permutation_1)
    assert permutation_2 == ["L", "D", "L", "R", "R", "L"]

    permutation_3 = find_next_permutation_over_L_and_R(edges=permutation_2)
    assert permutation_3 == ["L", "D", "R", "L", "L", "R"]

    permutation_4 = find_next_permutation_over_L_and_R(edges=permutation_3)
    assert permutation_4 == ["L", "D", "R", "L", "R", "L"]

    permutation_5 = find_next_permutation_over_L_and_R(edges=permutation_4)
    assert permutation_5 == ["L", "D", "R", "R", "L", "L"]


def test_generate_next_permutation_of_edges():
    """Test to ensure that new permutation of arrays over D, L, R are
    visited in a circular manner and that no permutation is left behind.
    """
    array_to_permute = ["R", "D", "L", "R", "L", "L"]
    permutation_1 = generate_next_permutation_of_edges(
        edges=array_to_permute, downs=1, lefts=3, rights=2
    )
    assert permutation_1 == ["R", "D", "R", "L", "L", "L"]

    permutation_2 = generate_next_permutation_of_edges(
        edges=permutation_1, downs=1, lefts=3, rights=2
    )
    assert permutation_2 == ["D", "L", "L", "L", "R", "R"]

    permutation_3 = generate_next_permutation_of_edges(
        edges=permutation_2, downs=1, lefts=3, rights=2
    )
    assert permutation_3 == ["D", "L", "L", "R", "L", "R"]

    permutation_4 = generate_next_permutation_of_edges(
        edges=permutation_3, downs=1, lefts=3, rights=2
    )
    assert permutation_4 == ["D", "L", "L", "R", "R", "L"]


def test_check_permutation_is_valid():
    """Test that valid permutations return true and that the cases when a permutation is invalid return False"""
    valid_permutation = ["L", "L", "D", "R", "D"]
    assert check_permutation_is_valid(valid_permutation, 1)

    valid_permutation = ["L", "L", "D", "R", "D", "L"]
    assert check_permutation_is_valid(valid_permutation, 2)

    invalid_permutation = ["L", "L", "D", "R"]
    assert not check_permutation_is_valid(invalid_permutation, 1)

    invalid_permutation = ["L", "L", "R", "D", "D", "L"]
    assert not check_permutation_is_valid(invalid_permutation, 2)

    invalid_permutation = ["R", "L", "L", "D", "D", "L"]
    assert not check_permutation_is_valid(invalid_permutation, 1)
    assert not check_permutation_is_valid(invalid_permutation, 2)


def test_get_rate_of_state_00_graphically():
    """Test that for different values of the system capacity the values are as expected.
    Here the values of lambda_a, lambda_o and mu are set to 1 because that way all terms
    are forced to turn into one and the only thing that remains are their coefficients.

    This test basically checks that the sum of all the coefficients (i.e. the total
    number of spanning trees rooted at (0,0) of the model) can be found by the values
    generated by the matrix tree theorem.

    It also checks that for a parking capacity of 2 the equivalent values squaed hold."""

    system_capacity = 1
    matrix_tree_theorem_values = [1, 2, 5, 13, 34, 89]

    for value in matrix_tree_theorem_values:
        P00_rate = get_rate_of_state_00_graphically(
            lambda_a=1,
            lambda_o=1,
            mu=1,
            num_of_servers=1,
            threshold=1,
            system_capacity=system_capacity,
            parking_capacity=1,
        )
        assert P00_rate == value

        P00_rate = get_rate_of_state_00_graphically(
            lambda_a=1,
            lambda_o=1,
            mu=1,
            num_of_servers=1,
            threshold=1,
            system_capacity=system_capacity,
            parking_capacity=2,
        )
        assert P00_rate == value ** 2

        system_capacity += 1


def test_get_all_permutations_examples():
    """Test to ensure that function works as expected for specific examples"""
    assert get_all_permutations(1, 2, 3) == 60
    assert get_all_permutations(2, 5, 2) == 756
    assert get_all_permutations(6, 5, 4) == 630630
    assert get_all_permutations(10, 15, 20) == 10361546974682663760


@given(num=integers(min_value=1, max_value=30))
def test_get_all_permutations_when_two_of_the_inputs_are_1(num):
    """Ensure that function works as expected when two of the inputs are equal to one"""
    T = num + 2
    assert get_all_permutations(num, 1, 1) == (T - 1) * (T)
    assert get_all_permutations(1, num, 1) == (T - 1) * (T)
    assert get_all_permutations(1, 1, num) == (T - 1) * (T)


def test_get_permutations_ending_in_R_examples():
    """Test on specific examples for the function"""
    assert get_permutations_ending_in_R(100, 0, 200) == 0
    assert get_permutations_ending_in_R(2, 3, 4) == 420
    assert get_permutations_ending_in_R(4, 5, 4) == 34650
    assert get_permutations_ending_in_R(7, 8, 9) == 2804596080
    assert get_permutations_ending_in_R(10, 12, 13) == 327314719892160


@given(
    D=integers(min_value=1, max_value=10),
    R=integers(min_value=1, max_value=10),
    L=integers(min_value=1, max_value=10),
)
def test_get_permutations_ending_in_R_equivalence_to_all_permutations(D, R, L):
    """Test that the function is equivalent to the get_all_permutations function with R-1"""
    assert get_permutations_ending_in_R(D, R, L) == get_all_permutations(D, R - 1, L)


def test_get_permutations_ending_in_D_where_any_RL_exists_examples():
    """Test on specific examples for the function"""
    assert get_permutations_ending_in_D_where_any_RL_exists(200, 0, 300) == 0
    assert get_permutations_ending_in_D_where_any_RL_exists(120, 400, 0) == 0
    assert get_permutations_ending_in_D_where_any_RL_exists(1, 1, 1) == 1
    assert get_permutations_ending_in_D_where_any_RL_exists(2, 2, 2) == 21
    assert get_permutations_ending_in_D_where_any_RL_exists(3, 3, 3) == 460
    assert get_permutations_ending_in_D_where_any_RL_exists(6, 5, 4) == 220500


def test_get_permutations_ending_in_L_where_any_RL_exists_examples():
    """Test on specific examples for the function"""
    assert get_permutations_ending_in_L_where_any_RL_exists(300, 0, 1000) == 0
    assert get_permutations_ending_in_L_where_any_RL_exists(201, 150, 0) == 0
    assert get_permutations_ending_in_L_where_any_RL_exists(100, 200, 1) == 0
    assert get_permutations_ending_in_L_where_any_RL_exists(2, 2, 2) == 12
    assert get_permutations_ending_in_L_where_any_RL_exists(3, 3, 3) == 360
    assert get_permutations_ending_in_L_where_any_RL_exists(6, 5, 4) == 129360


def test_get_permutations_ending_in_RL_where_RL_exists_only_at_the_end_examples():
    """Test on specific examples for the function"""
    get_permutations_ending_in_RL_where_RL_exists_only_at_the_end(100, 0, 200) == 0
    get_permutations_ending_in_RL_where_RL_exists_only_at_the_end(100, 200, 0) == 0
    get_permutations_ending_in_RL_where_RL_exists_only_at_the_end(1, 1, 1) == 1
    get_permutations_ending_in_RL_where_RL_exists_only_at_the_end(1, 2, 3) == 6
    get_permutations_ending_in_RL_where_RL_exists_only_at_the_end(3, 4, 2) == 80
    get_permutations_ending_in_RL_where_RL_exists_only_at_the_end(5, 6, 4) == 14112


def test_get_coefficient_known_examples():
    """Test that the function works as expected for known examples"""
    known_examples = [
        [1, 1, 1],
        [2, 1, 1],
        [1, 2, 1],
        [1, 1, 2],
        [3, 1, 1],
        [2, 2, 1],
        [2, 1, 2],
        [1, 3, 1],
        [1, 2, 2],
        [1, 1, 3],
        [4, 1, 1],
        [3, 2, 1],
        [3, 1, 2],
        [2, 3, 1],
        [2, 2, 2],
        [2, 1, 3],
        [1, 4, 1],
        [1, 3, 2],
        [1, 2, 3],
        [1, 1, 4],
        [5, 1, 1],
        [4, 2, 1],
        [4, 1, 2],
        [3, 3, 1],
        [3, 2, 2],
        [3, 1, 3],
        [2, 4, 1],
        [2, 3, 2],
        [2, 2, 3],
        [2, 1, 4],
        [1, 5, 1],
        [1, 4, 2],
        [1, 3, 3],
        [1, 2, 4],
        [1, 1, 5],
        [6, 1, 1],
        [5, 2, 1],
        [5, 1, 2],
        [4, 3, 1],
        [4, 2, 2],
        [4, 1, 3],
        [3, 4, 1],
        [3, 3, 2],
        [3, 2, 3],
        [3, 1, 4],
        [2, 5, 1],
        [2, 4, 2],
        [2, 3, 3],
        [2, 2, 4],
        [2, 1, 5],
        [1, 6, 1],
        [1, 5, 2],
        [1, 4, 3],
        [1, 3, 4],
        [1, 2, 5],
        [1, 1, 6],
    ]

    known_coefficients = [
        2,
        6,
        2,
        3,
        12,
        9,
        12,
        2,
        3,
        4,
        20,
        24,
        30,
        12,
        18,
        20,
        2,
        3,
        4,
        5,
        30,
        50,
        60,
        40,
        60,
        60,
        15,
        24,
        30,
        30,
        2,
        3,
        4,
        5,
        6,
        42,
        90,
        105,
        100,
        150,
        140,
        60,
        100,
        120,
        105,
        18,
        30,
        40,
        45,
        42,
        2,
        3,
        4,
        5,
        6,
        7,
    ]
    for example, coefficient in zip(known_examples, known_coefficients):
        assert get_coefficient(example[0], example[1], example[2]) == coefficient


@given(
    D=integers(min_value=1, max_value=50),
    R=integers(min_value=1, max_value=50),
    L=integers(min_value=1, max_value=50),
)
def test_get_coefficient_known_scenarios(D, R, L):
    """Test special cases of the get_coefficient() function"""
    assert get_coefficient(1, R, L) == L + 1
    assert get_coefficient(D, 0, 0) == 1
    assert get_coefficient(0, 0, L) == 1
    assert get_coefficient(D, 1, 0) == D
    assert get_coefficient(D, 0, 1) == D + 1
