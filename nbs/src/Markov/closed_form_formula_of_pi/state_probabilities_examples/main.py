import ambulance_game as abg
import numpy as np
import sympy as sym
from sympy.abc import a, b, c, d, e, f, g, h, i, j


def get_symbolic_pi(num_of_servers, threshold, system_capacity, buffer_capacity):
    Q_sym = abg.markov.get_symbolic_transition_matrix(
        num_of_servers=num_of_servers,
        threshold=threshold,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
    )
    dimension = Q_sym.shape[0]
    if dimension > 7:
        return "Capacity of 6 exceeded"
    M_sym = sym.Matrix([Q_sym.transpose()[:-1, :], sym.ones(1, dimension)])
    b_sym = sym.Matrix([sym.zeros(dimension - 1, 1), [1]])
    system = M_sym.col_insert(dimension, b_sym)
    sol = sym.solve_linear_system_LU(system, [a, b, c, d, e, f, g])
    return sol


def get_symbolic_state_probabilities_1222():
    num_of_servers = 1
    threshold = 2
    system_capacity = 2
    buffer_capacity = 2

    sym_pi_1222 = get_symbolic_pi(
        num_of_servers=num_of_servers,
        threshold=threshold,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
    )
    all_states_1222 = abg.markov.build_states(
        threshold=threshold,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
    )

    sym_state_probs_1222 = [0 for _ in range(len(all_states_1222))]
    sym_state_probs_1222[0] = sym.factor(sym_pi_1222[a])  # (0,0)
    sym_state_probs_1222[1] = sym.factor(sym_pi_1222[b])  # (0,1)
    sym_state_probs_1222[2] = sym.factor(sym_pi_1222[c])  # (1,1)
    sym_state_probs_1222[3] = sym.factor(sym_pi_1222[d])  # (0,2)
    sym_state_probs_1222[4] = sym.factor(sym_pi_1222[e])  # (1,2)

    sym_state_recursive_ratios_1222 = sym.zeros(
        buffer_capacity + 1, system_capacity + 1
    )
    sym_state_recursive_ratios_1222[0, 0] = 1
    sym_state_recursive_ratios_1222[0, 1] = sym.factor(
        sym_state_probs_1222[1] / sym_state_probs_1222[0]
    )  # (0,0) -> (0,1)
    sym_state_recursive_ratios_1222[0, 2] = sym.factor(
        sym_state_probs_1222[2] / sym_state_probs_1222[1]
    )  # (0,1) -> (1,1)
    sym_state_recursive_ratios_1222[1, 2] = sym.factor(
        sym_state_probs_1222[3] / sym_state_probs_1222[2]
    )  # (0,1) -> (0,2)
    sym_state_recursive_ratios_1222[2, 2] = sym.factor(
        sym_state_probs_1222[4] / sym_state_probs_1222[3]
    )  # (0,2) -> (1,2)

    return sym_state_probs_1222, sym_state_recursive_ratios_1222


def get_symbolic_state_probabilities_1121():
    num_of_servers = 1
    threshold = 1
    system_capacity = 2
    buffer_capacity = 1

    all_states_1121 = abg.markov.build_states(
        threshold=threshold,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
    )
    sym_pi_1121 = get_symbolic_pi(
        num_of_servers=num_of_servers,
        threshold=threshold,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
    )
    sym_state_probs_1121 = [0 for _ in range(len(all_states_1121))]

    sym_state_probs_1121[0] = sym.factor(sym_pi_1121[a])  # (0,0)
    sym_state_probs_1121[1] = sym.factor(sym_pi_1121[b])  # (0,1)
    sym_state_probs_1121[2] = sym.factor(sym_pi_1121[c])  # (1,1)
    sym_state_probs_1121[3] = sym.factor(sym_pi_1121[d])  # (0,2)
    sym_state_probs_1121[4] = sym.factor(sym_pi_1121[e])  # (1,2)

    sym_state_recursive_ratios_1121 = sym.zeros(
        buffer_capacity + 1, system_capacity + 1
    )
    sym_state_recursive_ratios_1121[0, 0] = 1
    sym_state_recursive_ratios_1121[0, 1] = sym.factor(
        sym_state_probs_1121[1] / sym_state_probs_1121[0]
    )  # (0,0) -> (0,1)
    sym_state_recursive_ratios_1121[1, 1] = sym.factor(
        sym_state_probs_1121[2] / sym_state_probs_1121[1]
    )  # (0,1) -> (1,1)
    sym_state_recursive_ratios_1121[0, 2] = sym.factor(
        sym_state_probs_1121[3] / sym_state_probs_1121[1]
    )  # (0,1) -> (0,2)
    sym_state_recursive_ratios_1121[1, 2] = sym.factor(
        sym_state_probs_1121[4] / sym_state_probs_1121[3]
    )  # (0,2) -> (1,2)

    sym_state_recursive_ratios_right_1121 = sym_state_recursive_ratios_1121.copy()
    sym_state_recursive_ratios_right_1121[1, 2] = sym.factor(
        sym_state_probs_1121[4] / sym_state_probs_1121[2]
    )  # (1,1) -> (1,2)

    sym_state_recursive_ratios_P0_1121 = sym.zeros(
        buffer_capacity + 1, system_capacity + 1
    )
    sym_state_recursive_ratios_P0_1121[0, 0] = 1
    sym_state_recursive_ratios_P0_1121[0, 1] = sym.factor(
        sym_state_probs_1121[1] / sym_state_probs_1121[0]
    )  # (0,0) -> (0,1)
    sym_state_recursive_ratios_P0_1121[1, 1] = sym.factor(
        sym_state_probs_1121[2] / sym_state_probs_1121[0]
    )  # (0,0) -> (1,1)
    sym_state_recursive_ratios_P0_1121[0, 2] = sym.factor(
        sym_state_probs_1121[3] / sym_state_probs_1121[0]
    )  # (0,0) -> (0,2)
    sym_state_recursive_ratios_P0_1121[1, 2] = sym.factor(
        sym_state_probs_1121[4] / sym_state_probs_1121[0]
    )  # (0,0) -> (1,2)

    return (
        sym_state_probs_1121,
        sym_state_recursive_ratios_1121,
        sym_state_recursive_ratios_right_1121,
        sym_state_recursive_ratios_P0_1121,
    )


def get_symbolic_state_probabilities_1122():
    # num_of_servers = 1
    threshold = 1
    system_capacity = 2
    buffer_capacity = 2

    all_states_1122 = abg.markov.build_states(
        threshold=threshold,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
    )
    sym_state_probs_1122 = [0 for _ in range(len(all_states_1122))]

    sym_Lambda = sym.symbols("Lambda")
    sym_lambda_1 = sym.symbols("lambda_1")
    sym_lambda_2 = sym.symbols("lambda_2")
    sym_mu = sym.symbols("mu")

    sym_state_probs_1122[0] = (
        (sym_mu ** 6)
        + 2 * (sym_lambda_2) * (sym_mu ** 5)
        + (sym_lambda_2 ** 2) * (sym_mu ** 4)
    )  # (0,0)
    sym_state_probs_1122[1] = (sym_Lambda * sym_mu ** 3) * (
        sym_mu ** 2 + 2 * sym_mu * sym_lambda_2 + sym_lambda_2 ** 2
    )  # (0,1)
    sym_state_probs_1122[2] = (sym_Lambda * sym_lambda_2 * sym_mu ** 2) * (
        sym_lambda_2 ** 2
        + sym_lambda_2 * sym_lambda_1
        + sym_lambda_1 * sym_mu
        + sym_mu ** 2
        + 2 * sym_lambda_2 * sym_mu
    )  # (1,1)
    sym_state_probs_1122[3] = (sym_Lambda * sym_lambda_2 ** 2 * sym_mu) * (
        sym_lambda_2 ** 2
        + 2 * sym_lambda_1 * sym_lambda_2
        + 3 * sym_lambda_1 * sym_mu
        + sym_mu ** 2
        + 2 * sym_lambda_2 * sym_mu
        + sym_lambda_1 ** 2
    )  # (2,1)
    sym_state_probs_1122[4] = (sym_Lambda * sym_lambda_1 * sym_mu ** 3) * (
        sym_lambda_2 + sym_mu
    )  # (0,2)
    sym_state_probs_1122[5] = (
        sym_Lambda * sym_lambda_1 * sym_lambda_2 * sym_mu ** 2
    ) * (
        2 * sym_mu + sym_lambda_1 + sym_lambda_2
    )  # (1,2)
    sym_state_probs_1122[6] = (sym_Lambda * sym_lambda_1 * sym_lambda_2 ** 2) * (
        sym_lambda_1 ** 2
        + 4 * sym_lambda_1 * sym_mu
        + 2 * sym_lambda_1 * sym_lambda_2
        + 3 * sym_mu ** 2
        + sym_lambda_2 ** 2
        + 3 * sym_lambda_2 * sym_mu
    )  # (2,2)

    total_1122 = np.sum(sym_state_probs_1122)
    sym_state_probs_1122 = [i / total_1122 for i in sym_state_probs_1122]

    sym_state_recursive_ratios_1122 = sym.zeros(
        buffer_capacity + 1, system_capacity + 1
    )
    sym_state_recursive_ratios_1122[0, 0] = 1
    sym_state_recursive_ratios_1122[0, 1] = sym.factor(
        sym_state_probs_1122[1] / sym_state_probs_1122[0]
    )  # (0,0) -> (0,1)
    sym_state_recursive_ratios_1122[1, 1] = sym.factor(
        sym_state_probs_1122[2] / sym_state_probs_1122[1]
    )  # (0,1) -> (1,1)
    sym_state_recursive_ratios_1122[2, 1] = sym.factor(
        sym_state_probs_1122[3] / sym_state_probs_1122[2]
    )  # (1,1) -> (2,1)

    sym_state_recursive_ratios_1122[0, 2] = sym.factor(
        sym_state_probs_1122[4] / sym_state_probs_1122[1]
    )  # (0,1) -> (0,2)
    sym_state_recursive_ratios_1122[1, 2] = sym.factor(
        sym_state_probs_1122[5] / sym_state_probs_1122[4]
    )  # (0,2) -> (1,2)
    sym_state_recursive_ratios_1122[2, 2] = sym.factor(
        sym_state_probs_1122[6] / sym_state_probs_1122[5]
    )  # (1,2) -> (2,2)

    sym_state_recursive_ratios_right_1122 = sym_state_recursive_ratios_1122.copy()
    sym_state_recursive_ratios_right_1122[1, 2] = sym.factor(
        sym_state_probs_1122[5] / sym_state_probs_1122[2]
    )  # (1,1) -> (1,2)
    sym_state_recursive_ratios_right_1122[2, 2] = sym.factor(
        sym_state_probs_1122[6] / sym_state_probs_1122[3]
    )  # (2,1) -> (2,2)

    sym_state_recursive_ratios_P0_1122 = sym.zeros(
        buffer_capacity + 1, system_capacity + 1
    )
    sym_state_recursive_ratios_P0_1122[0, 0] = 1
    sym_state_recursive_ratios_P0_1122[0, 1] = sym.factor(
        sym_state_probs_1122[1] / sym_state_probs_1122[0]
    )  # (0,0) -> (0,1)
    sym_state_recursive_ratios_P0_1122[1, 1] = sym.factor(
        sym_state_probs_1122[2] / sym_state_probs_1122[0]
    )  # (0,0) -> (1,1)
    sym_state_recursive_ratios_P0_1122[2, 1] = sym.factor(
        sym_state_probs_1122[3] / sym_state_probs_1122[0]
    )  # (0,0) -> (2,1)

    sym_state_recursive_ratios_P0_1122[0, 2] = sym.factor(
        sym_state_probs_1122[4] / sym_state_probs_1122[0]
    )  # (0,0) -> (0,2)
    sym_state_recursive_ratios_P0_1122[1, 2] = sym.factor(
        sym_state_probs_1122[5] / sym_state_probs_1122[0]
    )  # (0,0) -> (1,2)
    sym_state_recursive_ratios_P0_1122[2, 2] = sym.factor(
        sym_state_probs_1122[6] / sym_state_probs_1122[0]
    )  # (0,0) -> (2,2)

    return (
        sym_state_probs_1122,
        sym_state_recursive_ratios_1122,
        sym_state_recursive_ratios_right_1122,
        sym_state_recursive_ratios_P0_1122,
    )


def get_symbolic_state_probabilities_1123():
    num_of_servers = 1
    threshold = 1
    system_capacity = 2
    buffer_capacity = 3

    Q_sym_1123 = abg.markov.get_symbolic_transition_matrix(
        num_of_servers, threshold, system_capacity, buffer_capacity
    )

    p00, p01, p11, p21, p31, p02, p12, p22, p32 = sym.symbols(
        "p00, p01, p11, p21, p31, p02, p12, p22, p32"
    )
    pi_1123 = sym.Matrix([p00, p01, p11, p21, p31, p02, p12, p22, p32])
    dimension_1123 = Q_sym_1123.shape[0]

    M_sym_1123 = sym.Matrix(
        [Q_sym_1123.transpose()[:-1, :], sym.ones(1, dimension_1123)]
    )
    sym_diff_equations_1123 = M_sym_1123 @ pi_1123

    b_sym_1123 = sym.Matrix([sym.zeros(dimension_1123 - 1, 1), [1]])

    eq0_1123 = sym.Eq(sym_diff_equations_1123[0], b_sym_1123[0])
    eq1_1123 = sym.Eq(sym_diff_equations_1123[1], b_sym_1123[1])
    eq2_1123 = sym.Eq(sym_diff_equations_1123[2], b_sym_1123[2])
    eq3_1123 = sym.Eq(sym_diff_equations_1123[3], b_sym_1123[3])
    eq4_1123 = sym.Eq(sym_diff_equations_1123[4], b_sym_1123[4])
    eq5_1123 = sym.Eq(sym_diff_equations_1123[5], b_sym_1123[5])
    eq6_1123 = sym.Eq(sym_diff_equations_1123[6], b_sym_1123[6])
    eq7_1123 = sym.Eq(sym_diff_equations_1123[7], b_sym_1123[7])
    eq8_1123 = sym.Eq(sym_diff_equations_1123[8], b_sym_1123[8])

    sym_state_probs_1123 = sym.solve(
        [
            eq0_1123,
            eq1_1123,
            eq2_1123,
            eq3_1123,
            eq4_1123,
            eq5_1123,
            eq6_1123,
            eq7_1123,
            eq8_1123,
        ],
        (p00, p01, p11, p21, p31, p02, p12, p22, p32),
    )

    sym_state_recursive_ratios_1123 = sym.zeros(
        buffer_capacity + 1, system_capacity + 1
    )
    sym_state_recursive_ratios_1123[0, 0] = 1
    sym_state_recursive_ratios_1123[0, 1] = sym.factor(
        sym_state_probs_1123[p01] / sym_state_probs_1123[p00]
    )  # (0,0) -> (0,1)
    sym_state_recursive_ratios_1123[1, 1] = sym.factor(
        sym_state_probs_1123[p11] / sym_state_probs_1123[p01]
    )  # (0,1) -> (1,1)
    sym_state_recursive_ratios_1123[2, 1] = sym.factor(
        sym_state_probs_1123[p21] / sym_state_probs_1123[p11]
    )  # (1,1) -> (2,1)
    sym_state_recursive_ratios_1123[3, 1] = sym.factor(
        sym_state_probs_1123[p31] / sym_state_probs_1123[p21]
    )  # (2,1) -> (3,1)
    sym_state_recursive_ratios_1123[0, 2] = sym.factor(
        sym_state_probs_1123[p02] / sym_state_probs_1123[p01]
    )  # (0,1) -> (0,2)
    sym_state_recursive_ratios_1123[1, 2] = sym.factor(
        sym_state_probs_1123[p12] / sym_state_probs_1123[p02]
    )  # (0,2) -> (1,2)
    sym_state_recursive_ratios_1123[2, 2] = sym.factor(
        sym_state_probs_1123[p22] / sym_state_probs_1123[p12]
    )  # (1,2) -> (2,2)
    sym_state_recursive_ratios_1123[2, 2] = sym.factor(
        sym_state_probs_1123[p32] / sym_state_probs_1123[p22]
    )  # (2,2) -> (3,2)

    sym_state_recursive_ratios_right_1123 = sym_state_recursive_ratios_1123.copy()
    sym_state_recursive_ratios_right_1123[1, 2] = sym.factor(
        sym_state_probs_1123[p12] / sym_state_probs_1123[p11]
    )  # (1,1) -> (1,2)
    sym_state_recursive_ratios_right_1123[2, 2] = sym.factor(
        sym_state_probs_1123[p22] / sym_state_probs_1123[p21]
    )  # (2,1) -> (2,2)
    sym_state_recursive_ratios_right_1123[3, 2] = sym.factor(
        sym_state_probs_1123[p32] / sym_state_probs_1123[p22]
    )  # (2,2) -> (3,2)

    sym_state_recursive_ratios_P0_1123 = sym.zeros(
        buffer_capacity + 1, system_capacity + 1
    )
    sym_state_recursive_ratios_P0_1123[0, 0] = 1
    sym_state_recursive_ratios_P0_1123[0, 1] = sym.factor(
        sym_state_probs_1123[p01] / sym_state_probs_1123[p00]
    )  # (0,0) -> (0,1)
    sym_state_recursive_ratios_P0_1123[1, 1] = sym.factor(
        sym_state_probs_1123[p11] / sym_state_probs_1123[p00]
    )  # (0,0) -> (1,1)
    sym_state_recursive_ratios_P0_1123[2, 1] = sym.factor(
        sym_state_probs_1123[p21] / sym_state_probs_1123[p00]
    )  # (0,0) -> (2,1)
    sym_state_recursive_ratios_P0_1123[3, 1] = sym.factor(
        sym_state_probs_1123[p31] / sym_state_probs_1123[p00]
    )  # (0,0) -> (3,1)
    sym_state_recursive_ratios_P0_1123[0, 2] = sym.factor(
        sym_state_probs_1123[p02] / sym_state_probs_1123[p00]
    )  # (0,0) -> (0,2)
    sym_state_recursive_ratios_P0_1123[1, 2] = sym.factor(
        sym_state_probs_1123[p12] / sym_state_probs_1123[p00]
    )  # (0,0) -> (1,2)
    sym_state_recursive_ratios_P0_1123[2, 2] = sym.factor(
        sym_state_probs_1123[p22] / sym_state_probs_1123[p00]
    )  # (0,0) -> (2,2)
    sym_state_recursive_ratios_P0_1123[3, 2] = sym.factor(
        sym_state_probs_1123[p32] / sym_state_probs_1123[p00]
    )  # (0,0) -> (3,2)

    return (
        sym_state_probs_1123,
        sym_state_recursive_ratios_1123,
        sym_state_recursive_ratios_right_1123,
        sym_state_recursive_ratios_P0_1123,
    )


def get_symbolic_state_probabilities_1341():
    # num_of_servers = 1
    threshold = 3
    system_capacity = 4
    buffer_capacity = 1

    all_states_1341 = abg.markov.build_states(
        threshold=threshold,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
    )
    sym_state_probs_1341 = [0 for _ in range(len(all_states_1341))]

    sym_Lambda = sym.symbols("Lambda")
    sym_lambda_1 = sym.symbols("lambda_1")
    sym_lambda_2 = sym.symbols("lambda_2")
    sym_mu = sym.symbols("mu")

    sym_state_probs_1341[0] = (sym_lambda_2) * (sym_mu ** 5) + (sym_mu ** 6)  # (0,0)
    sym_state_probs_1341[1] = sym_Lambda * sym_lambda_2 * (sym_mu ** 4) + sym_Lambda * (
        sym_mu ** 5
    )  # (0,1)
    sym_state_probs_1341[2] = (sym_Lambda ** 2) * sym_lambda_2 * (sym_mu ** 3) + (
        sym_Lambda ** 2
    ) * (
        sym_mu ** 4
    )  # (0,2)
    sym_state_probs_1341[3] = (sym_Lambda ** 3) * sym_lambda_2 * (sym_mu ** 2) + (
        sym_Lambda ** 3
    ) * (
        sym_mu ** 3
    )  # (0,3)
    sym_state_probs_1341[4] = (
        (sym_Lambda ** 3) * sym_lambda_1 * sym_lambda_2 * sym_mu
        + (sym_Lambda ** 3) * sym_lambda_2 * (sym_mu ** 2)
        + (sym_Lambda ** 3) * sym_lambda_2 * sym_lambda_2 * sym_mu
    )  # (1,3)
    sym_state_probs_1341[5] = (sym_Lambda ** 3) * sym_lambda_1 * (sym_mu ** 2)  # (0,4)
    sym_state_probs_1341[6] = (
        (sym_Lambda ** 3) * (sym_lambda_1 ** 2) * sym_lambda_2
        + (sym_Lambda ** 3) * sym_lambda_1 * (sym_lambda_2 ** 2)
        + 2 * (sym_Lambda ** 3) * sym_lambda_1 * sym_lambda_2 * sym_mu
    )  # (1,4)

    total_1341 = np.sum(sym_state_probs_1341)
    sym_state_probs_1341 = [i / total_1341 for i in sym_state_probs_1341]

    sym_state_recursive_ratios_1341 = sym.zeros(
        buffer_capacity + 1, system_capacity + 1
    )
    sym_state_recursive_ratios_1341[0, 0] = 1
    sym_state_recursive_ratios_1341[0, 1] = sym.factor(
        sym_state_probs_1341[1] / sym_state_probs_1341[0]
    )  # (0,0) -> (0,1)
    sym_state_recursive_ratios_1341[0, 2] = sym.factor(
        sym_state_probs_1341[2] / sym_state_probs_1341[1]
    )  # (0,1) -> (0,2)
    sym_state_recursive_ratios_1341[0, 3] = sym.factor(
        sym_state_probs_1341[3] / sym_state_probs_1341[2]
    )  # (0,2) -> (0,3)
    sym_state_recursive_ratios_1341[0, 4] = sym.factor(
        sym_state_probs_1341[5] / sym_state_probs_1341[3]
    )  # (0,3) -> (0,4)

    sym_state_recursive_ratios_1341[1, 3] = sym.factor(
        sym_state_probs_1341[4] / sym_state_probs_1341[3]
    )  # (0,3) -> (1,3)
    sym_state_recursive_ratios_1341[1, 4] = sym.factor(
        sym_state_probs_1341[6] / sym_state_probs_1341[5]
    )  # (0,4) -> (1,4)

    sym_state_recursive_ratios_right_1341 = sym_state_recursive_ratios_1341.copy()
    sym_state_recursive_ratios_right_1341[1, 4] = sym.factor(
        sym_state_probs_1341[6] / sym_state_probs_1341[4]
    )  # (1,3) -> (1,4)

    sym_state_recursive_ratios_P0_1341 = sym.zeros(
        buffer_capacity + 1, system_capacity + 1
    )
    sym_state_recursive_ratios_P0_1341[0, 0] = 1
    sym_state_recursive_ratios_P0_1341[0, 1] = sym.factor(
        sym_state_probs_1341[1] / sym_state_probs_1341[0]
    )  # (0,0) -> (0,1)
    sym_state_recursive_ratios_P0_1341[0, 2] = sym.factor(
        sym_state_probs_1341[2] / sym_state_probs_1341[0]
    )  # (0,0) -> (0,2)
    sym_state_recursive_ratios_P0_1341[0, 3] = sym.factor(
        sym_state_probs_1341[3] / sym_state_probs_1341[0]
    )  # (0,0) -> (0,3)

    sym_state_recursive_ratios_P0_1341[1, 3] = sym.factor(
        sym_state_probs_1341[4] / sym_state_probs_1341[0]
    )  # (0,0) -> (1,3)
    sym_state_recursive_ratios_P0_1341[0, 4] = sym.factor(
        sym_state_probs_1341[5] / sym_state_probs_1341[0]
    )  # (0,0) -> (0,4)
    sym_state_recursive_ratios_P0_1341[1, 4] = sym.factor(
        sym_state_probs_1341[6] / sym_state_probs_1341[0]
    )  # (0,0) -> (1,4)

    return (
        sym_state_probs_1341,
        sym_state_recursive_ratios_1341,
        sym_state_recursive_ratios_right_1341,
        sym_state_recursive_ratios_P0_1341,
    )


def get_symbolic_state_probabilities_1131():
    # num_of_servers = 1
    threshold = 1
    system_capacity = 3
    buffer_capacity = 1

    all_states_1131 = abg.markov.build_states(
        threshold=threshold,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
    )
    sym_state_probs_1131 = [0 for _ in range(len(all_states_1131))]

    sym_Lambda = sym.symbols("Lambda")
    sym_lambda_1 = sym.symbols("lambda_1")
    sym_lambda_2 = sym.symbols("lambda_2")
    sym_mu = sym.symbols("mu")

    # (0,0)
    sym_state_probs_1131[0] = (
        (sym_mu ** 6)
        + 2 * (sym_lambda_2 * (sym_mu ** 5))
        + ((sym_lambda_2 ** 2) * (sym_mu ** 4))
        + (sym_lambda_1 * sym_lambda_2 * (sym_mu ** 4))
    )
    # (0,1)
    sym_state_probs_1131[1] = sym_state_probs_1131[0] * sym_Lambda / sym_mu
    # (1,1)
    sym_state_probs_1131[2] = (
        (sym_Lambda * (sym_lambda_1 ** 2) * sym_lambda_2 * (sym_mu ** 2))
        + (sym_Lambda * sym_lambda_2 * sym_lambda_1 * (sym_mu ** 3))
        + 2 * (sym_Lambda * sym_lambda_1 * (sym_lambda_2 ** 2) * (sym_mu ** 2))
        + 2 * (sym_Lambda * (sym_lambda_2 ** 2) * (sym_mu ** 3))
        + (sym_Lambda * (sym_lambda_2 ** 3) * (sym_mu ** 2))
        + (sym_Lambda * sym_lambda_2 * (sym_mu ** 4))
    )
    # (0,2)
    sym_state_probs_1131[3] = (
        sym_Lambda * sym_lambda_1 * sym_mu ** 3 * (sym_lambda_2 + sym_mu)
    )
    # (1,2)
    sym_state_probs_1131[4] = (sym_Lambda * sym_lambda_2 * sym_lambda_1 * sym_mu) * (
        (sym_lambda_2 ** 2)
        + 2 * sym_lambda_2 * sym_lambda_1
        + 3 * sym_lambda_2 * sym_mu
        + (sym_lambda_1 ** 2)
        + 2 * sym_lambda_1 * sym_mu
        + 2 * (sym_mu ** 2)
    )
    # (0,3)
    sym_state_probs_1131[5] = sym_Lambda * (sym_lambda_1 ** 2) * (sym_mu ** 3)
    # (1,3)
    sym_state_probs_1131[6] = (sym_Lambda * sym_lambda_2 * (sym_lambda_1 ** 2)) * (
        (sym_lambda_2 ** 2)
        + 2 * sym_lambda_2 * sym_lambda_1
        + 3 * sym_lambda_2 * sym_mu
        + (sym_lambda_1 ** 2)
        + 2 * sym_lambda_1 * sym_mu
        + 3 * (sym_mu ** 2)
    )

    denominator = (
        sym_Lambda * sym_lambda_2 ** 3 * sym_lambda_1 ** 2
        + sym_Lambda * sym_lambda_2 ** 3 * sym_lambda_1 * sym_mu
        + sym_Lambda * sym_lambda_2 ** 3 * sym_mu ** 2
        + 2 * sym_Lambda * sym_lambda_2 ** 2 * sym_lambda_1 ** 3
        + 5 * sym_Lambda * sym_lambda_2 ** 2 * sym_lambda_1 ** 2 * sym_mu
        + 5 * sym_Lambda * sym_lambda_2 ** 2 * sym_lambda_1 * sym_mu ** 2
        + 3 * sym_Lambda * sym_lambda_2 ** 2 * sym_mu ** 3
        + sym_Lambda * sym_lambda_2 * sym_lambda_1 ** 4
        + 3 * sym_Lambda * sym_lambda_2 * sym_lambda_1 ** 3 * sym_mu
        + 6 * sym_Lambda * sym_lambda_2 * sym_lambda_1 ** 2 * sym_mu ** 2
        + 5 * sym_Lambda * sym_lambda_2 * sym_lambda_1 * sym_mu ** 3
        + 3 * sym_Lambda * sym_lambda_2 * sym_mu ** 4
        + sym_Lambda * sym_lambda_1 ** 2 * sym_mu ** 3
        + sym_Lambda * sym_lambda_1 * sym_mu ** 4
        + sym_Lambda * sym_mu ** 5
        + sym_lambda_2 ** 2 * sym_mu ** 4
        + sym_lambda_2 * sym_lambda_1 * sym_mu ** 4
        + 2 * sym_lambda_2 * sym_mu ** 5
        + sym_mu ** 6
    )

    sym_state_probs_1131 = [i / denominator for i in sym_state_probs_1131]

    sym_state_recursive_ratios_1131 = sym.zeros(
        buffer_capacity + 1, system_capacity + 1
    )
    sym_state_recursive_ratios_1131[0, 0] = 1
    sym_state_recursive_ratios_1131[0, 1] = sym.factor(
        sym_state_probs_1131[1] / sym_state_probs_1131[0]
    )  # (0,0) -> (0,1)
    sym_state_recursive_ratios_1131[1, 1] = sym.factor(
        sym_state_probs_1131[2] / sym_state_probs_1131[1]
    )  # (0,1) -> (1,1)
    sym_state_recursive_ratios_1131[0, 2] = sym.factor(
        sym_state_probs_1131[3] / sym_state_probs_1131[1]
    )  # (0,1) -> (0,2)
    sym_state_recursive_ratios_1131[1, 2] = sym.factor(
        sym_state_probs_1131[4] / sym_state_probs_1131[3]
    )  # (0,2) -> (1,2)
    sym_state_recursive_ratios_1131[0, 3] = sym.factor(
        sym_state_probs_1131[5] / sym_state_probs_1131[3]
    )  # (0,2) -> (0,3)
    sym_state_recursive_ratios_1131[1, 3] = sym.factor(
        sym_state_probs_1131[6] / sym_state_probs_1131[5]
    )  # (0,3) -> (1,3)

    sym_state_recursive_ratios_right_1131 = sym_state_recursive_ratios_1131.copy()
    sym_state_recursive_ratios_right_1131[1, 2] = sym.factor(
        sym_state_probs_1131[4] / sym_state_probs_1131[2]
    )  # (1,1) -> (1,2)
    sym_state_recursive_ratios_right_1131[1, 3] = sym.factor(
        sym_state_probs_1131[6] / sym_state_probs_1131[4]
    )  # (1,2) -> (1,3)

    sym_state_recursive_ratios_P0_1131 = sym.zeros(
        buffer_capacity + 1, system_capacity + 1
    )
    sym_state_recursive_ratios_P0_1131[0, 0] = 1
    sym_state_recursive_ratios_P0_1131[0, 1] = sym.factor(
        sym_state_probs_1131[1] / sym_state_probs_1131[0]
    )  # (0,0) -> (0,1)
    sym_state_recursive_ratios_P0_1131[1, 1] = sym.factor(
        sym_state_probs_1131[2] / sym_state_probs_1131[0]
    )  # (0,0) -> (1,1)
    sym_state_recursive_ratios_P0_1131[0, 2] = sym.factor(
        sym_state_probs_1131[3] / sym_state_probs_1131[0]
    )  # (0,0) -> (0,2)
    sym_state_recursive_ratios_P0_1131[1, 2] = sym.factor(
        sym_state_probs_1131[4] / sym_state_probs_1131[0]
    )  # (0,0) -> (1,2)
    sym_state_recursive_ratios_P0_1131[0, 3] = sym.factor(
        sym_state_probs_1131[5] / sym_state_probs_1131[0]
    )  # (0,0) -> (0,3)
    sym_state_recursive_ratios_P0_1131[1, 3] = sym.factor(
        sym_state_probs_1131[6] / sym_state_probs_1131[0]
    )  # (0,0) -> (1,3)

    return (
        sym_state_probs_1131,
        sym_state_recursive_ratios_1131,
        sym_state_recursive_ratios_right_1131,
        sym_state_recursive_ratios_P0_1131,
    )


def get_symbolic_state_probabilities_1132():
    num_of_servers = 1
    threshold = 1
    system_capacity = 3
    buffer_capacity = 2

    Q_sym_1132 = abg.markov.get_symbolic_transition_matrix(
        num_of_servers, threshold, system_capacity, buffer_capacity
    )

    p00, p01, p11, p21, p02, p12, p22, p03, p13, p23 = sym.symbols(
        "p00, p01, p11, p21, p02, p12, p22, p03, p13, p23"
    )
    pi_1132 = sym.Matrix([p00, p01, p11, p21, p02, p12, p22, p03, p13, p23])
    dimension_1132 = Q_sym_1132.shape[0]

    M_sym_1132 = sym.Matrix(
        [Q_sym_1132.transpose()[:-1, :], sym.ones(1, dimension_1132)]
    )
    sym_diff_equations_1132 = M_sym_1132 @ pi_1132

    b_sym_1132 = sym.Matrix([sym.zeros(dimension_1132 - 1, 1), [1]])

    eq0_1132 = sym.Eq(sym_diff_equations_1132[0], b_sym_1132[0])
    eq1_1132 = sym.Eq(sym_diff_equations_1132[1], b_sym_1132[1])
    eq2_1132 = sym.Eq(sym_diff_equations_1132[2], b_sym_1132[2])
    eq3_1132 = sym.Eq(sym_diff_equations_1132[3], b_sym_1132[3])
    eq4_1132 = sym.Eq(sym_diff_equations_1132[4], b_sym_1132[4])
    eq5_1132 = sym.Eq(sym_diff_equations_1132[5], b_sym_1132[5])
    eq6_1132 = sym.Eq(sym_diff_equations_1132[6], b_sym_1132[6])
    eq7_1132 = sym.Eq(sym_diff_equations_1132[7], b_sym_1132[7])
    eq8_1132 = sym.Eq(sym_diff_equations_1132[8], b_sym_1132[8])
    eq9_1132 = sym.Eq(sym_diff_equations_1132[9], b_sym_1132[9])

    sym_state_probs_1132 = sym.solve(
        [
            eq0_1132,
            eq1_1132,
            eq2_1132,
            eq3_1132,
            eq4_1132,
            eq5_1132,
            eq6_1132,
            eq7_1132,
            eq8_1132,
            eq9_1132,
        ],
        (p00, p01, p11, p21, p02, p12, p22, p03, p13, p23),
    )

    sym_state_recursive_ratios_1132 = sym.zeros(
        buffer_capacity + 1, system_capacity + 1
    )
    sym_state_recursive_ratios_1132[0, 0] = 1
    sym_state_recursive_ratios_1132[0, 1] = sym.factor(
        sym_state_probs_1132[p01] / sym_state_probs_1132[p00]
    )  # (0,0) -> (0,1)
    sym_state_recursive_ratios_1132[1, 1] = sym.factor(
        sym_state_probs_1132[p11] / sym_state_probs_1132[p01]
    )  # (0,1) -> (1,1)
    sym_state_recursive_ratios_1132[2, 1] = sym.factor(
        sym_state_probs_1132[p21] / sym_state_probs_1132[p11]
    )  # (1,1) -> (2,1)
    sym_state_recursive_ratios_1132[0, 2] = sym.factor(
        sym_state_probs_1132[p02] / sym_state_probs_1132[p01]
    )  # (0,1) -> (0,2)
    sym_state_recursive_ratios_1132[1, 2] = sym.factor(
        sym_state_probs_1132[p12] / sym_state_probs_1132[p02]
    )  # (0,2) -> (1,2)
    sym_state_recursive_ratios_1132[2, 2] = sym.factor(
        sym_state_probs_1132[p22] / sym_state_probs_1132[p12]
    )  # (1,2) -> (2,2)
    sym_state_recursive_ratios_1132[0, 3] = sym.factor(
        sym_state_probs_1132[p03] / sym_state_probs_1132[p02]
    )  # (0,2) -> (0,3)
    sym_state_recursive_ratios_1132[1, 3] = sym.factor(
        sym_state_probs_1132[p13] / sym_state_probs_1132[p03]
    )  # (0,3) -> (1,3)
    sym_state_recursive_ratios_1132[2, 3] = sym.factor(
        sym_state_probs_1132[p23] / sym_state_probs_1132[p13]
    )  # (1,3) -> (2,3)

    sym_state_recursive_ratios_right_1132 = sym_state_recursive_ratios_1132.copy()
    sym_state_recursive_ratios_right_1132[1, 2] = sym.factor(
        sym_state_probs_1132[p12] / sym_state_probs_1132[p11]
    )  # (1,1) -> (1,2)
    sym_state_recursive_ratios_right_1132[1, 3] = sym.factor(
        sym_state_probs_1132[p13] / sym_state_probs_1132[p12]
    )  # (1,2) -> (1,3)
    sym_state_recursive_ratios_right_1132[2, 2] = sym.factor(
        sym_state_probs_1132[p22] / sym_state_probs_1132[p21]
    )  # (2,1) -> (2,2)
    sym_state_recursive_ratios_right_1132[2, 3] = sym.factor(
        sym_state_probs_1132[p23] / sym_state_probs_1132[p22]
    )  # (2,2) -> (2,3)

    sym_state_recursive_ratios_P0_1132 = sym.zeros(
        buffer_capacity + 1, system_capacity + 1
    )
    sym_state_recursive_ratios_P0_1132[0, 0] = 1
    sym_state_recursive_ratios_P0_1132[0, 1] = sym.factor(
        sym_state_probs_1132[p01] / sym_state_probs_1132[p00]
    )  # (0,0) -> (0,1)
    sym_state_recursive_ratios_P0_1132[1, 1] = sym.factor(
        sym_state_probs_1132[p11] / sym_state_probs_1132[p00]
    )  # (0,0) -> (1,1)
    sym_state_recursive_ratios_P0_1132[2, 1] = sym.factor(
        sym_state_probs_1132[p21] / sym_state_probs_1132[p00]
    )  # (0,0) -> (2,1)
    sym_state_recursive_ratios_P0_1132[0, 2] = sym.factor(
        sym_state_probs_1132[p02] / sym_state_probs_1132[p00]
    )  # (0,0) -> (0,2)
    sym_state_recursive_ratios_P0_1132[1, 2] = sym.factor(
        sym_state_probs_1132[p12] / sym_state_probs_1132[p00]
    )  # (0,0) -> (1,2)
    sym_state_recursive_ratios_P0_1132[2, 2] = sym.factor(
        sym_state_probs_1132[p22] / sym_state_probs_1132[p00]
    )  # (0,0) -> (2,2)
    sym_state_recursive_ratios_P0_1132[0, 3] = sym.factor(
        sym_state_probs_1132[p03] / sym_state_probs_1132[p00]
    )  # (0,0) -> (0,3)
    sym_state_recursive_ratios_P0_1132[1, 3] = sym.factor(
        sym_state_probs_1132[p13] / sym_state_probs_1132[p00]
    )  # (0,0) -> (1,3)
    sym_state_recursive_ratios_P0_1132[2, 3] = sym.factor(
        sym_state_probs_1132[p23] / sym_state_probs_1132[p00]
    )  # (0,0) -> (2,3)

    return (
        sym_state_probs_1132,
        sym_state_recursive_ratios_1132,
        sym_state_recursive_ratios_right_1132,
        sym_state_recursive_ratios_P0_1132,
    )


def get_symbolic_state_probabilities_1141():
    num_of_servers = 1
    threshold = 1
    system_capacity = 4
    buffer_capacity = 1

    Q_sym_1141 = abg.markov.get_symbolic_transition_matrix(
        num_of_servers, threshold, system_capacity, buffer_capacity
    )

    p00, p01, p11, p02, p12, p03, p13, p04, p14 = sym.symbols(
        "p00, p01, p11, p02, p12, p03, p13, p04, p14"
    )
    pi_1141 = sym.Matrix([p00, p01, p11, p02, p12, p03, p13, p04, p14])
    dimension_1141 = Q_sym_1141.shape[0]

    M_sym_1141 = sym.Matrix(
        [Q_sym_1141.transpose()[:-1, :], sym.ones(1, dimension_1141)]
    )
    sym_diff_equations_1141 = M_sym_1141 @ pi_1141

    b_sym_1141 = sym.Matrix([sym.zeros(dimension_1141 - 1, 1), [1]])

    eq0_1141 = sym.Eq(sym_diff_equations_1141[0], b_sym_1141[0])
    eq1_1141 = sym.Eq(sym_diff_equations_1141[1], b_sym_1141[1])
    eq2_1141 = sym.Eq(sym_diff_equations_1141[2], b_sym_1141[2])
    eq3_1141 = sym.Eq(sym_diff_equations_1141[3], b_sym_1141[3])
    eq4_1141 = sym.Eq(sym_diff_equations_1141[4], b_sym_1141[4])
    eq5_1141 = sym.Eq(sym_diff_equations_1141[5], b_sym_1141[5])
    eq6_1141 = sym.Eq(sym_diff_equations_1141[6], b_sym_1141[6])
    eq7_1141 = sym.Eq(sym_diff_equations_1141[7], b_sym_1141[7])
    eq8_1141 = sym.Eq(sym_diff_equations_1141[8], b_sym_1141[8])

    sym_state_probs_1141 = sym.solve(
        [
            eq0_1141,
            eq1_1141,
            eq2_1141,
            eq3_1141,
            eq4_1141,
            eq5_1141,
            eq6_1141,
            eq7_1141,
            eq8_1141,
        ],
        (p00, p01, p11, p02, p12, p03, p13, p04, p14),
    )

    sym_state_recursive_ratios_1141 = sym.zeros(
        buffer_capacity + 1, system_capacity + 1
    )
    sym_state_recursive_ratios_1141[0, 0] = 1
    sym_state_recursive_ratios_1141[0, 1] = sym.factor(
        sym_state_probs_1141[p01] / sym_state_probs_1141[p00]
    )  # (0,0) -> (0,1)
    sym_state_recursive_ratios_1141[1, 1] = sym.factor(
        sym_state_probs_1141[p11] / sym_state_probs_1141[p01]
    )  # (0,1) -> (1,1)
    sym_state_recursive_ratios_1141[0, 2] = sym.factor(
        sym_state_probs_1141[p02] / sym_state_probs_1141[p01]
    )  # (0,1) -> (0,2)
    sym_state_recursive_ratios_1141[1, 2] = sym.factor(
        sym_state_probs_1141[p12] / sym_state_probs_1141[p02]
    )  # (0,2) -> (1,2)
    sym_state_recursive_ratios_1141[0, 3] = sym.factor(
        sym_state_probs_1141[p03] / sym_state_probs_1141[p02]
    )  # (0,2) -> (0,3)
    sym_state_recursive_ratios_1141[1, 3] = sym.factor(
        sym_state_probs_1141[p13] / sym_state_probs_1141[p03]
    )  # (0,3) -> (1,3)
    sym_state_recursive_ratios_1141[0, 4] = sym.factor(
        sym_state_probs_1141[p04] / sym_state_probs_1141[p03]
    )  # (0,3) -> (0,4)
    sym_state_recursive_ratios_1141[1, 4] = sym.factor(
        sym_state_probs_1141[p14] / sym_state_probs_1141[p04]
    )  # (0,4) -> (1,4)

    sym_state_recursive_ratios_right_1141 = sym_state_recursive_ratios_1141.copy()
    sym_state_recursive_ratios_right_1141[1, 2] = sym.factor(
        sym_state_probs_1141[p12] / sym_state_probs_1141[p11]
    )  # (1,1) -> (1,2)
    sym_state_recursive_ratios_right_1141[1, 3] = sym.factor(
        sym_state_probs_1141[p13] / sym_state_probs_1141[p12]
    )  # (1,2) -> (1,3)
    sym_state_recursive_ratios_right_1141[1, 4] = sym.factor(
        sym_state_probs_1141[p14] / sym_state_probs_1141[p13]
    )  # (1,3) -> (1,4)

    sym_state_recursive_ratios_P0_1141 = sym.zeros(
        buffer_capacity + 1, system_capacity + 1
    )
    sym_state_recursive_ratios_P0_1141[0, 0] = 1
    sym_state_recursive_ratios_P0_1141[0, 1] = sym.factor(
        sym_state_probs_1141[p01] / sym_state_probs_1141[p00]
    )  # (0,0) -> (0,1)
    sym_state_recursive_ratios_P0_1141[1, 1] = sym.factor(
        sym_state_probs_1141[p11] / sym_state_probs_1141[p00]
    )  # (0,0) -> (1,1)
    sym_state_recursive_ratios_P0_1141[0, 2] = sym.factor(
        sym_state_probs_1141[p02] / sym_state_probs_1141[p00]
    )  # (0,0) -> (0,2)
    sym_state_recursive_ratios_P0_1141[1, 2] = sym.factor(
        sym_state_probs_1141[p12] / sym_state_probs_1141[p00]
    )  # (0,0) -> (1,2)
    sym_state_recursive_ratios_P0_1141[0, 3] = sym.factor(
        sym_state_probs_1141[p03] / sym_state_probs_1141[p00]
    )  # (0,0) -> (0,3)
    sym_state_recursive_ratios_P0_1141[1, 3] = sym.factor(
        sym_state_probs_1141[p13] / sym_state_probs_1141[p00]
    )  # (0,0) -> (1,3)
    sym_state_recursive_ratios_P0_1141[0, 4] = sym.factor(
        sym_state_probs_1141[p04] / sym_state_probs_1141[p00]
    )  # (0,0) -> (0,4)
    sym_state_recursive_ratios_P0_1141[1, 4] = sym.factor(
        sym_state_probs_1141[p14] / sym_state_probs_1141[p00]
    )  # (0,0) -> (1,4)

    return (
        sym_state_probs_1141,
        sym_state_recursive_ratios_1141,
        sym_state_recursive_ratios_right_1141,
        sym_state_recursive_ratios_P0_1141,
    )


def get_symbolic_state_probabilities_1142():
    num_of_servers = 1
    threshold = 1
    system_capacity = 4
    buffer_capacity = 2

    Q_sym_1142 = abg.markov.get_symbolic_transition_matrix(
        num_of_servers=num_of_servers,
        threshold=threshold,
        system_capacity=system_capacity,
        buffer_capacity=buffer_capacity,
    )

    p00, p01, p11, p21, p02, p12, p22, p03, p13, p23, p04, p14, p24 = sym.symbols(
        "p00, p01, p11, p21, p02, p12, p22, p03, p13, p23, p04, p14, p24"
    )
    pi_1142 = sym.Matrix(
        [p00, p01, p11, p21, p02, p12, p22, p03, p13, p23, p04, p14, p24]
    )
    dimension_1142 = Q_sym_1142.shape[0]

    M_sym_1142 = sym.Matrix(
        [Q_sym_1142.transpose()[:-1, :], sym.ones(1, dimension_1142)]
    )
    sym_diff_equations_1142 = M_sym_1142 @ pi_1142

    b_sym_1142 = sym.Matrix([sym.zeros(dimension_1142 - 1, 1), [1]])

    eq0_1142 = sym.Eq(sym_diff_equations_1142[0], b_sym_1142[0])
    eq1_1142 = sym.Eq(sym_diff_equations_1142[1], b_sym_1142[1])
    eq2_1142 = sym.Eq(sym_diff_equations_1142[2], b_sym_1142[2])
    eq3_1142 = sym.Eq(sym_diff_equations_1142[3], b_sym_1142[3])
    eq4_1142 = sym.Eq(sym_diff_equations_1142[4], b_sym_1142[4])
    eq5_1142 = sym.Eq(sym_diff_equations_1142[5], b_sym_1142[5])
    eq6_1142 = sym.Eq(sym_diff_equations_1142[6], b_sym_1142[6])
    eq7_1142 = sym.Eq(sym_diff_equations_1142[7], b_sym_1142[7])
    eq8_1142 = sym.Eq(sym_diff_equations_1142[8], b_sym_1142[8])
    eq9_1142 = sym.Eq(sym_diff_equations_1142[9], b_sym_1142[9])
    eq10_1142 = sym.Eq(sym_diff_equations_1142[10], b_sym_1142[10])
    eq11_1142 = sym.Eq(sym_diff_equations_1142[11], b_sym_1142[11])
    eq12_1142 = sym.Eq(sym_diff_equations_1142[12], b_sym_1142[12])

    sym_state_probs_1142 = sym.solve(
        [
            eq0_1142,
            eq1_1142,
            eq2_1142,
            eq3_1142,
            eq4_1142,
            eq5_1142,
            eq6_1142,
            eq7_1142,
            eq8_1142,
            eq9_1142,
            eq10_1142,
            eq11_1142,
            eq12_1142,
        ],
        (p00, p01, p11, p21, p02, p12, p22, p03, p13, p23, p04, p14, p24),
    )

    sym_state_recursive_ratios_1142 = sym.zeros(
        buffer_capacity + 1, system_capacity + 1
    )
    sym_state_recursive_ratios_1142[0, 0] = 1
    sym_state_recursive_ratios_1142[0, 1] = sym.factor(
        sym_state_probs_1142[p01] / sym_state_probs_1142[p00]
    )  # (0,0) -> (0,1)
    sym_state_recursive_ratios_1142[1, 1] = sym.factor(
        sym_state_probs_1142[p11] / sym_state_probs_1142[p01]
    )  # (0,1) -> (1,1)
    sym_state_recursive_ratios_1142[2, 1] = sym.factor(
        sym_state_probs_1142[p21] / sym_state_probs_1142[p11]
    )  # (1,1) -> (2,1)
    sym_state_recursive_ratios_1142[0, 2] = sym.factor(
        sym_state_probs_1142[p02] / sym_state_probs_1142[p01]
    )  # (0,1) -> (0,2)
    sym_state_recursive_ratios_1142[1, 2] = sym.factor(
        sym_state_probs_1142[p12] / sym_state_probs_1142[p02]
    )  # (0,2) -> (1,2)
    sym_state_recursive_ratios_1142[2, 2] = sym.factor(
        sym_state_probs_1142[p22] / sym_state_probs_1142[p12]
    )  # (1,2) -> (2,2)
    sym_state_recursive_ratios_1142[0, 3] = sym.factor(
        sym_state_probs_1142[p03] / sym_state_probs_1142[p02]
    )  # (0,2) -> (0,3)
    sym_state_recursive_ratios_1142[1, 3] = sym.factor(
        sym_state_probs_1142[p13] / sym_state_probs_1142[p03]
    )  # (0,3) -> (1,3)
    sym_state_recursive_ratios_1142[2, 3] = sym.factor(
        sym_state_probs_1142[p23] / sym_state_probs_1142[p13]
    )  # (1,3) -> (2,3)
    sym_state_recursive_ratios_1142[0, 4] = sym.factor(
        sym_state_probs_1142[p04] / sym_state_probs_1142[p03]
    )  # (0,3) -> (0,4)
    sym_state_recursive_ratios_1142[1, 4] = sym.factor(
        sym_state_probs_1142[p14] / sym_state_probs_1142[p04]
    )  # (0,4) -> (1,4)
    sym_state_recursive_ratios_1142[2, 4] = sym.factor(
        sym_state_probs_1142[p24] / sym_state_probs_1142[p14]
    )  # (1,4) -> (2,4)

    sym_state_recursive_ratios_right_1142 = sym_state_recursive_ratios_1142.copy()
    sym_state_recursive_ratios_right_1142[1, 2] = sym.factor(
        sym_state_probs_1142[p12] / sym_state_probs_1142[p11]
    )  # (1,1) -> (1,2)
    sym_state_recursive_ratios_right_1142[1, 3] = sym.factor(
        sym_state_probs_1142[p13] / sym_state_probs_1142[p12]
    )  # (1,2) -> (1,3)
    sym_state_recursive_ratios_right_1142[1, 4] = sym.factor(
        sym_state_probs_1142[p14] / sym_state_probs_1142[p13]
    )  # (1,3) -> (1,4)
    sym_state_recursive_ratios_right_1142[2, 2] = sym.factor(
        sym_state_probs_1142[p22] / sym_state_probs_1142[p21]
    )  # (2,1) -> (2,2)
    sym_state_recursive_ratios_right_1142[2, 3] = sym.factor(
        sym_state_probs_1142[p23] / sym_state_probs_1142[p22]
    )  # (2,2) -> (2,3)
    sym_state_recursive_ratios_right_1142[2, 4] = sym.factor(
        sym_state_probs_1142[p24] / sym_state_probs_1142[p23]
    )  # (2,3) -> (2,4)

    sym_state_recursive_ratios_P0_1142 = sym.zeros(
        buffer_capacity + 1, system_capacity + 1
    )
    sym_state_recursive_ratios_P0_1142[0, 0] = 1
    sym_state_recursive_ratios_P0_1142[0, 1] = sym.factor(
        sym_state_probs_1142[p01] / sym_state_probs_1142[p00]
    )  # (0,0) -> (0,1)
    sym_state_recursive_ratios_P0_1142[1, 1] = sym.factor(
        sym_state_probs_1142[p11] / sym_state_probs_1142[p00]
    )  # (0,0) -> (1,1)
    sym_state_recursive_ratios_P0_1142[2, 1] = sym.factor(
        sym_state_probs_1142[p21] / sym_state_probs_1142[p00]
    )  # (0,0) -> (2,1)

    sym_state_recursive_ratios_P0_1142[0, 2] = sym.factor(
        sym_state_probs_1142[p02] / sym_state_probs_1142[p00]
    )  # (0,0) -> (0,2)
    sym_state_recursive_ratios_P0_1142[1, 2] = sym.factor(
        sym_state_probs_1142[p12] / sym_state_probs_1142[p00]
    )  # (0,0) -> (1,2)
    sym_state_recursive_ratios_P0_1142[2, 2] = sym.factor(
        sym_state_probs_1142[p22] / sym_state_probs_1142[p00]
    )  # (0,0) -> (2,2)

    sym_state_recursive_ratios_P0_1142[0, 3] = sym.factor(
        sym_state_probs_1142[p03] / sym_state_probs_1142[p00]
    )  # (0,0) -> (0,3)
    sym_state_recursive_ratios_P0_1142[1, 3] = sym.factor(
        sym_state_probs_1142[p13] / sym_state_probs_1142[p00]
    )  # (0,0) -> (1,3)
    sym_state_recursive_ratios_P0_1142[2, 3] = sym.factor(
        sym_state_probs_1142[p23] / sym_state_probs_1142[p00]
    )  # (0,0) -> (2,3)

    sym_state_recursive_ratios_P0_1142[0, 4] = sym.factor(
        sym_state_probs_1142[p04] / sym_state_probs_1142[p00]
    )  # (0,0) -> (0,4)
    sym_state_recursive_ratios_P0_1142[1, 4] = sym.factor(
        sym_state_probs_1142[p14] / sym_state_probs_1142[p00]
    )  # (0,0) -> (1,4)
    sym_state_recursive_ratios_P0_1142[2, 4] = sym.factor(
        sym_state_probs_1142[p24] / sym_state_probs_1142[p00]
    )  # (0,0) -> (2,4)

    return (
        sym_state_probs_1142,
        sym_state_recursive_ratios_1142,
        sym_state_recursive_ratios_right_1142,
        sym_state_recursive_ratios_P0_1142,
    )


def get_symbolic_state_probabilities_1151():
    num_of_servers = 1
    threshold = 1
    system_capacity = 5
    buffer_capacity = 1

    Q_sym_1151 = abg.markov.get_symbolic_transition_matrix(
        num_of_servers, threshold, system_capacity, buffer_capacity
    )

    p00, p01, p11, p02, p12, p03, p13, p04, p14, p05, p15 = sym.symbols(
        "p00, p01, p11, p02, p12, p03, p13, p04, p14, p05, p15"
    )
    pi_1151 = sym.Matrix([p00, p01, p11, p02, p12, p03, p13, p04, p14, p05, p15])
    dimension_1151 = Q_sym_1151.shape[0]

    M_sym_1151 = sym.Matrix(
        [Q_sym_1151.transpose()[:-1, :], sym.ones(1, dimension_1151)]
    )
    sym_diff_equations_1151 = M_sym_1151 @ pi_1151

    b_sym_1151 = sym.Matrix([sym.zeros(dimension_1151 - 1, 1), [1]])

    eq0_1151 = sym.Eq(sym_diff_equations_1151[0], b_sym_1151[0])
    eq1_1151 = sym.Eq(sym_diff_equations_1151[1], b_sym_1151[1])
    eq2_1151 = sym.Eq(sym_diff_equations_1151[2], b_sym_1151[2])
    eq3_1151 = sym.Eq(sym_diff_equations_1151[3], b_sym_1151[3])
    eq4_1151 = sym.Eq(sym_diff_equations_1151[4], b_sym_1151[4])
    eq5_1151 = sym.Eq(sym_diff_equations_1151[5], b_sym_1151[5])
    eq6_1151 = sym.Eq(sym_diff_equations_1151[6], b_sym_1151[6])
    eq7_1151 = sym.Eq(sym_diff_equations_1151[7], b_sym_1151[7])
    eq8_1151 = sym.Eq(sym_diff_equations_1151[8], b_sym_1151[8])
    eq9_1151 = sym.Eq(sym_diff_equations_1151[9], b_sym_1151[9])
    eq10_1151 = sym.Eq(sym_diff_equations_1151[10], b_sym_1151[10])

    sym_state_probs_1151 = sym.solve(
        [
            eq0_1151,
            eq1_1151,
            eq2_1151,
            eq3_1151,
            eq4_1151,
            eq5_1151,
            eq6_1151,
            eq7_1151,
            eq8_1151,
            eq9_1151,
            eq10_1151,
        ],
        (p00, p01, p11, p02, p12, p03, p13, p04, p14, p05, p15),
    )

    sym_state_recursive_ratios_1151 = sym.zeros(
        buffer_capacity + 1, system_capacity + 1
    )
    sym_state_recursive_ratios_1151[0, 0] = 1
    sym_state_recursive_ratios_1151[0, 1] = sym.factor(
        sym_state_probs_1151[p01] / sym_state_probs_1151[p00]
    )  # (0,0) -> (0,1)
    sym_state_recursive_ratios_1151[1, 1] = sym.factor(
        sym_state_probs_1151[p11] / sym_state_probs_1151[p01]
    )  # (0,1) -> (1,1)
    sym_state_recursive_ratios_1151[0, 2] = sym.factor(
        sym_state_probs_1151[p02] / sym_state_probs_1151[p01]
    )  # (0,1) -> (0,2)
    sym_state_recursive_ratios_1151[1, 2] = sym.factor(
        sym_state_probs_1151[p12] / sym_state_probs_1151[p02]
    )  # (0,2) -> (1,2)
    sym_state_recursive_ratios_1151[0, 3] = sym.factor(
        sym_state_probs_1151[p03] / sym_state_probs_1151[p02]
    )  # (0,2) -> (0,3)
    sym_state_recursive_ratios_1151[1, 3] = sym.factor(
        sym_state_probs_1151[p13] / sym_state_probs_1151[p03]
    )  # (0,3) -> (1,3)
    sym_state_recursive_ratios_1151[0, 4] = sym.factor(
        sym_state_probs_1151[p04] / sym_state_probs_1151[p03]
    )  # (0,3) -> (0,4)
    sym_state_recursive_ratios_1151[1, 4] = sym.factor(
        sym_state_probs_1151[p14] / sym_state_probs_1151[p04]
    )  # (0,4) -> (1,4)
    sym_state_recursive_ratios_1151[0, 5] = sym.factor(
        sym_state_probs_1151[p05] / sym_state_probs_1151[p04]
    )  # (0,4) -> (0,5)
    sym_state_recursive_ratios_1151[1, 5] = sym.factor(
        sym_state_probs_1151[p15] / sym_state_probs_1151[p05]
    )  # (0,5) -> (1,5)

    sym_state_recursive_ratios_right_1151 = sym_state_recursive_ratios_1151.copy()
    sym_state_recursive_ratios_right_1151[1, 2] = sym.factor(
        sym_state_probs_1151[p12] / sym_state_probs_1151[p11]
    )  # (1,1) -> (1,2)
    sym_state_recursive_ratios_right_1151[1, 3] = sym.factor(
        sym_state_probs_1151[p13] / sym_state_probs_1151[p12]
    )  # (1,2) -> (1,3)
    sym_state_recursive_ratios_right_1151[1, 4] = sym.factor(
        sym_state_probs_1151[p14] / sym_state_probs_1151[p13]
    )  # (1,3) -> (1,4)
    sym_state_recursive_ratios_right_1151[1, 5] = sym.factor(
        sym_state_probs_1151[p15] / sym_state_probs_1151[p14]
    )  # (1,4) -> (1,5)

    sym_state_recursive_ratios_P0_1151 = sym.zeros(
        buffer_capacity + 1, system_capacity + 1
    )
    sym_state_recursive_ratios_P0_1151[0, 0] = 1
    sym_state_recursive_ratios_P0_1151[0, 1] = sym.factor(
        sym_state_probs_1151[p01] / sym_state_probs_1151[p00]
    )  # (0,0) -> (0,1)
    sym_state_recursive_ratios_P0_1151[1, 1] = sym.factor(
        sym_state_probs_1151[p11] / sym_state_probs_1151[p00]
    )  # (0,0) -> (1,1)
    sym_state_recursive_ratios_P0_1151[0, 2] = sym.factor(
        sym_state_probs_1151[p02] / sym_state_probs_1151[p00]
    )  # (0,0) -> (0,2)
    sym_state_recursive_ratios_P0_1151[1, 2] = sym.factor(
        sym_state_probs_1151[p12] / sym_state_probs_1151[p00]
    )  # (0,0) -> (1,2)
    sym_state_recursive_ratios_P0_1151[0, 3] = sym.factor(
        sym_state_probs_1151[p03] / sym_state_probs_1151[p00]
    )  # (0,0) -> (0,3)
    sym_state_recursive_ratios_P0_1151[1, 3] = sym.factor(
        sym_state_probs_1151[p13] / sym_state_probs_1151[p00]
    )  # (0,0) -> (1,3)
    sym_state_recursive_ratios_P0_1151[0, 4] = sym.factor(
        sym_state_probs_1151[p04] / sym_state_probs_1151[p00]
    )  # (0,0) -> (0,4)
    sym_state_recursive_ratios_P0_1151[1, 4] = sym.factor(
        sym_state_probs_1151[p14] / sym_state_probs_1151[p00]
    )  # (0,0) -> (1,4)
    sym_state_recursive_ratios_P0_1151[0, 5] = sym.factor(
        sym_state_probs_1151[p05] / sym_state_probs_1151[p00]
    )  # (0,0) -> (0,5)
    sym_state_recursive_ratios_P0_1151[1, 5] = sym.factor(
        sym_state_probs_1151[p15] / sym_state_probs_1151[p00]
    )  # (0,0) -> (1,5)

    return (
        sym_state_probs_1151,
        sym_state_recursive_ratios_1151,
        sym_state_recursive_ratios_right_1151,
        sym_state_recursive_ratios_P0_1151,
    )


def get_symbolic_state_probabilities_1161():
    num_of_servers = 1
    threshold = 1
    system_capacity = 6
    buffer_capacity = 1

    Q_sym_1161 = abg.markov.get_symbolic_transition_matrix(
        num_of_servers, threshold, system_capacity, buffer_capacity
    )

    p00, p01, p11, p02, p12, p03, p13, p04, p14, p05, p15, p06, p16 = sym.symbols(
        "p00, p01, p11, p02, p12, p03, p13, p04, p14, p05, p15, p06, p16"
    )
    pi_1161 = sym.Matrix(
        [p00, p01, p11, p02, p12, p03, p13, p04, p14, p05, p15, p06, p16]
    )
    dimension_1161 = Q_sym_1161.shape[0]

    M_sym_1161 = sym.Matrix(
        [Q_sym_1161.transpose()[:-1, :], sym.ones(1, dimension_1161)]
    )
    sym_diff_equations_1161 = M_sym_1161 @ pi_1161

    b_sym_1161 = sym.Matrix([sym.zeros(dimension_1161 - 1, 1), [1]])

    eq0_1161 = sym.Eq(sym_diff_equations_1161[0], b_sym_1161[0])
    eq1_1161 = sym.Eq(sym_diff_equations_1161[1], b_sym_1161[1])
    eq2_1161 = sym.Eq(sym_diff_equations_1161[2], b_sym_1161[2])
    eq3_1161 = sym.Eq(sym_diff_equations_1161[3], b_sym_1161[3])
    eq4_1161 = sym.Eq(sym_diff_equations_1161[4], b_sym_1161[4])
    eq5_1161 = sym.Eq(sym_diff_equations_1161[5], b_sym_1161[5])
    eq6_1161 = sym.Eq(sym_diff_equations_1161[6], b_sym_1161[6])
    eq7_1161 = sym.Eq(sym_diff_equations_1161[7], b_sym_1161[7])
    eq8_1161 = sym.Eq(sym_diff_equations_1161[8], b_sym_1161[8])
    eq9_1161 = sym.Eq(sym_diff_equations_1161[9], b_sym_1161[9])
    eq10_1161 = sym.Eq(sym_diff_equations_1161[10], b_sym_1161[10])
    eq11_1161 = sym.Eq(sym_diff_equations_1161[11], b_sym_1161[11])
    eq12_1161 = sym.Eq(sym_diff_equations_1161[12], b_sym_1161[12])

    sym_state_probs_1161 = sym.solve(
        [
            eq0_1161,
            eq1_1161,
            eq2_1161,
            eq3_1161,
            eq4_1161,
            eq5_1161,
            eq6_1161,
            eq7_1161,
            eq8_1161,
            eq9_1161,
            eq10_1161,
            eq11_1161,
            eq12_1161,
        ],
        (p00, p01, p11, p02, p12, p03, p13, p04, p14, p05, p15, p06, p16),
    )

    sym_state_recursive_ratios_1161 = sym.zeros(
        buffer_capacity + 1, system_capacity + 1
    )
    sym_state_recursive_ratios_1161[0, 0] = 1
    sym_state_recursive_ratios_1161[0, 1] = sym.factor(
        sym_state_probs_1161[p01] / sym_state_probs_1161[p00]
    )  # (0,0) -> (0,1)
    sym_state_recursive_ratios_1161[1, 1] = sym.factor(
        sym_state_probs_1161[p11] / sym_state_probs_1161[p01]
    )  # (0,1) -> (1,1)
    sym_state_recursive_ratios_1161[0, 2] = sym.factor(
        sym_state_probs_1161[p02] / sym_state_probs_1161[p01]
    )  # (0,1) -> (0,2)
    sym_state_recursive_ratios_1161[1, 2] = sym.factor(
        sym_state_probs_1161[p12] / sym_state_probs_1161[p02]
    )  # (0,2) -> (1,2)
    sym_state_recursive_ratios_1161[0, 3] = sym.factor(
        sym_state_probs_1161[p03] / sym_state_probs_1161[p02]
    )  # (0,2) -> (0,3)
    sym_state_recursive_ratios_1161[1, 3] = sym.factor(
        sym_state_probs_1161[p13] / sym_state_probs_1161[p03]
    )  # (0,3) -> (1,3)
    sym_state_recursive_ratios_1161[0, 4] = sym.factor(
        sym_state_probs_1161[p04] / sym_state_probs_1161[p03]
    )  # (0,3) -> (0,4)
    sym_state_recursive_ratios_1161[1, 4] = sym.factor(
        sym_state_probs_1161[p14] / sym_state_probs_1161[p04]
    )  # (0,4) -> (1,4)
    sym_state_recursive_ratios_1161[0, 5] = sym.factor(
        sym_state_probs_1161[p05] / sym_state_probs_1161[p04]
    )  # (0,4) -> (0,5)
    sym_state_recursive_ratios_1161[1, 5] = sym.factor(
        sym_state_probs_1161[p15] / sym_state_probs_1161[p05]
    )  # (0,5) -> (1,5)
    sym_state_recursive_ratios_1161[0, 6] = sym.factor(
        sym_state_probs_1161[p06] / sym_state_probs_1161[p05]
    )  # (0,5) -> (0,6)
    sym_state_recursive_ratios_1161[1, 6] = sym.factor(
        sym_state_probs_1161[p16] / sym_state_probs_1161[p06]
    )  # (0,6) -> (1,6)

    sym_state_recursive_ratios_right_1161 = sym_state_recursive_ratios_1161.copy()
    sym_state_recursive_ratios_right_1161[1, 2] = sym.factor(
        sym_state_probs_1161[p12] / sym_state_probs_1161[p11]
    )  # (1,1) -> (1,2)
    sym_state_recursive_ratios_right_1161[1, 3] = sym.factor(
        sym_state_probs_1161[p13] / sym_state_probs_1161[p12]
    )  # (1,2) -> (1,3)
    sym_state_recursive_ratios_right_1161[1, 4] = sym.factor(
        sym_state_probs_1161[p14] / sym_state_probs_1161[p13]
    )  # (1,3) -> (1,4)
    sym_state_recursive_ratios_right_1161[1, 5] = sym.factor(
        sym_state_probs_1161[p15] / sym_state_probs_1161[p14]
    )  # (1,4) -> (1,5)
    sym_state_recursive_ratios_right_1161[1, 6] = sym.factor(
        sym_state_probs_1161[p16] / sym_state_probs_1161[p15]
    )  # (1,5) -> (1,6)

    sym_state_recursive_ratios_P0_1161 = sym.zeros(
        buffer_capacity + 1, system_capacity + 1
    )
    sym_state_recursive_ratios_P0_1161[0, 0] = 1
    sym_state_recursive_ratios_P0_1161[0, 1] = sym.factor(
        sym_state_probs_1161[p01] / sym_state_probs_1161[p00]
    )  # (0,0) -> (0,1)
    sym_state_recursive_ratios_P0_1161[1, 1] = sym.factor(
        sym_state_probs_1161[p11] / sym_state_probs_1161[p00]
    )  # (0,0) -> (1,1)
    sym_state_recursive_ratios_P0_1161[0, 2] = sym.factor(
        sym_state_probs_1161[p02] / sym_state_probs_1161[p00]
    )  # (0,0) -> (0,2)
    sym_state_recursive_ratios_P0_1161[1, 2] = sym.factor(
        sym_state_probs_1161[p12] / sym_state_probs_1161[p00]
    )  # (0,0) -> (1,2)
    sym_state_recursive_ratios_P0_1161[0, 3] = sym.factor(
        sym_state_probs_1161[p03] / sym_state_probs_1161[p00]
    )  # (0,0) -> (0,3)
    sym_state_recursive_ratios_P0_1161[1, 3] = sym.factor(
        sym_state_probs_1161[p13] / sym_state_probs_1161[p00]
    )  # (0,0) -> (1,3)
    sym_state_recursive_ratios_P0_1161[0, 4] = sym.factor(
        sym_state_probs_1161[p04] / sym_state_probs_1161[p00]
    )  # (0,0) -> (0,4)
    sym_state_recursive_ratios_P0_1161[1, 4] = sym.factor(
        sym_state_probs_1161[p14] / sym_state_probs_1161[p00]
    )  # (0,0) -> (1,4)
    sym_state_recursive_ratios_P0_1161[0, 5] = sym.factor(
        sym_state_probs_1161[p05] / sym_state_probs_1161[p00]
    )  # (0,0) -> (0,5)
    sym_state_recursive_ratios_P0_1161[1, 5] = sym.factor(
        sym_state_probs_1161[p15] / sym_state_probs_1161[p00]
    )  # (0,0) -> (1,5)
    sym_state_recursive_ratios_P0_1161[0, 6] = sym.factor(
        sym_state_probs_1161[p06] / sym_state_probs_1161[p00]
    )  # (0,0) -> (0,6)
    sym_state_recursive_ratios_P0_1161[1, 6] = sym.factor(
        sym_state_probs_1161[p16] / sym_state_probs_1161[p00]
    )  # (0,0) -> (1,6)

    return (
        sym_state_probs_1161,
        sym_state_recursive_ratios_1161,
        sym_state_recursive_ratios_right_1161,
        sym_state_recursive_ratios_P0_1161,
    )


def get_symbolic_state_probabilities_1171():
    num_of_servers = 1
    threshold = 1
    system_capacity = 7
    buffer_capacity = 1

    Q_sym_1171 = abg.markov.get_symbolic_transition_matrix(
        num_of_servers, threshold, system_capacity, buffer_capacity
    )

    (
        p00,
        p01,
        p11,
        p02,
        p12,
        p03,
        p13,
        p04,
        p14,
        p05,
        p15,
        p06,
        p16,
        p07,
        p17,
    ) = sym.symbols(
        "p00, p01, p11, p02, p12, p03, p13, p04, p14, p05, p15, p06, p16, p07, p17"
    )
    pi_1171 = sym.Matrix(
        [p00, p01, p11, p02, p12, p03, p13, p04, p14, p05, p15, p06, p16, p07, p17]
    )
    dimension_1171 = Q_sym_1171.shape[0]

    M_sym_1171 = sym.Matrix(
        [Q_sym_1171.transpose()[:-1, :], sym.ones(1, dimension_1171)]
    )
    sym_diff_equations_1171 = M_sym_1171 @ pi_1171

    b_sym_1171 = sym.Matrix([sym.zeros(dimension_1171 - 1, 1), [1]])

    eq0_1171 = sym.Eq(sym_diff_equations_1171[0], b_sym_1171[0])
    eq1_1171 = sym.Eq(sym_diff_equations_1171[1], b_sym_1171[1])
    eq2_1171 = sym.Eq(sym_diff_equations_1171[2], b_sym_1171[2])
    eq3_1171 = sym.Eq(sym_diff_equations_1171[3], b_sym_1171[3])
    eq4_1171 = sym.Eq(sym_diff_equations_1171[4], b_sym_1171[4])
    eq5_1171 = sym.Eq(sym_diff_equations_1171[5], b_sym_1171[5])
    eq6_1171 = sym.Eq(sym_diff_equations_1171[6], b_sym_1171[6])
    eq7_1171 = sym.Eq(sym_diff_equations_1171[7], b_sym_1171[7])
    eq8_1171 = sym.Eq(sym_diff_equations_1171[8], b_sym_1171[8])
    eq9_1171 = sym.Eq(sym_diff_equations_1171[9], b_sym_1171[9])
    eq10_1171 = sym.Eq(sym_diff_equations_1171[10], b_sym_1171[10])
    eq11_1171 = sym.Eq(sym_diff_equations_1171[11], b_sym_1171[11])
    eq12_1171 = sym.Eq(sym_diff_equations_1171[12], b_sym_1171[12])
    eq13_1171 = sym.Eq(sym_diff_equations_1171[13], b_sym_1171[13])
    eq14_1171 = sym.Eq(sym_diff_equations_1171[14], b_sym_1171[14])

    sym_state_probs_1171 = sym.solve(
        [
            eq0_1171,
            eq1_1171,
            eq2_1171,
            eq3_1171,
            eq4_1171,
            eq5_1171,
            eq6_1171,
            eq7_1171,
            eq8_1171,
            eq9_1171,
            eq10_1171,
            eq11_1171,
            eq12_1171,
            eq13_1171,
            eq14_1171,
        ],
        (p00, p01, p11, p02, p12, p03, p13, p04, p14, p05, p15, p06, p16, p07, p17),
    )

    sym_state_recursive_ratios_1171 = sym.zeros(
        buffer_capacity + 1, system_capacity + 1
    )
    sym_state_recursive_ratios_1171[0, 0] = 1
    sym_state_recursive_ratios_1171[0, 1] = sym.factor(
        sym_state_probs_1171[p01] / sym_state_probs_1171[p00]
    )  # (0,0) -> (0,1)
    sym_state_recursive_ratios_1171[1, 1] = sym.factor(
        sym_state_probs_1171[p11] / sym_state_probs_1171[p01]
    )  # (0,1) -> (1,1)
    sym_state_recursive_ratios_1171[0, 2] = sym.factor(
        sym_state_probs_1171[p02] / sym_state_probs_1171[p01]
    )  # (0,1) -> (0,2)
    sym_state_recursive_ratios_1171[1, 2] = sym.factor(
        sym_state_probs_1171[p12] / sym_state_probs_1171[p02]
    )  # (0,2) -> (1,2)
    sym_state_recursive_ratios_1171[0, 3] = sym.factor(
        sym_state_probs_1171[p03] / sym_state_probs_1171[p02]
    )  # (0,2) -> (0,3)
    sym_state_recursive_ratios_1171[1, 3] = sym.factor(
        sym_state_probs_1171[p13] / sym_state_probs_1171[p03]
    )  # (0,3) -> (1,3)
    sym_state_recursive_ratios_1171[0, 4] = sym.factor(
        sym_state_probs_1171[p04] / sym_state_probs_1171[p03]
    )  # (0,3) -> (0,4)
    sym_state_recursive_ratios_1171[1, 4] = sym.factor(
        sym_state_probs_1171[p14] / sym_state_probs_1171[p04]
    )  # (0,4) -> (1,4)
    sym_state_recursive_ratios_1171[0, 5] = sym.factor(
        sym_state_probs_1171[p05] / sym_state_probs_1171[p04]
    )  # (0,4) -> (0,5)
    sym_state_recursive_ratios_1171[1, 5] = sym.factor(
        sym_state_probs_1171[p15] / sym_state_probs_1171[p05]
    )  # (0,5) -> (1,5)
    sym_state_recursive_ratios_1171[0, 6] = sym.factor(
        sym_state_probs_1171[p06] / sym_state_probs_1171[p05]
    )  # (0,5) -> (0,6)
    sym_state_recursive_ratios_1171[1, 6] = sym.factor(
        sym_state_probs_1171[p16] / sym_state_probs_1171[p06]
    )  # (0,6) -> (1,6)
    sym_state_recursive_ratios_1171[0, 7] = sym.factor(
        sym_state_probs_1171[p07] / sym_state_probs_1171[p06]
    )  # (0,6) -> (0,7)
    sym_state_recursive_ratios_1171[1, 7] = sym.factor(
        sym_state_probs_1171[p17] / sym_state_probs_1171[p07]
    )  # (0,7) -> (1,7)

    sym_state_recursive_ratios_right_1171 = sym_state_recursive_ratios_1171.copy()
    sym_state_recursive_ratios_right_1171[1, 2] = sym.factor(
        sym_state_probs_1171[p12] / sym_state_probs_1171[p11]
    )  # (1,1) -> (1,2)
    sym_state_recursive_ratios_right_1171[1, 3] = sym.factor(
        sym_state_probs_1171[p13] / sym_state_probs_1171[p12]
    )  # (1,2) -> (1,3)
    sym_state_recursive_ratios_right_1171[1, 4] = sym.factor(
        sym_state_probs_1171[p14] / sym_state_probs_1171[p13]
    )  # (1,3) -> (1,4)
    sym_state_recursive_ratios_right_1171[1, 5] = sym.factor(
        sym_state_probs_1171[p15] / sym_state_probs_1171[p14]
    )  # (1,4) -> (1,5)
    sym_state_recursive_ratios_right_1171[1, 6] = sym.factor(
        sym_state_probs_1171[p16] / sym_state_probs_1171[p15]
    )  # (1,5) -> (1,6)
    sym_state_recursive_ratios_right_1171[1, 7] = sym.factor(
        sym_state_probs_1171[p17] / sym_state_probs_1171[p16]
    )  # (1,6) -> (1,7)

    sym_state_recursive_ratios_P0_1171 = sym.zeros(
        buffer_capacity + 1, system_capacity + 1
    )
    sym_state_recursive_ratios_P0_1171[0, 0] = 1
    sym_state_recursive_ratios_P0_1171[0, 1] = sym.factor(
        sym_state_probs_1171[p01] / sym_state_probs_1171[p00]
    )  # (0,0) -> (0,1)
    sym_state_recursive_ratios_P0_1171[1, 1] = sym.factor(
        sym_state_probs_1171[p11] / sym_state_probs_1171[p00]
    )  # (0,0) -> (1,1)
    sym_state_recursive_ratios_P0_1171[0, 2] = sym.factor(
        sym_state_probs_1171[p02] / sym_state_probs_1171[p00]
    )  # (0,0) -> (0,2)
    sym_state_recursive_ratios_P0_1171[1, 2] = sym.factor(
        sym_state_probs_1171[p12] / sym_state_probs_1171[p00]
    )  # (0,0) -> (1,2)
    sym_state_recursive_ratios_P0_1171[0, 3] = sym.factor(
        sym_state_probs_1171[p03] / sym_state_probs_1171[p00]
    )  # (0,0) -> (0,3)
    sym_state_recursive_ratios_P0_1171[1, 3] = sym.factor(
        sym_state_probs_1171[p13] / sym_state_probs_1171[p00]
    )  # (0,0) -> (1,3)
    sym_state_recursive_ratios_P0_1171[0, 4] = sym.factor(
        sym_state_probs_1171[p04] / sym_state_probs_1171[p00]
    )  # (0,0) -> (0,4)
    sym_state_recursive_ratios_P0_1171[1, 4] = sym.factor(
        sym_state_probs_1171[p14] / sym_state_probs_1171[p00]
    )  # (0,0) -> (1,4)
    sym_state_recursive_ratios_P0_1171[0, 5] = sym.factor(
        sym_state_probs_1171[p05] / sym_state_probs_1171[p00]
    )  # (0,0) -> (0,5)
    sym_state_recursive_ratios_P0_1171[1, 5] = sym.factor(
        sym_state_probs_1171[p15] / sym_state_probs_1171[p00]
    )  # (0,0) -> (1,5)
    sym_state_recursive_ratios_P0_1171[0, 6] = sym.factor(
        sym_state_probs_1171[p06] / sym_state_probs_1171[p00]
    )  # (0,0) -> (0,6)
    sym_state_recursive_ratios_P0_1171[1, 6] = sym.factor(
        sym_state_probs_1171[p16] / sym_state_probs_1171[p00]
    )  # (0,0) -> (1,6)
    sym_state_recursive_ratios_P0_1171[0, 7] = sym.factor(
        sym_state_probs_1171[p07] / sym_state_probs_1171[p00]
    )  # (0,0) -> (0,7)
    sym_state_recursive_ratios_P0_1171[1, 7] = sym.factor(
        sym_state_probs_1171[p17] / sym_state_probs_1171[p00]
    )  # (0,0) -> (1,7)

    return (
        sym_state_probs_1171,
        sym_state_recursive_ratios_1171,
        sym_state_recursive_ratios_right_1171,
        sym_state_recursive_ratios_P0_1171,
    )


def get_symbolic_state_probabilities_1181():
    num_of_servers = 1
    threshold = 1
    system_capacity = 8
    buffer_capacity = 1

    Q_sym_1181 = abg.markov.get_symbolic_transition_matrix(
        num_of_servers, threshold, system_capacity, buffer_capacity
    )

    (
        p00,
        p01,
        p11,
        p02,
        p12,
        p03,
        p13,
        p04,
        p14,
        p05,
        p15,
        p06,
        p16,
        p07,
        p17,
        p08,
        p18,
    ) = sym.symbols(
        "p00, p01, p11, p02, p12, p03, p13, p04, p14, p05, p15, p06, p16, p07, p17, p08, p18"
    )
    pi_1181 = sym.Matrix(
        [
            p00,
            p01,
            p11,
            p02,
            p12,
            p03,
            p13,
            p04,
            p14,
            p05,
            p15,
            p06,
            p16,
            p07,
            p17,
            p08,
            p18,
        ]
    )
    dimension_1181 = Q_sym_1181.shape[0]

    M_sym_1181 = sym.Matrix(
        [Q_sym_1181.transpose()[:-1, :], sym.ones(1, dimension_1181)]
    )
    sym_diff_equations_1181 = M_sym_1181 @ pi_1181

    b_sym_1181 = sym.Matrix([sym.zeros(dimension_1181 - 1, 1), [1]])

    eq0_1181 = sym.Eq(sym_diff_equations_1181[0], b_sym_1181[0])
    eq1_1181 = sym.Eq(sym_diff_equations_1181[1], b_sym_1181[1])
    eq2_1181 = sym.Eq(sym_diff_equations_1181[2], b_sym_1181[2])
    eq3_1181 = sym.Eq(sym_diff_equations_1181[3], b_sym_1181[3])
    eq4_1181 = sym.Eq(sym_diff_equations_1181[4], b_sym_1181[4])
    eq5_1181 = sym.Eq(sym_diff_equations_1181[5], b_sym_1181[5])
    eq6_1181 = sym.Eq(sym_diff_equations_1181[6], b_sym_1181[6])
    eq7_1181 = sym.Eq(sym_diff_equations_1181[7], b_sym_1181[7])
    eq8_1181 = sym.Eq(sym_diff_equations_1181[8], b_sym_1181[8])
    eq9_1181 = sym.Eq(sym_diff_equations_1181[9], b_sym_1181[9])
    eq10_1181 = sym.Eq(sym_diff_equations_1181[10], b_sym_1181[10])
    eq11_1181 = sym.Eq(sym_diff_equations_1181[11], b_sym_1181[11])
    eq12_1181 = sym.Eq(sym_diff_equations_1181[12], b_sym_1181[12])
    eq13_1181 = sym.Eq(sym_diff_equations_1181[13], b_sym_1181[13])
    eq14_1181 = sym.Eq(sym_diff_equations_1181[14], b_sym_1181[14])
    eq15_1181 = sym.Eq(sym_diff_equations_1181[15], b_sym_1181[15])
    eq16_1181 = sym.Eq(sym_diff_equations_1181[16], b_sym_1181[16])

    sym_state_probs_1181 = sym.solve(
        [
            eq0_1181,
            eq1_1181,
            eq2_1181,
            eq3_1181,
            eq4_1181,
            eq5_1181,
            eq6_1181,
            eq7_1181,
            eq8_1181,
            eq9_1181,
            eq10_1181,
            eq11_1181,
            eq12_1181,
            eq13_1181,
            eq14_1181,
            eq15_1181,
            eq16_1181,
        ],
        (
            p00,
            p01,
            p11,
            p02,
            p12,
            p03,
            p13,
            p04,
            p14,
            p05,
            p15,
            p06,
            p16,
            p07,
            p17,
            p08,
            p18,
        ),
    )

    sym_state_recursive_ratios_1181 = sym.zeros(
        buffer_capacity + 1, system_capacity + 1
    )
    sym_state_recursive_ratios_1181[0, 0] = 1
    sym_state_recursive_ratios_1181[0, 1] = sym.factor(
        sym_state_probs_1181[p01] / sym_state_probs_1181[p00]
    )  # (0,0) -> (0,1)
    sym_state_recursive_ratios_1181[1, 1] = sym.factor(
        sym_state_probs_1181[p11] / sym_state_probs_1181[p01]
    )  # (0,1) -> (1,1)
    sym_state_recursive_ratios_1181[0, 2] = sym.factor(
        sym_state_probs_1181[p02] / sym_state_probs_1181[p01]
    )  # (0,1) -> (0,2)
    sym_state_recursive_ratios_1181[1, 2] = sym.factor(
        sym_state_probs_1181[p12] / sym_state_probs_1181[p02]
    )  # (0,2) -> (1,2)
    sym_state_recursive_ratios_1181[0, 3] = sym.factor(
        sym_state_probs_1181[p03] / sym_state_probs_1181[p02]
    )  # (0,2) -> (0,3)
    sym_state_recursive_ratios_1181[1, 3] = sym.factor(
        sym_state_probs_1181[p13] / sym_state_probs_1181[p03]
    )  # (0,3) -> (1,3)
    sym_state_recursive_ratios_1181[0, 4] = sym.factor(
        sym_state_probs_1181[p04] / sym_state_probs_1181[p03]
    )  # (0,3) -> (0,4)
    sym_state_recursive_ratios_1181[1, 4] = sym.factor(
        sym_state_probs_1181[p14] / sym_state_probs_1181[p04]
    )  # (0,4) -> (1,4)
    sym_state_recursive_ratios_1181[0, 5] = sym.factor(
        sym_state_probs_1181[p05] / sym_state_probs_1181[p04]
    )  # (0,4) -> (0,5)
    sym_state_recursive_ratios_1181[1, 5] = sym.factor(
        sym_state_probs_1181[p15] / sym_state_probs_1181[p05]
    )  # (0,5) -> (1,5)
    sym_state_recursive_ratios_1181[0, 6] = sym.factor(
        sym_state_probs_1181[p06] / sym_state_probs_1181[p05]
    )  # (0,5) -> (0,6)
    sym_state_recursive_ratios_1181[1, 6] = sym.factor(
        sym_state_probs_1181[p16] / sym_state_probs_1181[p06]
    )  # (0,6) -> (1,6)
    sym_state_recursive_ratios_1181[0, 7] = sym.factor(
        sym_state_probs_1181[p07] / sym_state_probs_1181[p06]
    )  # (0,6) -> (0,7)
    sym_state_recursive_ratios_1181[1, 7] = sym.factor(
        sym_state_probs_1181[p17] / sym_state_probs_1181[p07]
    )  # (0,7) -> (1,7)
    sym_state_recursive_ratios_1181[0, 8] = sym.factor(
        sym_state_probs_1181[p08] / sym_state_probs_1181[p07]
    )  # (0,7) -> (0,8)
    sym_state_recursive_ratios_1181[1, 8] = sym.factor(
        sym_state_probs_1181[p18] / sym_state_probs_1181[p08]
    )  # (0,8) -> (1,8)

    sym_state_recursive_ratios_right_1181 = sym_state_recursive_ratios_1181.copy()
    sym_state_recursive_ratios_right_1181[1, 2] = sym.factor(
        sym_state_probs_1181[p12] / sym_state_probs_1181[p11]
    )  # (1,1) -> (1,2)
    sym_state_recursive_ratios_right_1181[1, 3] = sym.factor(
        sym_state_probs_1181[p13] / sym_state_probs_1181[p12]
    )  # (1,2) -> (1,3)
    sym_state_recursive_ratios_right_1181[1, 4] = sym.factor(
        sym_state_probs_1181[p14] / sym_state_probs_1181[p13]
    )  # (1,3) -> (1,4)
    sym_state_recursive_ratios_right_1181[1, 5] = sym.factor(
        sym_state_probs_1181[p15] / sym_state_probs_1181[p14]
    )  # (1,4) -> (1,5)
    sym_state_recursive_ratios_right_1181[1, 6] = sym.factor(
        sym_state_probs_1181[p16] / sym_state_probs_1181[p15]
    )  # (1,5) -> (1,6)
    sym_state_recursive_ratios_right_1181[1, 7] = sym.factor(
        sym_state_probs_1181[p17] / sym_state_probs_1181[p16]
    )  # (1,6) -> (1,7)
    sym_state_recursive_ratios_right_1181[1, 8] = sym.factor(
        sym_state_probs_1181[p18] / sym_state_probs_1181[p17]
    )  # (1,7) -> (1,8)

    sym_state_recursive_ratios_P0_1181 = sym.zeros(
        buffer_capacity + 1, system_capacity + 1
    )
    sym_state_recursive_ratios_P0_1181[0, 0] = 1
    sym_state_recursive_ratios_P0_1181[0, 1] = sym.factor(
        sym_state_probs_1181[p01] / sym_state_probs_1181[p00]
    )  # (0,0) -> (0,1)
    sym_state_recursive_ratios_P0_1181[1, 1] = sym.factor(
        sym_state_probs_1181[p11] / sym_state_probs_1181[p00]
    )  # (0,0) -> (1,1)
    sym_state_recursive_ratios_P0_1181[0, 2] = sym.factor(
        sym_state_probs_1181[p02] / sym_state_probs_1181[p00]
    )  # (0,0) -> (0,2)
    sym_state_recursive_ratios_P0_1181[1, 2] = sym.factor(
        sym_state_probs_1181[p12] / sym_state_probs_1181[p00]
    )  # (0,0) -> (1,2)
    sym_state_recursive_ratios_P0_1181[0, 3] = sym.factor(
        sym_state_probs_1181[p03] / sym_state_probs_1181[p00]
    )  # (0,0) -> (0,3)
    sym_state_recursive_ratios_P0_1181[1, 3] = sym.factor(
        sym_state_probs_1181[p13] / sym_state_probs_1181[p00]
    )  # (0,0) -> (1,3)
    sym_state_recursive_ratios_P0_1181[0, 4] = sym.factor(
        sym_state_probs_1181[p04] / sym_state_probs_1181[p00]
    )  # (0,0) -> (0,4)
    sym_state_recursive_ratios_P0_1181[1, 4] = sym.factor(
        sym_state_probs_1181[p14] / sym_state_probs_1181[p00]
    )  # (0,0) -> (1,4)
    sym_state_recursive_ratios_P0_1181[0, 5] = sym.factor(
        sym_state_probs_1181[p05] / sym_state_probs_1181[p00]
    )  # (0,0) -> (0,5)
    sym_state_recursive_ratios_P0_1181[1, 5] = sym.factor(
        sym_state_probs_1181[p15] / sym_state_probs_1181[p00]
    )  # (0,0) -> (1,5)
    sym_state_recursive_ratios_P0_1181[0, 6] = sym.factor(
        sym_state_probs_1181[p06] / sym_state_probs_1181[p00]
    )  # (0,0) -> (0,6)
    sym_state_recursive_ratios_P0_1181[1, 6] = sym.factor(
        sym_state_probs_1181[p16] / sym_state_probs_1181[p00]
    )  # (0,0) -> (1,6)
    sym_state_recursive_ratios_P0_1181[0, 7] = sym.factor(
        sym_state_probs_1181[p07] / sym_state_probs_1181[p00]
    )  # (0,0) -> (0,7)
    sym_state_recursive_ratios_P0_1181[1, 7] = sym.factor(
        sym_state_probs_1181[p17] / sym_state_probs_1181[p00]
    )  # (0,0) -> (1,7)
    sym_state_recursive_ratios_P0_1181[0, 8] = sym.factor(
        sym_state_probs_1181[p08] / sym_state_probs_1181[p00]
    )  # (0,0) -> (0,8)
    sym_state_recursive_ratios_P0_1181[1, 8] = sym.factor(
        sym_state_probs_1181[p18] / sym_state_probs_1181[p00]
    )  # (0,0) -> (1,8)

    return (
        sym_state_probs_1181,
        sym_state_recursive_ratios_1181,
        sym_state_recursive_ratios_right_1181,
        sym_state_recursive_ratios_P0_1181,
    )


def get_symbolic_state_probabilities_1191():
    num_of_servers = 1
    threshold = 1
    system_capacity = 9
    buffer_capacity = 1

    Q_sym_1191 = abg.markov.get_symbolic_transition_matrix(
        num_of_servers, threshold, system_capacity, buffer_capacity
    )

    (
        p00,
        p01,
        p11,
        p02,
        p12,
        p03,
        p13,
        p04,
        p14,
        p05,
        p15,
        p06,
        p16,
        p07,
        p17,
        p08,
        p18,
        p09,
        p19,
    ) = sym.symbols(
        "p00, p01, p11, p02, p12, p03, p13, p04, p14, p05, p15, p06, p16, p07, p17, p08, p18, p09, p19"
    )
    pi_1191 = sym.Matrix(
        [
            p00,
            p01,
            p11,
            p02,
            p12,
            p03,
            p13,
            p04,
            p14,
            p05,
            p15,
            p06,
            p16,
            p07,
            p17,
            p08,
            p18,
            p09,
            p19,
        ]
    )
    dimension_1191 = Q_sym_1191.shape[0]

    M_sym_1191 = sym.Matrix(
        [Q_sym_1191.transpose()[:-1, :], sym.ones(1, dimension_1191)]
    )
    sym_diff_equations_1191 = M_sym_1191 @ pi_1191

    b_sym_1191 = sym.Matrix([sym.zeros(dimension_1191 - 1, 1), [1]])

    eq0_1191 = sym.Eq(sym_diff_equations_1191[0], b_sym_1191[0])
    eq1_1191 = sym.Eq(sym_diff_equations_1191[1], b_sym_1191[1])
    eq2_1191 = sym.Eq(sym_diff_equations_1191[2], b_sym_1191[2])
    eq3_1191 = sym.Eq(sym_diff_equations_1191[3], b_sym_1191[3])
    eq4_1191 = sym.Eq(sym_diff_equations_1191[4], b_sym_1191[4])
    eq5_1191 = sym.Eq(sym_diff_equations_1191[5], b_sym_1191[5])
    eq6_1191 = sym.Eq(sym_diff_equations_1191[6], b_sym_1191[6])
    eq7_1191 = sym.Eq(sym_diff_equations_1191[7], b_sym_1191[7])
    eq8_1191 = sym.Eq(sym_diff_equations_1191[8], b_sym_1191[8])
    eq9_1191 = sym.Eq(sym_diff_equations_1191[9], b_sym_1191[9])
    eq10_1191 = sym.Eq(sym_diff_equations_1191[10], b_sym_1191[10])
    eq11_1191 = sym.Eq(sym_diff_equations_1191[11], b_sym_1191[11])
    eq12_1191 = sym.Eq(sym_diff_equations_1191[12], b_sym_1191[12])
    eq13_1191 = sym.Eq(sym_diff_equations_1191[13], b_sym_1191[13])
    eq14_1191 = sym.Eq(sym_diff_equations_1191[14], b_sym_1191[14])
    eq15_1191 = sym.Eq(sym_diff_equations_1191[15], b_sym_1191[15])
    eq16_1191 = sym.Eq(sym_diff_equations_1191[16], b_sym_1191[16])
    eq17_1191 = sym.Eq(sym_diff_equations_1191[17], b_sym_1191[17])
    eq18_1191 = sym.Eq(sym_diff_equations_1191[18], b_sym_1191[18])

    sym_state_probs_1191 = sym.solve(
        [
            eq0_1191,
            eq1_1191,
            eq2_1191,
            eq3_1191,
            eq4_1191,
            eq5_1191,
            eq6_1191,
            eq7_1191,
            eq8_1191,
            eq9_1191,
            eq10_1191,
            eq11_1191,
            eq12_1191,
            eq13_1191,
            eq14_1191,
            eq15_1191,
            eq16_1191,
            eq17_1191,
            eq18_1191,
        ],
        (
            p00,
            p01,
            p11,
            p02,
            p12,
            p03,
            p13,
            p04,
            p14,
            p05,
            p15,
            p06,
            p16,
            p07,
            p17,
            p08,
            p18,
            p09,
            p19,
        ),
    )

    sym_state_recursive_ratios_1191 = sym.zeros(
        buffer_capacity + 1, system_capacity + 1
    )
    sym_state_recursive_ratios_1191[0, 0] = 1
    sym_state_recursive_ratios_1191[0, 1] = sym.factor(
        sym_state_probs_1191[p01] / sym_state_probs_1191[p00]
    )  # (0,0) -> (0,1)
    sym_state_recursive_ratios_1191[1, 1] = sym.factor(
        sym_state_probs_1191[p11] / sym_state_probs_1191[p01]
    )  # (0,1) -> (1,1)
    sym_state_recursive_ratios_1191[0, 2] = sym.factor(
        sym_state_probs_1191[p02] / sym_state_probs_1191[p01]
    )  # (0,1) -> (0,2)
    sym_state_recursive_ratios_1191[1, 2] = sym.factor(
        sym_state_probs_1191[p12] / sym_state_probs_1191[p02]
    )  # (0,2) -> (1,2)
    sym_state_recursive_ratios_1191[0, 3] = sym.factor(
        sym_state_probs_1191[p03] / sym_state_probs_1191[p02]
    )  # (0,2) -> (0,3)
    sym_state_recursive_ratios_1191[1, 3] = sym.factor(
        sym_state_probs_1191[p13] / sym_state_probs_1191[p03]
    )  # (0,3) -> (1,3)
    sym_state_recursive_ratios_1191[0, 4] = sym.factor(
        sym_state_probs_1191[p04] / sym_state_probs_1191[p03]
    )  # (0,3) -> (0,4)
    sym_state_recursive_ratios_1191[1, 4] = sym.factor(
        sym_state_probs_1191[p14] / sym_state_probs_1191[p04]
    )  # (0,4) -> (1,4)
    sym_state_recursive_ratios_1191[0, 5] = sym.factor(
        sym_state_probs_1191[p05] / sym_state_probs_1191[p04]
    )  # (0,4) -> (0,5)
    sym_state_recursive_ratios_1191[1, 5] = sym.factor(
        sym_state_probs_1191[p15] / sym_state_probs_1191[p05]
    )  # (0,5) -> (1,5)
    sym_state_recursive_ratios_1191[0, 6] = sym.factor(
        sym_state_probs_1191[p06] / sym_state_probs_1191[p05]
    )  # (0,5) -> (0,6)
    sym_state_recursive_ratios_1191[1, 6] = sym.factor(
        sym_state_probs_1191[p16] / sym_state_probs_1191[p06]
    )  # (0,6) -> (1,6)
    sym_state_recursive_ratios_1191[0, 7] = sym.factor(
        sym_state_probs_1191[p07] / sym_state_probs_1191[p06]
    )  # (0,6) -> (0,7)
    sym_state_recursive_ratios_1191[1, 7] = sym.factor(
        sym_state_probs_1191[p17] / sym_state_probs_1191[p07]
    )  # (0,7) -> (1,7)
    sym_state_recursive_ratios_1191[0, 8] = sym.factor(
        sym_state_probs_1191[p08] / sym_state_probs_1191[p07]
    )  # (0,7) -> (0,8)
    sym_state_recursive_ratios_1191[1, 8] = sym.factor(
        sym_state_probs_1191[p18] / sym_state_probs_1191[p08]
    )  # (0,8) -> (1,8)
    sym_state_recursive_ratios_1191[0, 9] = sym.factor(
        sym_state_probs_1191[p09] / sym_state_probs_1191[p08]
    )  # (0,8) -> (0,9)
    sym_state_recursive_ratios_1191[1, 9] = sym.factor(
        sym_state_probs_1191[p19] / sym_state_probs_1191[p09]
    )  # (0,9) -> (1,9)

    sym_state_recursive_ratios_right_1191 = sym_state_recursive_ratios_1191.copy()
    sym_state_recursive_ratios_right_1191[1, 2] = sym.factor(
        sym_state_probs_1191[p12] / sym_state_probs_1191[p11]
    )  # (1,1) -> (1,2)
    sym_state_recursive_ratios_right_1191[1, 3] = sym.factor(
        sym_state_probs_1191[p13] / sym_state_probs_1191[p12]
    )  # (1,2) -> (1,3)
    sym_state_recursive_ratios_right_1191[1, 4] = sym.factor(
        sym_state_probs_1191[p14] / sym_state_probs_1191[p13]
    )  # (1,3) -> (1,4)
    sym_state_recursive_ratios_right_1191[1, 5] = sym.factor(
        sym_state_probs_1191[p15] / sym_state_probs_1191[p14]
    )  # (1,4) -> (1,5)
    sym_state_recursive_ratios_right_1191[1, 6] = sym.factor(
        sym_state_probs_1191[p16] / sym_state_probs_1191[p15]
    )  # (1,5) -> (1,6)
    sym_state_recursive_ratios_right_1191[1, 7] = sym.factor(
        sym_state_probs_1191[p17] / sym_state_probs_1191[p16]
    )  # (1,6) -> (1,7)
    sym_state_recursive_ratios_right_1191[1, 8] = sym.factor(
        sym_state_probs_1191[p18] / sym_state_probs_1191[p17]
    )  # (1,7) -> (1,8)
    sym_state_recursive_ratios_right_1191[1, 8] = sym.factor(
        sym_state_probs_1191[p18] / sym_state_probs_1191[p17]
    )  # (1,8) -> (1,9)

    sym_state_recursive_ratios_P0_1191 = sym.zeros(
        buffer_capacity + 1, system_capacity + 1
    )
    sym_state_recursive_ratios_P0_1191[0, 0] = 1
    sym_state_recursive_ratios_P0_1191[0, 1] = sym.factor(
        sym_state_probs_1191[p01] / sym_state_probs_1191[p00]
    )  # (0,0) -> (0,1)
    sym_state_recursive_ratios_P0_1191[1, 1] = sym.factor(
        sym_state_probs_1191[p11] / sym_state_probs_1191[p00]
    )  # (0,0) -> (1,1)
    sym_state_recursive_ratios_P0_1191[0, 2] = sym.factor(
        sym_state_probs_1191[p02] / sym_state_probs_1191[p00]
    )  # (0,0) -> (0,2)
    sym_state_recursive_ratios_P0_1191[1, 2] = sym.factor(
        sym_state_probs_1191[p12] / sym_state_probs_1191[p00]
    )  # (0,0) -> (1,2)
    sym_state_recursive_ratios_P0_1191[0, 3] = sym.factor(
        sym_state_probs_1191[p03] / sym_state_probs_1191[p00]
    )  # (0,0) -> (0,3)
    sym_state_recursive_ratios_P0_1191[1, 3] = sym.factor(
        sym_state_probs_1191[p13] / sym_state_probs_1191[p00]
    )  # (0,0) -> (1,3)
    sym_state_recursive_ratios_P0_1191[0, 4] = sym.factor(
        sym_state_probs_1191[p04] / sym_state_probs_1191[p00]
    )  # (0,0) -> (0,4)
    sym_state_recursive_ratios_P0_1191[1, 4] = sym.factor(
        sym_state_probs_1191[p14] / sym_state_probs_1191[p00]
    )  # (0,0) -> (1,4)
    sym_state_recursive_ratios_P0_1191[0, 5] = sym.factor(
        sym_state_probs_1191[p05] / sym_state_probs_1191[p00]
    )  # (0,0) -> (0,5)
    sym_state_recursive_ratios_P0_1191[1, 5] = sym.factor(
        sym_state_probs_1191[p15] / sym_state_probs_1191[p00]
    )  # (0,0) -> (1,5)
    sym_state_recursive_ratios_P0_1191[0, 6] = sym.factor(
        sym_state_probs_1191[p06] / sym_state_probs_1191[p00]
    )  # (0,0) -> (0,6)
    sym_state_recursive_ratios_P0_1191[1, 6] = sym.factor(
        sym_state_probs_1191[p16] / sym_state_probs_1191[p00]
    )  # (0,0) -> (1,6)
    sym_state_recursive_ratios_P0_1191[0, 7] = sym.factor(
        sym_state_probs_1191[p07] / sym_state_probs_1191[p00]
    )  # (0,0) -> (0,7)
    sym_state_recursive_ratios_P0_1191[1, 7] = sym.factor(
        sym_state_probs_1191[p17] / sym_state_probs_1191[p00]
    )  # (0,0) -> (1,7)
    sym_state_recursive_ratios_P0_1191[0, 8] = sym.factor(
        sym_state_probs_1191[p08] / sym_state_probs_1191[p00]
    )  # (0,0) -> (0,8)
    sym_state_recursive_ratios_P0_1191[1, 8] = sym.factor(
        sym_state_probs_1191[p18] / sym_state_probs_1191[p00]
    )  # (0,0) -> (1,8)
    sym_state_recursive_ratios_P0_1191[0, 9] = sym.factor(
        sym_state_probs_1191[p09] / sym_state_probs_1191[p00]
    )  # (0,0) -> (0,9)
    sym_state_recursive_ratios_P0_1191[1, 9] = sym.factor(
        sym_state_probs_1191[p19] / sym_state_probs_1191[p00]
    )  # (0,0) -> (1,9)

    return (
        sym_state_probs_1191,
        sym_state_recursive_ratios_1191,
        sym_state_recursive_ratios_right_1191,
        sym_state_recursive_ratios_P0_1191,
    )
