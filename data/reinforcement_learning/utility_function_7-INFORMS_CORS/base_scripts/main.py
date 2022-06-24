"""Reinforcement learning algorithm with e = 0"""
# pylint: disable=invalid-name
import copy
import sys

import numpy as np

from utils import (
    utility_function_7,
    run_simulation,
    pick_policy,
    accept_policy,
    are_within_limits,
    update_server_utilities,
    output_to_file,
)


def team_expertise_priority(srv, ind):  # pylint: disable=unused-argument
    """
    Servers priority based on expertise
    """
    if srv.id_number == 1:
        return 0
    if srv.id_number == 4:
        return 1
    return np.random.random()


def main(e_parameter, lambda_1=None, lambda_2=None, mu=None):
    """The main function"""
    # Model parameters
    if lambda_1 is None:
        lambda_1 = 0.5
    if lambda_2 is None:
        lambda_2 = 1
    if mu is None:
        mu = 0.7

    num_of_servers = 4
    threshold = 7
    system_capacity = 10
    buffer_capacity = 7

    rates = {}
    # for server_id in range(1, num_of_servers + 1):
    #     rates[server_id] = {}
    #     for u in range(buffer_capacity + 1):
    #         for v in range(system_capacity + 1):
    #             if v >= threshold or u == 0:  # hope I don't regret adding this
    #                 rates[server_id][(u, v)] = mu
    rates = {
        1: {
            (0, 0): 1.857212831491963e-10,
            (0, 1): 9.829250307419246,
            (0, 2): 9.966419119464636,
            (0, 3): 7.758644765957512,
            (0, 4): 9.796510208180443,
            (0, 5): 0.0002832620433827927,
            (0, 6): 0.5959398629220433,
            (0, 7): 0.05143943036535784,
            (0, 8): 0.6755039557338152,
            (0, 9): 0.6696894547207239,
            (0, 10): 0.7,
            (1, 7): 1.2231017041018992,
            (1, 8): 1.3630231870740324,
            (1, 9): 0.8360808754540313,
            (1, 10): 0.014104342511269128,
            (2, 7): 0.9308548561436386,
            (2, 8): 0.7,
            (2, 9): 0.8656962770241506,
            (2, 10): 0.7,
            (3, 7): 1.0285415175784764,
            (3, 8): 0.8417482776196715,
            (3, 9): 0.7,
            (3, 10): 0.19731953623189227,
            (4, 7): 1.0245445064797432,
            (4, 8): 0.7,
            (4, 9): 1.143244744716609,
            (4, 10): 1.0063562857498898,
            (5, 7): 0.7,
            (5, 8): 0.22731459280997923,
            (5, 9): 0.7,
            (5, 10): 0.7,
            (6, 7): 1.3253248021053188,
            (6, 8): 0.7,
            (6, 9): 0.7,
            (6, 10): 0.7,
            (7, 7): 0.293593068135948,
            (7, 8): 0.7,
            (7, 9): 0.7,
            (7, 10): 0.7,
        },
        2: {
            (0, 0): 1.02108525068257e-08,
            (0, 1): 1.3789177446521946e-14,
            (0, 2): 9.986016422564258,
            (0, 3): 9.984976877256226,
            (0, 4): 0.06826258913325856,
            (0, 5): 0.0006566365181224989,
            (0, 6): 0.3542825289949733,
            (0, 7): 0.9055212957772897,
            (0, 8): 1.0368585318493375,
            (0, 9): 0.7,
            (0, 10): 0.7,
            (1, 7): 0.7247767642952159,
            (1, 8): 0.34280286511383967,
            (1, 9): 0.7,
            (1, 10): 0.3719640254921561,
            (2, 7): 1.103681609150602,
            (2, 8): 0.3393746320198415,
            (2, 9): 0.7938317457054868,
            (2, 10): 0.7,
            (3, 7): 0.7,
            (3, 8): 1.2870035294722308,
            (3, 9): 1.135823608255728,
            (3, 10): 0.7,
            (4, 7): 0.7,
            (4, 8): 0.7,
            (4, 9): 0.7,
            (4, 10): 0.7,
            (5, 7): 0.7,
            (5, 8): 1.2459562543923948,
            (5, 9): 0.7,
            (5, 10): 0.7,
            (6, 7): 0.7,
            (6, 8): 0.7,
            (6, 9): 0.7,
            (6, 10): 0.7,
            (7, 7): 0.7,
            (7, 8): 0.7,
            (7, 9): 0.7,
            (7, 10): 0.9392962613682695,
        },
        3: {
            (0, 0): 1.6069074481877692e-21,
            (0, 1): 1.5341640668511688e-14,
            (0, 2): 9.886106292391327,
            (0, 3): 8.9397196668379,
            (0, 4): 4.770849872696185,
            (0, 5): 0.13032156649071744,
            (0, 6): 0.06113410438883264,
            (0, 7): 0.3148695116528211,
            (0, 8): 0.0317474496592404,
            (0, 9): 0.7,
            (0, 10): 0.7,
            (1, 7): 0.3446938570681205,
            (1, 8): 0.8625616033210646,
            (1, 9): 0.47401147366403285,
            (1, 10): 0.7,
            (2, 7): 0.7,
            (2, 8): 0.808215513624737,
            (2, 9): 0.033445317433309335,
            (2, 10): 0.9636712577998997,
            (3, 7): 1.1727889796802924,
            (3, 8): 0.23062895168915026,
            (3, 9): 0.7,
            (3, 10): 0.4457170776567797,
            (4, 7): 0.38714528620774247,
            (4, 8): 0.7,
            (4, 9): 0.48045923496715526,
            (4, 10): 0.7,
            (5, 7): 0.7,
            (5, 8): 0.7,
            (5, 9): 0.7,
            (5, 10): 0.7,
            (6, 7): 0.04837216365783392,
            (6, 8): 0.7,
            (6, 9): 0.3469797932084303,
            (6, 10): 0.7,
            (7, 7): 0.7,
            (7, 8): 0.7,
            (7, 9): 0.7,
            (7, 10): 0.7,
        },
        4: {
            (0, 0): 3.3173391749900726e-14,
            (0, 1): 2.3936783123548787e-13,
            (0, 2): 1.672342662163624e-07,
            (0, 3): 1.1632207276779377e-10,
            (0, 4): 9.790813342928042,
            (0, 5): 4.538381806325202e-05,
            (0, 6): 1.4689551570835802,
            (0, 7): 2.424717034337268,
            (0, 8): 1.0775558593883183,
            (0, 9): 0.218159399657841,
            (0, 10): 0.7,
            (1, 7): 0.7,
            (1, 8): 1.1098167928175382,
            (1, 9): 0.5833880260236037,
            (1, 10): 0.7,
            (2, 7): 0.7,
            (2, 8): 1.4687559386719966,
            (2, 9): 0.7,
            (2, 10): 0.7,
            (3, 7): 0.7,
            (3, 8): 0.7,
            (3, 9): 0.7,
            (3, 10): 0.7,
            (4, 7): 1.0219381342005964,
            (4, 8): 0.22944319929738374,
            (4, 9): 0.7,
            (4, 10): 0.7,
            (5, 7): 1.16797522062088,
            (5, 8): 0.5503373917957836,
            (5, 9): 0.7,
            (5, 10): 0.7,
            (6, 7): 1.246598328835758,
            (6, 8): 0.7,
            (6, 9): 0.7,
            (6, 10): 0.7,
            (7, 7): 1.0551706348861802,
            (7, 8): 0.6106248831514205,
            (7, 9): 0.7,
            (7, 10): 0.7,
        },
    }

    # Model parameters in dictionary format
    parameters = {
        "lambda_1": lambda_1,
        "lambda_2": lambda_2,
        "num_of_servers": num_of_servers,
        "threshold": threshold,
        "system_capacity": system_capacity,
        "buffer_capacity": buffer_capacity,
        "runtime": 2000,
        "num_of_trials": 10,
        "seed_num": 0,
        "server_priority_function": team_expertise_priority,
    }

    # Reinforcement learning parameters
    initial_rates = copy.deepcopy(rates)
    server_utilities = [
        -float("inf") for srv in range(1, parameters["num_of_servers"] + 1)
    ]
    num_of_iterations = 5000
    current_utility_function = utility_function_7

    # Define the directory name that the results will be stored in
    key_params = [
        "lambda_1",
        "lambda_2",
        "num_of_servers",
        "threshold",
        "system_capacity",
        "buffer_capacity",
    ]
    filename_parameters = {
        key: value for key, value in parameters.items() if key in key_params
    }
    filepath = (
        "results/e="
        + str(e_parameter)
        + ","
        + str(filename_parameters)
        .replace(" ", "")
        .replace("'", "")
        .replace(":", "=")
        .replace("{", "")
        .replace("}", "")
        + ",mu="
        + str(mu)
    )

    # Reinforcement learning algorithm
    Qs, state_probs = run_simulation(parameters, initial_rates)
    for _ in range(num_of_iterations):
        new_rates, server_id = pick_policy(Qs, rates, num_of_servers)
        Qs, new_state_probs = run_simulation(parameters, new_rates)
        if are_within_limits(new_rates) and accept_policy(
            Qs=Qs,
            utility_function=current_utility_function,
            e_parameter=e_parameter,
            server_utilities=server_utilities,
            current_server_id=server_id,
        ):
            server_utilities = update_server_utilities(
                Qs=Qs,
                utility_function=current_utility_function,
                e_parameter=e_parameter,
                num_of_servers=parameters["num_of_servers"],
            )
            rates = copy.deepcopy(new_rates)
            state_probs = copy.deepcopy(new_state_probs)

        output_to_file(str(server_utilities), filepath + "/utilities.csv")

        str_rates = str([list(rates[server].values()) for server in rates])
        output_to_file(str_rates, filepath + "/rates.csv")

        str_state_probs = str(list(state_probs.flatten()))
        output_to_file(str_state_probs, filepath + "/state_probs.csv")


if __name__ == "__main__":
    arguments = sys.argv
    arguments = [
        float(arguments[i]) if i < len(arguments) and arguments[i] != "None" else None
        for i in range(1, 5)
    ]
    e_value = arguments[0]
    lambda_1_value = arguments[1]
    lambda_2_value = arguments[2]
    mu_value = arguments[3]

    if e_value < 0 or e_value > 1:
        raise ValueError("Please provide a value of e between 0 and 1")

    main(
        e_parameter=e_value,
        lambda_1=lambda_1_value,
        lambda_2=lambda_2_value,
        mu=mu_value,
    )
