from network_simulations import het_net
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import copy
import time

"""
first setup the network according using the het_net class then consolidate all of the information from the network to solve the central optimization problem
"""

def test_dist_debug():
    num_users = 1
    num_antenna = 1
    step_size = 1e-4
    power_limit = 200
    interferenceConstraint = 1
    network = het_net.Het_Network(5, 5, num_users, num_antenna, interferenceConstraint
                                  , power_limit, power_vector_setup=True, random=False)
    # network.update_beam_formers()
    min_corr = copy.deepcopy(network)
    # min_corr.change_power_limit(10)
    min_corr.update_beam_formers()
    set_corr = copy.deepcopy(network)
    set_corr.update_beam_formers(set=True)
    # Choose number of iterations to allow
    num_iterations = 200
    utilities, duals, feasibility, intf = network.allocate_power_step(num_iterations, step_size)
    min_corr_utilities, min_corr_duals, min_corr_feasibility, intf = min_corr.allocate_power_step(num_iterations, step_size)
    set_utilities, set_duals, set_feasibility, intf = set_corr.allocate_power_step(num_iterations, step_size)
    duals = np.asarray(duals)
    min_corr_duals = np.asarray(min_corr_duals)
    set_duals = np.asarray(set_duals)

    network.print_layout()

    length = int(duals.shape[1]/2)+1

    print(feasibility, "\n")
    print(min_corr_feasibility, "\n")
    print(set_feasibility)


    plt.figure(2)
    plt.plot(np.arange(num_iterations + 1), utilities, label="moore-penrose")
    plt.plot(np.arange(num_iterations + 1), min_corr_utilities, label="minimized correlation")
    plt.plot(np.arange(num_iterations + 1), set_utilities, label="minimized correlation set")
    plt.legend(loc="lower left")
    plt.title(label="social_utility")
    plt.ylabel("Social Utility (User SNR)")
    plt.xlabel("Iteration")
    time_path = "Output/utility_"+f"{time.time()}"+"curves.png"
    plt.savefig(time_path, format="png")

    plt.figure(3)
    plt.subplot(length,2,1)
    plt.plot(np.arange(num_iterations + 1), utilities, label="moore-penrose")
    plt.plot(np.arange(num_iterations + 1), min_corr_utilities, label="minimized correlation")
    plt.plot(np.arange(num_iterations + 1), set_utilities, label="minimized correlation set")
    plt.legend(loc="lower left")
    plt.title(label="social_utility")

    labels = ['pow_dual', 'pos_dual', 'int_dual', 'interference', 'ave_power']
    for columns in range(duals.shape[1]):
        plt.subplot(length, 2, columns+2)
        plt.plot(duals[:, columns], label="moore-penrose")
        plt.plot(min_corr_duals[:, columns], label="minimized correlation")
        plt.plot(set_duals[:, columns], label="minimized correlation set")
        plt.title(labels[columns])
    time_path = "Output/convergence_" + f"{time.time()}" + "curves.png"
    plt.savefig(time_path, format="png")
    plt.show()

