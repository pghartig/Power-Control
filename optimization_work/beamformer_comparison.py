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
    noisePowers = 10
    numMacroUsers = 5
    pow_dual = 1
    int_dual = 10
    pos_dual = 1e-5
    num_users = 5
    num_antenna = 10
    step_size_pow = 1e-4
    step_size_int = 10
    num_iterations = 300
    numBaseStations = 5
    interferenceThreshold = .1
    userPower = 200
    network = het_net.Het_Network(numBaseStations, numMacroUsers, num_users, num_antenna, interferenceThreshold, int_dual, pow_dual, pos_dual,
                                   userPower,
                                  power_vector_setup=True,
                                  random=False)
    imperfect_optimized = copy.deepcopy(network)
    # min_corr.change_power_limit(10)
    imperfect_optimized.update_beam_formers(csi=True, imperfectCsiNoisePower=noisePowers)
    min_correlation = copy.deepcopy(network)
    min_correlation.update_beam_formers(optimize=True, imperfectCsiNoisePower=noisePowers)
    # Choose number of iterations to allow
    network.update_beam_formers(imperfectCsiNoisePower=noisePowers)
    utilities, duals, feasibility, intf = network.allocate_power_step(num_iterations, step_size_pow, step_size_int)
    min_corr_utilities, min_corr_duals, min_corr_feasibility, intf = min_correlation.allocate_power_step(num_iterations, step_size_pow, step_size_int)
    csi_utilities, cse_duals, cse_feasibility, intf = imperfect_optimized.allocate_power_step(num_iterations, step_size_pow, step_size_int)
    duals = np.asarray(duals)
    min_corr_duals = np.asarray(min_corr_duals)
    set_duals = np.asarray(cse_duals)

    # network.print_layout()

    length = int(duals.shape[1]/2)+1

    print(feasibility, "\n")
    print(min_corr_feasibility, "\n")
    print(cse_feasibility)


    plt.figure(2)
    plt.plot(np.arange(num_iterations + 1), utilities, label="moore-penrose")
    plt.plot(np.arange(num_iterations + 1), min_corr_utilities, label="minimized correlation")
    plt.plot(np.arange(num_iterations + 1), csi_utilities, label="csi set")
    plt.legend(loc="lower left")
    plt.title(label="social_utility")
    plt.ylabel("Social Utility (User SNR)")
    plt.xlabel("Iteration")
    time_path = "Output/utility_"+f"{time.time()}"+"curves.png"
    # plt.savefig(time_path, format="png")

    plt.figure(3)
    plt.subplot(length,2,1)
    plt.plot(np.arange(num_iterations + 1), utilities, label="moore-penrose")
    plt.plot(np.arange(num_iterations + 1), min_corr_utilities, label="minimized correlation")
    plt.plot(np.arange(num_iterations + 1), csi_utilities, label="csi")
    plt.legend(loc="lower left")
    plt.title(label="social_utility")

    labels = ['pow_dual', 'pos_dual', 'int_dual', 'interference', 'ave_power']
    for columns in range(duals.shape[1]):
        plt.subplot(length, 2, columns+2)
        plt.plot(duals[:, columns], label="moore-penrose")
        plt.plot(min_corr_duals[:, columns], label="minimized correlation")
        plt.plot(set_duals[:, columns], label="csi set")
        plt.title(labels[columns])
    time_path = "Output/convergence_" + f"{time.time()}" + "curves.png"
    # plt.savefig(time_path, format="png")
    plt.show()

