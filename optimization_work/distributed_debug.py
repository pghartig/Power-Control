from network_simulations import het_net
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import copy

"""
first setup the network according using the het_net class then consolidate all of the information from the network to solve the central optimization problem
"""

def test_dist_debug():
    num_users = 10
    num_antenna = num_users + 10
    step_size = 1e-2
    network = het_net.Het_Network(5, 40, num_users, num_antenna, 1, 10, power_vector_setup=True)
    # network.update_beam_formers()
    for_comp = copy.deepcopy(network)
    # for_comp.change_power_limit(10)
    # for_comp.update_beam_formers(set=True)
    for_comp.update_beam_formers()
    # Choose number of iterations to allow
    num_iterations = 200
    utilities, duals, feasibility = network.allocate_power_step(num_iterations, step_size)
    test_utilities, test_duals, test_feasibility = for_comp.allocate_power_step(num_iterations, step_size)
    duals = np.asarray(duals)
    test_duals = np.asarray(test_duals)

    network.print_layout()

    length = int(duals.shape[1]/2)+1

    print(feasibility, test_feasibility)

    plt.figure()
    plt.subplot(length,2,1)
    plt.plot(np.arange(num_iterations + 1), utilities, label="base")
    plt.plot(np.arange(num_iterations + 1), test_utilities, label="comparison")
    plt.legend(loc="lower left")
    plt.title(label="social_utility")

    labels = ['pow_dual','pos_dual','int_dual','interference','ave_power']
    for columns in range(test_duals.shape[1]):
        plt.subplot(length, 2, columns+2)
        plt.plot(duals[:, columns], label="base")
        plt.plot(test_duals[:, columns], label="test")
        plt.title(labels[columns])
        plt.legend(loc="lower left")
    plt.show()

