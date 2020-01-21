from network_simulations import het_net
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import copy

"""
first setup the network according using the het_net class then consolidate all of the information from the network to solve the central optimization problem
"""

def test_dist_optimization():
    num_users = 10
    num_antenna = num_users + 5
    network = het_net.Het_Network(5, 30, num_users, num_antenna, .1, power_vector_setup=True)
    # network.update_beam_formers()
    for_comp = copy.deepcopy(network)
    for_comp.update_beam_formers()
    # Choose number of iterations to allow
    num_iterations = 50
    utilities, duals = network.allocate_power_step(num_iterations)
    test_utilities, test_duals = for_comp.allocate_power_step(num_iterations)

    network.print_layout()
    plt.figure()
    plt.plot(np.arange(num_iterations+1), utilities, label = "moore-pensose")
    plt.plot(np.arange(num_iterations+1), test_utilities, label = "optimized")
    plt.legend(loc = "lower left")
    plt.figure()
    duals = np.asarray(duals)
    for columns in range(duals.shape[1]):
        plt.plot(duals[:, columns])

    plt.figure()
    test_duals = np.asarray(test_duals)
    for columns in range(test_duals.shape[1]):
        plt.plot(test_duals[:, columns])


    plt.show()

