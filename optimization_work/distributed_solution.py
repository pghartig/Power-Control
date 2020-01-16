from network_simulations import het_net
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

"""
first setup the network according using the het_net class then consolidate all of the information from the network to solve the central optimization problem
"""

def test_dist_optimization():
    num_users = 10
    num_antenna = num_users + 5
    network = het_net.Het_Network(5, 30, num_users, num_antenna, .1, power_vector_setup=True)

    # Choose number of iterations to allow
    num_iterations = 100
    utilities, duals = network.allocate_power_step(num_iterations)
    network.print_layout()
    plt.figure()
    plt.plot(np.arange(num_iterations+1), utilities)
    plt.figure()
    duals = np.asarray(duals)
    for columns in range(duals.shape[1]):
        plt.plot(duals[:, columns])
    plt.show()

