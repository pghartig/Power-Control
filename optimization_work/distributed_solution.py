from network_simulations import het_net
import cvxpy as cp
"""
first setup the network according using the het_net class then consolidate all of the information from the network to solve the central optimization problem
"""

def test_central_optimization():
    network = het_net.Het_Network(10, 10, 5, 10, 1)

    # Choose number of iterations to allow
    num_iterations = 10
    network.allocate_power_step(num_iterations)

    network.print_layout()