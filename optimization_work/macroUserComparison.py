from network_simulations import het_net
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import copy
import time

"""
first setup the network according using the het_net class then consolidate all of the information from the network to solve the central optimization problem
"""

def test_macro_compare():
    num_users = 2
    num_antenna = 8
    step_size = 1e-2
    numMacroUserList = [40, 70, 90]
    previousNumberUsers = numMacroUserList[0]
    num_iterations = 200
    interferenceThreshold = 10
    numMacroUsers = numMacroUserList[0]
    network = het_net.Het_Network(5, numMacroUsers, num_users, num_antenna,
                                  interferenceThreshold=interferenceThreshold, power_limit=5,
                                  power_vector_setup=True,
                                  random=False)
    plt.figure(2)
    currNetwork = copy.deepcopy(network)
    for numMacroUsers in numMacroUserList:
        currNetwork = copy.deepcopy(currNetwork)
        currNetwork.add_macro_users(numMacroUsers - previousNumberUsers, interferenceThreshold)
        workingCopy = copy.deepcopy(currNetwork)
        previousNumberUsers = numMacroUsers
        # workingCopy.update_beam_formers()
        utilities, duals, feasibility = workingCopy.allocate_power_step(num_iterations, step_size)
        # duals = np.asarray(duals)
        previousNumberUsers = numMacroUsers
        plt.plot(np.arange(num_iterations + 1), utilities, label=f"{numMacroUsers}")
        print(feasibility, "\n")


    plt.legend(loc="lower left")
    plt.title(label="social_utility")
    plt.ylabel("Social Utility (User SNR)")
    plt.xlabel("Iteration")
    time_path = "Output/utility_"+f"{time.time()}"+"curves.png"
    plt.savefig(time_path, format="png")

    # network.print_layout()

    plt.show()

