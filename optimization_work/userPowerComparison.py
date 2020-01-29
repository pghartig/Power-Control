from network_simulations import het_net
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import copy
import time

"""
first setup the network according using the het_net class then consolidate all of the information from the network to solve the central optimization problem
"""

def test_power_compare():
    num_users = 1
    num_antenna = 1
    step_size = 1e-3
    userPowerList = [1, 10, 15, 25]
    previousNumberUsers = userPowerList[0]
    num_iterations = 400
    numMacroUsers = 50
    interferenceThreshold = 1
    userPower = userPowerList[0]
    network = het_net.Het_Network(10, numMacroUsers, num_users, num_antenna,
                                  interferenceThreshold=interferenceThreshold, power_limit=userPower,
                                  power_vector_setup=True,
                                  random=False)
    plt.figure(2)
    currNetwork = copy.deepcopy(network)
    check  = []
    for powerLimit in userPowerList:
        currNetwork = copy.deepcopy(currNetwork)
        currNetwork.change_power_limit(powerLimit)
        workingCopy = copy.deepcopy(currNetwork)
        utilities, duals, feasibility, constraints = workingCopy.allocate_power_step(num_iterations, step_size)
        # duals = np.asarray(duals)
        previousNumberUsers = powerLimit
        plt.subplot(2, 1, 1)
        plt.plot(np.arange(num_iterations + 1), utilities, label=f"{powerLimit}")
        plt.subplot(2, 1, 2)
        plt.plot(np.arange(num_iterations), constraints, label=f"{powerLimit}")
        print(feasibility, "\n")
        check.append(np.asarray(utilities))

    check = np.asarray(check)
    plt.legend(loc="lower left")
    plt.title(label="social_utility")
    plt.ylabel("Social Utility (User SNR)")
    plt.xlabel("Iteration")
    time_path = "Output/utility_"+f"{time.time()}"+"curves.png"
    plt.savefig(time_path, format="png")

    # network.print_layout()

    plt.show()

