from network_simulations import het_net
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import copy
import time

"""
first setup the network according using the het_net class then consolidate all of the information from the network to solve the central optimization problem
"""

def test_interference_constraint_compare():
    num_users = 3
    num_antenna = 3
    step_size = 1e-2
    interferenceConstList = [10, 20, 30, 100]
    interferenceConst = interferenceConstList[0]
    num_iterations = 200
    numMacroUsers = 50
    userPower = 5
    network = het_net.Het_Network(5, numMacroUsers, num_users, num_antenna,
                                  interferenceThreshold=interferenceConst, power_limit=userPower,
                                  power_vector_setup=True,
                                  random=False)
    plt.figure(2)
    currNetwork = copy.deepcopy(network)
    for interferenceConst in interferenceConstList:
        currNetwork = copy.deepcopy(currNetwork)
        currNetwork.change_interference_constraint(interferenceConst)
        workingCopy = copy.deepcopy(currNetwork)
        utilities, duals, feasibility = workingCopy.allocate_power_step(num_iterations, step_size)
        # duals = np.asarray(duals)
        plt.plot(np.arange(num_iterations + 1), utilities, label=f"{interferenceConst}")
        print(feasibility, "\n")


    plt.legend(loc="lower left")
    plt.title(label="social_utility")
    plt.ylabel("Social Utility (User SNR)")
    plt.xlabel("Iteration")
    time_path = "Output/utility_"+f"{time.time()}"+"curves.png"
    plt.savefig(time_path, format="png")

    # network.print_layout()

    plt.show()

