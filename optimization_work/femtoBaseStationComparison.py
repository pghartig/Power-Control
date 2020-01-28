from network_simulations import het_net
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import copy
import time

"""
first setup the network according using the het_net class then consolidate all of the information from the network to solve the central optimization problem
"""

def test_femto_compare():
    num_users = 2
    num_femto = 5
    num_antenna = 4
    step_size = 1e-3
    numFemtoUserList = [10, 15, 20]
    previousNumberUsers = numFemtoUserList[0]
    numMacroUsers = 20
    num_iterations = 100
    interferenceThreshold = 10
    powerLimit = 1
    network = het_net.Het_Network(previousNumberUsers, numMacroUsers, num_users, num_antenna,
                                  interferenceThreshold=interferenceThreshold, power_limit=powerLimit,
                                  power_vector_setup=True,
                                  random=False)
    plt.figure(2)
    currNetwork = copy.deepcopy(network)
    for numFemtoUsers in numFemtoUserList:
        currNetwork = copy.deepcopy(currNetwork)
        currNetwork.addFemtoBaseStation(numFemtoUsers - previousNumberUsers,  num_users, num_antenna,
                                        powerVectorSetup=True, powerLimit=powerLimit, random=False)
        workingCopy = copy.deepcopy(currNetwork)
        workingCopy.update_beam_formers()
        utilities, duals, feasibility = workingCopy.allocate_power_step(num_iterations, step_size)
        # duals = np.asarray(duals)
        previousNumberUsers = numFemtoUsers
        plt.plot(np.arange(num_iterations + 1), utilities, label=f"{numFemtoUsers}")
        print(feasibility, "\n")


    plt.legend(loc="lower left")
    plt.title(label="social_utility")
    plt.ylabel("Social Utility (User SNR)")
    plt.xlabel("Iteration")
    time_path = "Output/utility_"+f"{time.time()}"+"curves.png"
    plt.savefig(time_path, format="png")

    # network.print_layout()

    plt.show()

