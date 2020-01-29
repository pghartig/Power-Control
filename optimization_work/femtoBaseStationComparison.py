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
    #Make sure to switch to average utility in this case to make a fair comparison
    num_users = 2
    num_antenna = 4
    step_size = 1e-2
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

    fig_main = plt.figure()
    util_plt = fig_main.add_subplot(2, 1, 1)
    util_plt.set_title("Power Convergence")
    util_plt.set_ylabel("Social Utility (System Capacity)")
    util_plt.set_xlabel("Iteration")
    extra_plt = fig_main.add_subplot(2, 1, 2)
    extra_plt.set_title("Interference Constraint Slack")
    extra_plt.set_ylabel("Average Constraint Slack ")
    extra_plt.set_xlabel("Iteration")
    currNetwork = copy.deepcopy(network)
    check = []

    for numFemtoUsers in numFemtoUserList:
        currNetwork = copy.deepcopy(currNetwork)
        currNetwork.addFemtoBaseStation(numFemtoUsers - previousNumberUsers,  num_users, num_antenna,
                                        powerVectorSetup=True, powerLimit=powerLimit, random=False)
        workingCopy = copy.deepcopy(currNetwork)
        # workingCopy.update_beam_formers()
        utilities, duals, feasibility, constraints = workingCopy.allocate_power_step(num_iterations, step_size)
        # duals = np.asarray(duals)
        previousNumberUsers = numFemtoUsers
        util_plt.plot(np.arange(num_iterations + 1), utilities, label=f"{numFemtoUsers}")
        extra_plt.plot(np.arange(num_iterations), constraints, label=f"{numFemtoUsers}")
        print(feasibility, "\n")
        check.append(np.asarray(utilities))


    check = np.asarray(check)
    util_plt.legend(loc="lower left")
    time_path = "Output/utility_"+f"{time.time()}"+"curves.png"
    plt.savefig(time_path, format="png")

    network.print_layout()

    plt.show()
