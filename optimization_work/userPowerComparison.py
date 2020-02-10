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

    int_dual = 1e-3
    pow_dual = 1e-3
    pos_dual = 1e-6
    num_users = 1
    num_antenna = 1
    step_size = 1e-3
    userPowerList = [10, 50]
    # userPowerList = [5, 10, 30]
    previousNumberUsers = userPowerList[0]
    num_iterations = 2000
    numMacroUsers = 5
    interferenceThreshold = 10
    userPower = userPowerList[0]
    network = het_net.Het_Network(5, numMacroUsers, num_users, num_antenna, interferenceThreshold, int_dual, pow_dual, pos_dual,
                                   userPower,
                                  power_vector_setup=True,
                                  random=False)
    # figsize = (5, 5)
    fig_main = plt.figure()
    util_plt = fig_main.add_subplot(1, 3, 1)
    util_plt.set_title("Power Convergence")
    util_plt.set_ylabel("Social Utility (System Capacity)")
    util_plt.set_xlabel("Iteration")
    extra_plt = fig_main.add_subplot(1, 3, 2)
    extra_plt.set_title("Interference Constraint Slack")
    extra_plt.set_ylabel("Average Constraint Slack ")
    extra_plt.set_xlabel("Iteration")
    extra_plt1 = fig_main.add_subplot(1, 3, 3)
    extra_plt1.set_title("Power Constraint Slack")
    extra_plt1.set_ylabel("Average Constraint Slack ")
    extra_plt1.set_xlabel("Iteration")
    currNetwork = copy.deepcopy(network)
    check = []
    for powerLimit in userPowerList:
        currNetwork = copy.deepcopy(currNetwork)
        currNetwork.change_power_limit(powerLimit)
        workingCopy = copy.deepcopy(currNetwork)
        utilities, duals, feasibility, constraints = workingCopy.allocate_power_step(num_iterations, step_size)
        duals = np.asarray(duals)
        util_plt.plot(np.arange(num_iterations + 1), utilities, label=f"{powerLimit}")
        extra_plt.plot(np.arange(num_iterations), constraints[0], label=f"min interference {powerLimit}")
        extra_plt1.plot(np.arange(num_iterations + 1), duals[:, 2], label=f"max interference dual {powerLimit}")

        # extra_plt.plot(np.arange(num_iterations), constraints[1], label=f"min power {powerLimit}")

        print(feasibility, "\n")
        check.append(np.asarray(utilities))

    util_plt.legend(loc="lower left")
    time_path = "Output/utility_"+f"{time.time()}"+"curves.png"
    plt.savefig(time_path, format="png")

    # network.print_layout()

    plt.show()
    pass

