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
    numMacroUserList = [5, 10]
    numMacroUsers = previousNumberUsers = numMacroUserList[0]
    # int_dual = 1e-1
    pow_dual = 1
    int_dual = 10
    # pow_dual = 1
    pos_dual = 1e-5
    num_users = 1
    num_antenna = 1
    step_size_pow = 1e-5
    step_size_int = 1e-1
    step_size_int = 1
    # userPowerList = [5, 10, 30]
    num_iterations = 10000
    numBaseStations = 5
    interferenceThreshold = 1
    userPower = 100
    network = het_net.Het_Network(numBaseStations, numMacroUsers, num_users, num_antenna, interferenceThreshold, int_dual, pow_dual, pos_dual,
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
    extra_plt.set_title("Min Interference Constraint Slack")
    extra_plt.set_ylabel("Average Constraint Slack ")
    extra_plt.set_xlabel("Iteration")
    extra_plt1 = fig_main.add_subplot(1, 3, 3)
    extra_plt1.set_title("Min Power Constraint Slack")
    extra_plt1.set_ylabel("Average Constraint Slack ")
    extra_plt1.set_xlabel("Iteration")
    currNetwork = copy.deepcopy(network)
    for numMacroUsers in numMacroUserList:
        currNetwork = copy.deepcopy(currNetwork)
        currNetwork.add_macro_users(numMacroUsers - previousNumberUsers, interferenceThreshold, int_dual)
        utilities, duals, feasibility, constraints = currNetwork.allocate_power_step(num_iterations, step_size_pow, step_size_int)
        duals = np.asarray(duals)
        previousNumberUsers = numMacroUsers
        util_plt.plot(np.arange(num_iterations + 1), utilities, label=f"{numMacroUsers}")
        extra_plt.plot(np.arange(num_iterations), constraints[0],'-' ,label=f"interference slack {numMacroUsers}")
        # extra_plt.plot(np.arange(num_iterations), constraints[1],'-', label=f"interference slack {numMacroUsers}")
        extra_plt1.plot(np.arange(num_iterations), constraints[2], label=f"power slack {numMacroUsers}")
        # extra_plt1.plot(np.arange(num_iterations), constraints[3], label=f"power slack {numMacroUsers}")


    util_plt.legend(loc="lower left")
    time_path = "Output/utility_"+f"{time.time()}"+"curves.png"
    plt.savefig(time_path, format="png")
    # network.print_layout()
    plt.show()
    pass

