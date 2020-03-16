from network_simulations.het_net import *
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import copy
import time

"""
first setup the network according using the het_net class then consolidate all of the information from the network to solve the central optimization problem
"""

def test_power_compare():

    pow_dual = 1e-5
    int_dual = pow_dual
    pos_dual = pow_dual
    num_users = 1
    num_antenna = 1
    step_size_pow = 1e-1
    step_size_int = step_size_pow
    userPowerList = [100]
    num_iterations = 500
    numMacroUsers = 10
    numBaseStations = 5
    interferenceThreshold = 1e-4
    userPower = userPowerList[0]
    network = HetNet(numBaseStations, numMacroUsers, num_users, num_antenna, interferenceThreshold, int_dual, pow_dual, pos_dual,
                                   userPower,
                                  power_vector_setup=True,
                                  random=False)
    # figsize = (5, 5)
    for powerLimit in userPowerList:
        fig_main = plt.figure()
        util_plt = fig_main.add_subplot(1, 3, 1)
        # util_plt.set_title("Power Convergence")
        util_plt.set_ylabel("Social Utility (System Capacity)")
        util_plt.set_xlabel("Iteration")
        extra_plt = fig_main.add_subplot(1, 3, 2)
        # extra_plt.set_title("Interference Constraint Slack")
        extra_plt.set_ylabel("Min. Power Constraint Slack")
        extra_plt.set_xlabel("Iteration")
        extra_plt1 = fig_main.add_subplot(1, 3, 3)
        # extra_plt1.set_title("Power Constraint Slack")
        extra_plt1.set_ylabel("Min. Interference Constraint Slack")
        extra_plt1.set_xlabel("Iteration")
        check = []
        currNetwork = copy.deepcopy(network)
        currNetwork.change_power_limit(powerLimit)
        utilities, duals, feasibility, constraints = currNetwork.allocate_power_step(num_iterations, step_size_pow, step_size_int)
        duals = np.asarray(duals)
        util_plt.plot(np.arange(num_iterations + 1), utilities, label=f"FBS Power: {powerLimit}")
        extra_plt.plot(np.arange(num_iterations), constraints[2], label=f"power constraint slack")
        # extra_plt.plot(np.arange(num_iterations), constraints[0], '-', label=f"interference slack")
        # extra_plt.plot(np.arange(num_iterations), constraints[0],'-', label=f"interference slack {powerLimit}")
        extra_plt1.plot(np.arange(num_iterations), constraints[0], '-', label=f"interference slack")

        # extra_plt1.plot(np.arange(num_iterations+1), duals[:, 0], '-', label=f"power slack {powerLimit}")
        # extra_plt1.plot(np.arange(num_iterations+1), duals[:, 2], label=f"int slack {powerLimit}")

        print(feasibility, "\n")
        check.append(np.asarray(utilities))

        util_plt.legend(loc="lower left")
        # extra_plt1.legend(loc="lower left")
        # extra_plt.legend(loc="lower left")

        plt.tight_layout()
        time_path = "Output/utility_"+f"{time.time()}" + f"{powerLimit}" + "curves.png"
        plt.savefig(time_path, format="png")
        # network.print_layout()
        plt.show()
    pass

