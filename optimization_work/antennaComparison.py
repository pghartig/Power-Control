from network_simulations.het_net import *
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import copy
import time

"""
first setup the network according using the het_net class then consolidate all of the information from the network to solve the central optimization problem
"""

def test_antenna_compare():
    pow_dual = 1e-3
    int_dual = pow_dual
    pos_dual = pow_dual
    num_users = 5
    step_size_pow = 1e-1
    step_size_int = step_size_pow
    numAntennaList = [1, 2, 2]
    num_iterations = 200
    numMacroUsers = 10
    numBaseStations = 5
    interferenceThreshold = 1
    userPower = 50
    num_antenna = 4
    currNetwork = HetNet(numBaseStations, numMacroUsers, num_users, num_antenna, interferenceThreshold,
                         int_dual, pow_dual, pos_dual,
                         userPower,
                         power_vector_setup=True,
                         random=False)


    cur = num_antenna
    for add_antenna in numAntennaList:
        cur += add_antenna
        currNetwork.add_antennas(add_antenna)
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
        utilities, min_utilities, max_utilities, duals, feasibility, constraints = currNetwork.allocate_power_step(num_iterations, step_size_pow, step_size_int)
        duals = np.asarray(duals)
        util_plt.plot(np.arange(num_iterations), utilities, label=f"FBS Antennas: {cur}")
        extra_plt.plot(np.arange(num_iterations), constraints[2], label=f"min.")
        extra_plt.plot(np.arange(num_iterations), constraints[3], label=f"max.")
        extra_plt1.plot(np.arange(num_iterations), constraints[0], label=f"min.")
        extra_plt1.plot(np.arange(num_iterations), constraints[1],  label=f"max.")

        print(feasibility, "\n")
        check.append(np.asarray(utilities))

        util_plt.legend(loc="lower left")
        # extra_plt1.legend(loc="lower left")
        # extra_plt.legend(loc="lower left")

        plt.tight_layout()
        time_path = "Output/utility_"+f"{time.time()}" + f"{num_antenna}" + "curves.png"
        plt.savefig(time_path, format="png")
        # network.print_layout()
    plt.show()

