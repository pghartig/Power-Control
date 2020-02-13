from network_simulations import het_net
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import copy
import time

"""
first setup the network according using the het_net class then consolidate all of the information from the network to solve the central optimization problem
"""

def test_antenna_compare():
    numAntennaList = [10, 20]
    numMacroUsers = 10
    # int_dual = 1e-1
    pow_dual = 1
    int_dual = 10
    # pow_dual = 1
    pos_dual = 1e-5
    num_users = 10
    step_size_pow = 1e-5
    step_size_int = 1e-1
    step_size_int = 1
    num_iterations = 5000
    numBaseStations = 5
    interferenceThreshold = 1
    userPower = 200
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
    for numAntenna in numAntennaList:
        currNetwork = het_net.Het_Network(numBaseStations, numMacroUsers, num_users, numAntenna, interferenceThreshold,
                                      int_dual, pow_dual, pos_dual,
                                      userPower,
                                      power_vector_setup=True,
                                      random=False)

        # currNetwork.update_beam_formers()
        utilities, duals, feasibility, constraints = currNetwork.allocate_power_step(num_iterations, step_size_pow, step_size_int)
        util_plt.plot(np.arange(num_iterations + 1), utilities, label=f"{numAntenna}")
        extra_plt.plot(np.arange(num_iterations), constraints[0], '-', label=f"interference slack {numAntenna}")
        extra_plt.plot(np.arange(num_iterations), constraints[1], '-', label=f"interference slack {numAntenna}")
        extra_plt1.plot(np.arange(num_iterations), constraints[2], label=f"power slack {numAntenna}")
        extra_plt1.plot(np.arange(num_iterations), constraints[3], label=f"power slack {numAntenna}")

    util_plt.legend(loc="lower left")
    time_path = "Output/utility_"+f"{time.time()}"+"curves.png"
    plt.savefig(time_path, format="png")

    plt.show()
    pass

