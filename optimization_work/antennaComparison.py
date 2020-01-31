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
    num_users = 5
    step_size = 1e-2
    numAntennaList = [10, 30, 50]
    previousNumberAntenna = numAntennaList[0]
    num_antenna = previousNumberAntenna
    num_iterations = 500
    numMacroUsers = 10
    interferenceThreshold = 10
    userPower = 100
    network = het_net.Het_Network(10, numMacroUsers, num_users, num_antenna,
                                  interferenceThreshold=interferenceThreshold, power_limit=userPower,
                                  power_vector_setup=True,
                                  random=False)
    # figsize = (5, 5)
    fig_main = plt.figure()
    util_plt = fig_main.add_subplot(1, 2, 1)
    util_plt.set_title("Power Convergence")
    util_plt.set_ylabel("Social Utility (System Capacity)")
    util_plt.set_xlabel("Iteration")
    extra_plt = fig_main.add_subplot(1, 2, 2)
    extra_plt.set_title("Interference Constraint Slack")
    extra_plt.set_ylabel("Average Constraint Slack ")
    extra_plt.set_xlabel("Iteration")
    check = []
    for numAntenna in numAntennaList:
        workingCopy = het_net.Het_Network(10, numMacroUsers, num_users, numAntenna,
                                      interferenceThreshold=interferenceThreshold, power_limit=userPower,
                                      power_vector_setup=True,
                                      random=False)

        workingCopy.update_beam_formers()
        utilities, duals, feasibility, constraints = workingCopy.allocate_power_step(num_iterations, step_size)
        util_plt.plot(np.arange(num_iterations + 1), utilities, label=f"{numAntenna}")
        extra_plt.plot(np.arange(num_iterations), constraints, label=f"{numAntenna}")
        print(feasibility, "\n")
        check.append(np.asarray(utilities))

    util_plt.legend(loc="lower left")
    time_path = "Output/utility_"+f"{time.time()}"+"curves.png"
    plt.savefig(time_path, format="png")

    network.print_layout()

    plt.show()
    pass

