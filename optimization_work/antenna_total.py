from network_simulations.het_net import *
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import copy
import time

"""
first setup the network according using the het_net class then consolidate all of the information from the network to 
solve the central optimization problem
"""

pow_dual = 1e-2
int_dual = pow_dual
pos_dual = pow_dual
num_users = 1
step_size_pow = 1e-2
step_size_int = step_size_pow
numAntennaList = [5]
num_iterations = 100
numMacroUsers = 6
numBaseStations = 3
# interferenceThreshold = 1
# userPower = 100
interferenceThreshold = 100
userPower = 1
num_antenna = 1
currNetwork = HetNet(numBaseStations, numMacroUsers, num_users, num_antenna, interferenceThreshold,
                     int_dual, pow_dual, pos_dual,
                     userPower,
                     power_vector_setup=True,
                     random=False)
central = []
distributed = []
central_utilities = []
distributed_utilities = []
difference = []

cur = num_antenna
for ind, add_antenna in enumerate(numAntennaList):
    cur += add_antenna
    # currNetwork.add_antennas(add_antenna)
    distributed_net = copy.deepcopy(currNetwork)

    check = []
    utilities, min_utilities, max_utilities, duals, feasibility, constraints =\
        distributed_net.allocate_power_step(num_iterations, step_size_pow, step_size_int)
    distributed.append(distributed_net.distributed)
    distributed_utilities.append(distributed_net.get_social_utility())

    print(f"Distributed Feasibility: {distributed_net.verify_feasibility()}")


    central_net = copy.deepcopy(currNetwork)
    central_net.allocate_power_central()
    central.append(central_net.central_powers)
    central_utilities.append(central_net.get_social_utility())
    print(f"Central Feasibility: {central_net.verify_feasibility()}")

    power_dif = []
    for ind1, bs in enumerate(central[ind]):
        print(f" central: {central[ind][ind1].value} vs distr: {distributed[ind][ind1]}")
        power_dif.append(np.linalg.norm(central[ind][ind1].value - distributed[ind][ind1]))
    difference.append(power_dif)
    duals = np.asarray(duals)

    dual_plot = plt.figure()
    dual_plt = dual_plot.add_subplot(1, 1, 1)
    fig_main = plt.figure()
    util_plt = fig_main.add_subplot(1, 3, 1)
    # util_plt.set_title("Power Convergence")
    util_plt.set_ylabel("Social Utility (System Capacity)")
    util_plt.set_xlabel("Iteration")
    extra_plt = fig_main.add_subplot(1, 3, 2)
    # extra_plt.set_title("Interference Constraint Slack")
    extra_plt.set_ylabel("Power Constraint Slack")
    extra_plt.set_xlabel("Iteration")
    extra_plt1 = fig_main.add_subplot(1, 3, 3)
    # extra_plt1.set_title("Power Constraint Slack")
    extra_plt1.set_ylabel("Interference Constraint Slack")
    extra_plt1.set_xlabel("Iteration")
    util_plt.plot(np.arange(num_iterations), utilities, label=f"FBS Antennas: {cur}")
    extra_plt.plot(np.arange(num_iterations), constraints[2], label=f"min.")
    extra_plt.plot(np.arange(num_iterations), constraints[3], label=f"max.")
    extra_plt1.plot(np.arange(num_iterations), constraints[0], label=f"min.")
    extra_plt1.plot(np.arange(num_iterations), constraints[1],  label=f"max.")
    # np.std(pow_dual), np.max(pow_dual), np.std(int_dual), np.max(int_dual)
    dual_plt.plot(duals[:, 1], label=f"power.")
    dual_plt.plot(duals[:, 3], label=f"interference.")
    dual_plt.legend(loc="lower left")
    check.append(np.asarray(utilities))

    util_plt.legend(loc="upper right")
    # extra_plt1.legend(loc="lower left")
    # extra_plt.legend(loc="lower left")

    plt.tight_layout()
    time_path = "Output/utility_"+f"{time.time()}" + f"{num_antenna}" + "curves.png"

print(f"central utilities {central_utilities[0]}")
print(f"dist utilities {distributed_utilities[0]}")
print(f"Power comparison {np.sum(difference)}")

# plt.show()

