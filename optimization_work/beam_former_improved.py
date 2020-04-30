from network_simulations.het_net import *
import matplotlib.pyplot as plt
import numpy as np
import copy
import time

"""
first setup the network according using the het_net class then consolidate all of the information from the network to 
solve the central optimization problem
"""

def test_beam_former():
    pow_dual = 1e-4
    int_dual = pow_dual
    pos_dual = pow_dual
    num_users = 5
    num_antenna = 15
    step_size_pow = 1e1
    step_size_int = step_size_pow
    num_iterations = 100
    numMacroUsers = 10
    numBaseStations = 5
    interferenceThreshold = 1
    userPower = 1000
    network = HetNet(numBaseStations, numMacroUsers, num_users, num_antenna, interferenceThreshold, int_dual, pow_dual, pos_dual,
                                   userPower,
                                  power_vector_setup=True,
                                  random=False)
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
    beam_type = ["Moore-Penrose", "Min Correlation"]
    noisePowers = 0
    for beam_former_choice in range(len(beam_type)):
        currNetwork = copy.deepcopy(network)
        # Choose the type of beamformers to use at BaseStations
        if beam_former_choice == 1:
            currNetwork.update_beam_formers(optimize=True, imperfectCsiNoisePower=noisePowers)
        if beam_former_choice == 2:
            currNetwork.update_beam_formers(optimize=True, null=True, imperfectCsiNoisePower=noisePowers)
        if beam_former_choice == 3:
            currNetwork.update_beam_formers(csi=True, imperfectCsiNoisePower=noisePowers)

        utilities, min_utilities, max_utilities, duals, feasibility, constraints = \
            currNetwork.allocate_power_step(num_iterations, step_size_pow, step_size_int)
        duals = np.asarray(duals)
        util_plt.plot(np.arange(num_iterations ), utilities, label=f"{beam_type[beam_former_choice]}")
        extra_plt.plot(np.arange(num_iterations), constraints[2], label=f"min: {beam_type[beam_former_choice]}")
        extra_plt.plot(np.arange(num_iterations), constraints[3], label=f"max: {beam_type[beam_former_choice]}")
        extra_plt1.plot(np.arange(num_iterations), constraints[0], '-', label=f"min: {beam_type[beam_former_choice]}")
        extra_plt1.plot(np.arange(num_iterations), constraints[1], '-', label=f"max: {beam_type[beam_former_choice]}")

        print(feasibility, "\n")
        check.append(np.asarray(utilities))

    plt.tight_layout()
    util_plt.legend(loc="lower left")
    extra_plt1.legend(loc="lower right")
    extra_plt.legend(loc="lower right")
    time_path = "Output/utility_"+f"{time.time()}"+"curves.png"
    plt.savefig(time_path, format="png")
    # network.print_layout()
    plt.show()
    pass

