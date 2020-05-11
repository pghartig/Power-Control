from network_simulations.het_net import *
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import copy
import time

"""
first setup the network according using the het_net class then consolidate all of the information from the network to solve the central optimization problem
"""
pow_dual = 1
int_dual = pow_dual
pos_dual = pow_dual
num_users = 10
num_antenna = 15
userPowerList = [10, 20 , 50 , 100 ]
numMacroUsers = 20
numBaseStations = 5
interferenceThreshold = 1
userPower = userPowerList[0]
network = HetNet(numBaseStations, numMacroUsers, num_users, num_antenna, interferenceThreshold, int_dual, pow_dual, pos_dual,
                               userPower,
                              power_vector_setup=True,
                              random=False)
# figsize = (5, 5)
dual_plot = plt.figure()
utility_plt = dual_plot.add_subplot(3, 1, 1)
intf = dual_plot.add_subplot(3, 1, 2)
pwr = dual_plot.add_subplot(3, 1, 3)

dual_check = []
beam_type = ["Moore-Penrose", "Min Correlation"]
noisePowers = 0
for beam_former_choice in range(len(beam_type)):
    interference_max = []
    interference_min = []
    power_max = []
    power_min = []
    utilities = []
    for powers in userPowerList:
        currNetwork = copy.deepcopy(network)
        currNetwork.change_power_limit(powers)        # Choose the type of beamformers to use at BaseStations
        if beam_former_choice == 1:
            currNetwork.update_beam_formers(optimize=True, imperfectCsiNoisePower=noisePowers)
        if beam_former_choice == 2:
            currNetwork.update_beam_formers(optimize=True, null=True, imperfectCsiNoisePower=noisePowers)
        if beam_former_choice == 3:
            currNetwork.update_beam_formers(csi=True, imperfectCsiNoisePower=noisePowers)
        utility, intereference, power = currNetwork.allocate_power_central()
        utilities.append(utility[0])
        power_max.append(np.max(power))
        power_min.append(np.min(power))
        interference_max.append(np.max(intereference))
        interference_min.append(np.min(intereference))


    utility_plt.plot(userPowerList, utilities, label=f"Social Utilit: {beam_type[beam_former_choice]}.")
    intf.plot(userPowerList, interference_max, label=f"Max: {beam_type[beam_former_choice]}")
    intf.plot(userPowerList, interference_min, label=f"Min: {beam_type[beam_former_choice]}")
    pwr.plot(userPowerList, power_max, label=f"Max: {beam_type[beam_former_choice]}")
    pwr.plot(userPowerList, power_min, label=f"Min: {beam_type[beam_former_choice]}")

intf.set_ylabel("Interference Constraint Slack")
intf.set_xlabel("Power")
pwr.set_ylabel("Power Constraint Slack")
pwr.set_xlabel("Power")
utility_plt.set_ylabel("Social Utility")
utility_plt.set_xlabel("Power")
utility_plt.legend(loc="upper left")
intf.legend(loc="upper left")
pwr.legend(loc="upper left")
time_path = "Output/utility_" + f"{time.time()}" + "curves.png"
plt.savefig(time_path, format="png")
plt.show()
