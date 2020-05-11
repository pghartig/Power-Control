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
num_users = 5
num_antenna = 4
userPowerList = [100]
# num_antenna_list = [10, 15, 20, 25, 30, 35, 40 , 45, 50]
add_antenna_list = [1,5,5,5,5,5,5,5]

# num_antenna_list = [10, 15]
num_iterations = 200
numMacroUsers = 10
numBaseStations = 5
interferenceThreshold = 1
userPower = userPowerList[0]
network = HetNet(numBaseStations, numMacroUsers, num_users, num_antenna, interferenceThreshold, int_dual, pow_dual, pos_dual,
                               userPower,
                              power_vector_setup=True,
                              random=False)
# figsize = (5, 5)
interference_max = []
interference_min = []
power_max = []
power_min = []

utilities = []
dual_plot = plt.figure()
utility_plt = dual_plot.add_subplot(3, 1, 1)
intf = dual_plot.add_subplot(3, 1, 2)
pwr = dual_plot.add_subplot(3, 1, 3)
currNetwork = copy.deepcopy(network)
curr_num= num_antenna
num_antenna_list = []
for add_antenna in add_antenna_list:
    curr_num += add_antenna
    num_antenna_list.append(curr_num)
    currNetwork.add_antennas(add_antenna)
    utility, intereference, power = currNetwork.allocate_power_central()
    utilities.append(utility[0])
    power_max.append(np.max(power))
    power_min.append(np.min(power))
    interference_max.append(np.max(intereference))
    interference_min.append(np.min(intereference))
    print(f"num antenna: {curr_num}")
    print(utility)
    print(intereference)
    print(power)

utility_plt.plot(num_antenna_list, utilities, label=f"Social Utility.")
intf.plot(num_antenna_list, interference_max, label=f"Max.")
intf.plot(num_antenna_list, interference_min, label=f"min.")
pwr.plot(num_antenna_list, power_max, label=f"Max.")
pwr.plot(num_antenna_list, power_min, label=f"Min.")
intf.set_ylabel("Interference Constraint Slack")
intf.set_xlabel("Number Antenna")
pwr.set_ylabel("Power Constraint Slack")
pwr.set_xlabel("Number Antenna")
utility_plt.set_ylabel("Social Utility")
utility_plt.set_xlabel("Number Antenna")
utility_plt.legend(loc="upper right")
intf.legend(loc="upper right")
pwr.legend(loc="upper right")

plt.show()
