from network_simulations.het_net import *
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import copy
import time
import seaborn

"""
first setup the network according using the het_net class then consolidate all of the information from the network to solve the central optimization problem
"""
pow_dual = 1
int_dual = pow_dual
pos_dual = pow_dual
num_users = 1
num_antenna = 1
SNRs_dB = np.linspace(1, 30, 5)
userPowerList = np.power(10, SNRs_dB/10)
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
dual_check = []
interference_max = []
interference_min = []
power_max = []
power_min = []
utilities = []

dual_plot = plt.figure()
utility_plt = dual_plot.add_subplot(3, 1, 1)
intf = dual_plot.add_subplot(3, 1, 2)
pwr = dual_plot.add_subplot(3, 1, 3)

for powerLimit in userPowerList:
    currNetwork = copy.deepcopy(network)
    currNetwork.change_power_limit(powerLimit)
    utility, intereference, power = currNetwork.allocate_power_central()
    utilities.append(utility[0])
    power_max.append(np.max(power))
    power_min.append(np.min(power))
    interference_max.append(np.max(intereference))
    interference_min.append(np.min(intereference))


utility_plt.plot(SNRs_dB, utilities, label=f"Social Utility.")
intf.plot(SNRs_dB, interference_max, label=f"Max.")
intf.plot(SNRs_dB, interference_min, label=f"Min.")
pwr.plot(SNRs_dB, power_max, label=f"Max.")
pwr.plot(SNRs_dB, power_min, label=f"Min.")
intf.set_ylabel("Interference Constraint Slack")
# intf.set_xlabel("FCBS Power")
pwr.set_ylabel("Power Constraint Slack")
pwr.set_xlabel("FCBS Power (dB)")
utility_plt.set_ylabel("Social Utility")
# utility_plt.set_xlabel("FCBS Power")
utility_plt.legend(loc="lower right")
intf.legend(loc="upper right")
pwr.legend(loc="upper right")
seaborn.despine(ax=utility_plt, offset=0)
seaborn.despine(ax=intf, offset=0)
seaborn.despine(ax=pwr, offset=0)
plt.show()
