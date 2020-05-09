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
num_antenna = 18
userPowerList = [1000]
num_antenna_list = [20]
numMacroUsers = 20
numBaseStations = 5
interferenceThreshold = 1
userPower = userPowerList[0]
network = HetNet(numBaseStations, numMacroUsers, num_users, num_antenna, interferenceThreshold, int_dual, pow_dual, pos_dual,
                               userPower,
                              power_vector_setup=True,
                              random=False)
# figsize = (5, 5)
dual_check = []
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
    utility, intereference, power = currNetwork.allocate_power_central()
    print(f"beamformer: {beam_type[beam_former_choice]}")
    print(utility)
    print(intereference)
    print(power)


