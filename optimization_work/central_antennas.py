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
num_users = 1
num_antenna = 1
userPowerList = [10]
num_antenna_list = [10, 15, 20]
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
for num_antenna in num_antenna_list:
    currNetwork = copy.deepcopy(network)
    currNetwork.add_antennas(num_antenna)
    utility, intereference, power = currNetwork.allocate_power_central()
    print(f"num antenna: {num_antenna}")
    print(utility)
    print(intereference)
    print(power)


