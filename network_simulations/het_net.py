import numpy as np

class het_network():
    def __init__(self, num_femto_cells, num_macro_users):

        self.base_stations = []
        num_femto_users, num_antenna = 3, 3
        [self.base_stations.append(femto_cell(num_femto_users, num_antenna)) for i in range(num_femto_cells)]
        self.macro_users = []
        interference_threshold = 5
        [self.macro_users.append(macro_user(interference_threshold)) for i in range(num_macro_users)]

    def get_femto_cells(self):
        return self.base_stations

    def allocate_power(self):
        return None

class femto_cell():
    def __init__(self, num_femto_users, num_antenna):
        self.users = []
        self.connect_users(num_femto_users)
        self.number_antennas = num_antenna
        self.macro_users = []

    #TODO type this parameter as macro user
    def reconize_macro_user(self, user):
        self.macro_users.append(user)



    def connect_users(self, num_femto_users):
        for user in range(num_femto_users):
            new_user = femto_user()
            self.users.append(new_user)

    def get_user_channel_matrices(self):
        uplink = []
        downlink = []
        for i in self.users:
            uplink.append(i.uplink_channel)
            downlink.append(i.downlink_channel)
        uplink = np.asarray(uplink)
        downlink = np.asarray(downlink)
        return uplink, downlink

    def get_macro_channel_matrix(self):
        uplink = []
        downlink = []
        for i in self.macro_users:
            uplink.append(i.uplink_channel)
            downlink.append(i.downlink_channel)
        uplink = np.asarray(uplink)
        downlink = np.asarray(downlink)
        return uplink, downlink

class macro_user():
    def __init__(self, interference_threshold):
        self.interference = 0
        self.interference_threshold = interference_threshold
        self.uplink_channel = 0
        self.downlink_channel = 0

    def get_channel_for_base_station(self,i):
        return self.uplink_channel[:, i], self.downlink_channel[:, i]


class femto_user():
    def __init__(self):
        self.SINR = 0
        self.uplink_channel = 0
        self.downlink_channel = 0

