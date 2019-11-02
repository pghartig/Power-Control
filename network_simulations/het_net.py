import numpy as np

class het_network():
    def __init__(self, num_femto_cells, num_macro_users):

        self.coverage_area = (100, 100)
        self.base_stations = []
        num_femto_users, num_antenna = 3, 3
        # these loops should probably be moved out of the constructor
        [self.base_stations.append(femto_cell(i, self, num_femto_users, num_antenna)) for i in range(num_femto_cells)]

        self.macro_users = []
        interference_threshold = 5
        [self.macro_users.append(macro_user(i, self, interference_threshold)) for i in range(num_macro_users)]
        self.update_macro_cells()


    def get_femto_cells(self):
        return self.base_stations

    def allocate_power(self):
        return None

    def update_macro_cells(self):
        for cell in self.base_stations:
            cell.reconize_macro_user(self.macro_users)

    def move_femto_users(self):
        for cell in self.base_stations:
            cell.move_users()
        return None

    def move_macro_users(self):
        return None

class femto_cell():
    def __init__(self, ID, network, num_femto_users, num_antenna):
        self.ID = ID
        self.users = []
        self.number_antennas = num_antenna
        self.macro_users = []
        self.network = network
        self.location = self.setup_location()
        self.coverage_size = np.array((5, 5))
        self.connect_users(num_femto_users)
        self.move_femto_users()



    #TODO type this parameter as macro user
    def reconize_macro_user(self, users):
        for macro_user in users:
            self.macro_users.append(macro_user)
            macro_user.add_interferer(self)

    def connect_users(self, num_femto_users):
        for user in range(num_femto_users):
            new_user = femto_user(user, self)
            self.users.append(new_user)
        self.move_femto_users()

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

    def move_femto_users(self):
        for user in self.users:
            user.move()

    def setup_location(self):
        return \
            np.array((np.random.randint(0,self.network.coverage_area[0]), np.random.randint(0,self.network.coverage_area[1])))

    def allocate_femto_users_power(self):
        power = 1
        for user in self.users:
            user.update_power(power)

#TODO if further types of users are added, make a user class and inherit from this

class macro_user():
    def __init__(self, ID, network, interference_threshold):
        self.ID = ID
        self.network = network
        self.interferers = []
        self.interference = 0
        self.interference_threshold = interference_threshold
        self.uplink_channel = 0
        self.downlink_channel = 0
        self.location = None
        self.move()

    def get_channel_for_base_station(self, i):
        return self.uplink_channel[:, i], self.downlink_channel[:, i]

    def add_interferer(self, femto_interferer):
        self.interferers.append(femto_interferer)

    def move(self):
        self.location = \
            (np.random.randint(self.network.coverage_area[0]), np.random.randint(self.network.coverage_area[1]))
        self.uplink_channel = np.random.randn()
        self.downlink_channel = np.random.randn()


class femto_user():
    def __init__(self, ID, parent):
        self.parent = parent
        self.ID = ID
        self.power_from_base_station = 0
        self.SINR = 0
        self.uplink_channel = 0
        self.downlink_channel = 0
        self.location = 0

    def move(self):
        test = 0
        self.location = self.parent.location  - self.parent.coverage_size/2 + \
            [np.random.randint(0,self.parent.coverage_size[0]), np.random.randint(0,self.parent.coverage_size[1])]
        self.uplink_channel = np.random.randn()
        self.downlink_channel = np.random.randn()

    def update_power(self, power):
        self.power_from_base_station = power
