import numpy as np

class het_network():
    def __init__(self, num_femto_cells, num_macro_users):

        self.coverage_area = (100, 100)
        self.base_stations = []
        num_femto_users, num_antenna = 1,1
        # these loops should probably be moved out of the constructor
        [self.base_stations.append(femto_base_station(i, self, num_femto_users, num_antenna)) for i in range(num_femto_cells)]

        self.macro_users = []
        interference_threshold = 5
        [self.macro_users.append(macro_user(i, self, interference_threshold)) for i in range(num_macro_users)]
        self.update_macro_cells()

    def get_femto_cells(self):
        return self.base_stations

    def allocate_power(self):
        [player.allocate_femto_users_power() for player in self.base_stations]

    def update_macro_cells(self):
        for cell in self.base_stations:
            cell.reconize_macro_user(self.macro_users)

    def move_femto_users(self):
        for cell in self.base_stations:
            cell.move_users()
        return None

    def move_macro_users(self):
        return None

class femto_base_station():
    def __init__(self, ID, network, num_femto_users, num_antenna,utility_function=np.sum):
        self.ID = ID
        self.users = []
        self.number_antennas = num_antenna
        self.macro_users = []
        self.network = network
        self.location = self.setup_location()
        self.coverage_size = np.array((5, 5))
        self.connect_users(num_femto_users)
        self.move_femto_users()
        self.utility_function = utility_function
        self.utility_evaluated = self.update_utility()


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
        self.update_utility()

    def update_utility(self):
        """

        :param utility_function: This should be a non-decreasing concave function
        :return:
        """
        self.utility_evaluated = self.utility_function(self.get_user_sinr())

    def get_user_sinr(self):
        all_sinr = []
        for user in self.users:
            all_sinr.append(user.get_sinr())
        return all_sinr

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
    def __init__(self, ID, parent, sigma_square=1):
        self.parent = parent
        self.ID = ID
        self.power_from_base_station = 0
        self.uplink_channel = 0
        self.downlink_channel = 0
        self.interference = 0
        self.noise_power = sigma_square
        self.location = 0
        self.SINR = (self.downlink_channel*self.power_from_base_station)/(self.noise_power+self.interference)

    def move(self):
        test = 0
        self.location = self.parent.location  - self.parent.coverage_size/2 + \
            [np.random.randint(0, self.parent.coverage_size[0]), np.random.randint(0,self.parent.coverage_size[1])]
        self.uplink_channel = np.random.randn()
        self.downlink_channel = np.random.randn()

    def update_power(self, power):
        self.power_from_base_station = power
        self.SINR = (self.downlink_channel*self.power_from_base_station)/(self.noise_power+self.interference)


    def get_sinr(self):
        return self.SINR
