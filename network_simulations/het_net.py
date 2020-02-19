import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import cvxpy as cp
import time
import math
class Het_Network():
    def __init__(self, num_femto_cells, num_macro_users, max_users,
                 max_antennas, interferenceThreshold, int_dual, pow_dual, pos_dual, power_limit, power_vector_setup=False, random=True):
        """
        TODO Enforce players have more antennas than users
        :param num_femto_cells:
        :param num_macro_users:
        :param max_users:
        :param max_antennas:
        :param interference_threshold:
        """
        self.coverage_area = (25, 25)
        self.base_stations = []
        # these loops should probably be moved out of the constructor
        if random ==False:
            [self.base_stations.append(Femto_Base_Station(i, self, max_users, max_antennas, power_vector_setup,
                                                          pow_dual, pos_dual, power_limit)) for i in range(num_femto_cells)]
        else:
            [self.base_stations.append(Femto_Base_Station(i, self, np.random.randint(1, max_users+1)
                                                          , np.random.randint(1, max_antennas+1),power_vector_setup, power_limit=power_limit))
         for i in range(num_femto_cells)]
        self.macro_users = []
        [self.macro_users.append(Macro_User(i, self, interferenceThreshold, int_dual)) for i in range(num_macro_users)]
        self.update_macro_cells()
        self.setup_base_stations()

        print("test")

    def get_network_channels(self):
        """

        :return: A list of the base stations and their downlink channels as two lists. The first as the channels to users
        and the second as channels to the macro users they interfere with.
        """
        ret = dict()
        for station in self.base_stations:
            station_l = dict()
            station_l["macro"] = station.get_macro_channel_matrices()
            station_l["femto"] = station.get_user_channel_matrices()
            ret[str(station.ID)] = station_l
        return ret

    def get_power_constraints(self):
        constraints = dict()
        for station in self.base_stations:
            constraints[str(station.ID)] = station.power_constraint
        return constraints

    def get_beam_formers(self):
        beam_formers = []
        for base_station in self.base_stations:
            beam_formers.append(base_station.beam_forming_matrix)
        return beam_formers

    def get_macro_matrices(self):
        ret = dict()
        for macro in self.macro_users:
            ret[str(macro.ID)] = macro.downlink_channels
        return ret

    def get_macro_thresholds(self):
        thresholds = dict()
        for macro in self.macro_users:
            thresholds[str(macro.ID)] = macro.interference_threshold
        return thresholds

    def get_femto_cells(self):
        return self.base_stations

    def allocate_power_step(self, num_iterations, step_size_pow, step_size_int):
        social_utility_vector = []
        average_duals = []
        feasibility = []
        interferenceSlackMin = []
        interferenceSlackMax = []
        powerSlackMin = []
        powerSlackMax = []
        social_utility_vector.append(self.get_social_utility())
        #   intitialize with correct duals given the starting allocation
        for i in range(num_iterations):
            average_duals.append(self.get_average_duals())
            #   First step in dual ascent -> find dual function
            [player.solve_local_opimization() for player in self.base_stations]
            #   Second step of dual ascent -> update dual variables based on values from first step
            self.__update_dual_variables(step_size_pow, step_size_int, i)
            social_utility_vector.append(self.get_social_utility())
            # feasibility.append(self.verify_feasibility())
            interferenceSlackMin.append(np.min(self.trackIntConstraints()))
            interferenceSlackMax.append(np.max(self.trackIntConstraints()))
            powerSlackMin.append(np.min(self.trackPowConstraints()))
            powerSlackMax.append(np.max(self.trackPowConstraints()))
            # step_size /= 2
        #End with resolving power to ensure feasibility
        average_duals.append(self.get_average_duals())
        [player.solve_local_opimization() for player in self.base_stations]
        return social_utility_vector, average_duals, self.verify_feasibility(), [interferenceSlackMin, interferenceSlackMax, powerSlackMin, powerSlackMax]

    def verify_feasibility(self):
        for bs in self.base_stations:
            if np.sum(bs.power_vector)-(.05*bs.power_constraint) >= bs.power_constraint or np.any(bs.power_vector < 0):
                return False
        for mcu in self.macro_users:
            if mcu.interference >= mcu.interference_threshold:
                return False
        return True

    def update_beam_formers(self, optimize=False, channel_set=False, csi=False, imperfectCsiNoisePower=0):
        for base_station in self.base_stations:
            base_station.update_beamformer(optimize=optimize, channel_set=channel_set, csi=csi, imperfectCsiNoisePower=imperfectCsiNoisePower)


    def __update_dual_variables(self, step_size_pow, step_size_int, itr_idx):
        """
        Update all dual variables in the distributed optimization problem
        :return:
        """
        # First update the dual variables of the macro users
        [player.update_dual_variables(step_size_pow, itr_idx) for player in self.base_stations]
        [macro_user.update_dual_variables(step_size_int, itr_idx) for macro_user in self.macro_users]

        # Second update the dual variables for the other constraints (Note this order doesn't matter)

    def update_macro_cells(self):
        """
        Have femto cell base stations find macro users
        :return:
        """
        for fc_base_station in self.base_stations:
            fc_base_station.reconize_macro_user(self.macro_users)

    def setup_base_stations(self):
        self.central_utility_function = 0
        for base_station in self.base_stations:
            base_station.setup_users()
            base_station.update_beamformer()

    def move_femto_users(self):
        for cell in self.base_stations:
            cell.move_users()
        return None

    def move_macro_users(self):
        return None

    def get_station_locations(self):
        locations = []
        for station in self.base_stations:
            locations.append(station.location)
        locations = np.asarray(locations)
        return locations

    def get_macro_locations(self):
        locations = []
        for user in self.macro_users:
            locations.append(user.location)
        locations = np.asarray(locations)
        return locations

    def get_social_utility(self):
        total = 0
        utilities = []
        for base_station in self.base_stations:
            utility = base_station.get_utility()
            total += utility
            utilities.append(utility)
        # return total
        return np.average(utilities)

    def get_base_stations(self):
        return self.base_stations

    def get_average_duals(self):
        """
        a function to be used for tracking convergence of the power allocation
        :return:
        """
        pow_dual = []
        pos_dual = []
        int_dual = []
        interference = []
        sum_power = []
        for bs in self.base_stations:
            pow_dual.append(bs.power_dual_variable)
            pos_dual.append(np.average(bs.positivity_dual_variable))
            sum_power.append(np.sum(bs.power_vector))
        for mcu in self.macro_users:
            int_dual.append(mcu.dual_variable)
            interference.append(mcu.interference)
        return np.average(pow_dual), np.average(pos_dual), np.max(int_dual), np.max(interference), np.average(sum_power)
        # return np.average(interference), np.average(ave_power)
        # return np.average(pow_dual), np.average(int_dual)

    def trackIntConstraints(self):
        interferenceSlack = []
        for mu in self.macro_users:
            interferenceSlack.append(mu.interference_threshold-mu.interference)
            # if mu.interference_threshold-mu.interference <0:
            #     print("check")
        return interferenceSlack


    def trackPowConstraints(self):
        powerSlack = []
        for bs in self.base_stations:
            powerSlack.append(bs.power_constraint - np.sum(bs.power_vector))
        return powerSlack

    def print_layout(self):
        # plt.figure()
        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(aspect="equal"))
        bs_locations = self.get_station_locations()
        for ind, station in enumerate(self.base_stations):
            fcu_locations = station.get_user_locations()
            plt.scatter(fcu_locations[:,0],fcu_locations[:,1], c=f"C{ind}", marker='^')
            plt.scatter(bs_locations[ind, 0], bs_locations[ind, 1],  c=f"C{ind}", marker='H')

        mu_locations = self.get_macro_locations()
        plt.scatter(mu_locations[:,0],mu_locations[:,1], marker='X')
        time_path = "Output/system" + f"{time.time()}" + "curves.png"
        blue_star = mlines.Line2D([], [], color='black', marker='^', linestyle='None',
                                 markersize=10, label="Femto User")
        red_square = mlines.Line2D([], [], color='black', marker='H', linestyle='None',
                                   markersize=10, label="Femto Base-Station")
        purple_triangle = mlines.Line2D([], [], color='black', marker='X', linestyle='None',
                                        markersize=10, label="Macro User")
        ax.legend(loc='lower left')
        fig.legend(handles=[blue_star, red_square, purple_triangle])
        # ax.legend(labels = ("Femto User", "Femto Base-Station", "Macro User"),handles= ('X', 'H', '^'))
        fig.savefig(time_path, format="png")
        # plt.savefig(time_path, format="png")

    def change_power_limit(self, new_limit):
        for bs in self.base_stations:
            bs.power_constraint = new_limit

    def change_number_antenna(self, num_antenna):
        for bs in self.base_stations:
            bs.change_num_antenna(num_antenna)

    def change_interference_constraint(self, interferenceThreshold):
        for mu in self.macro_users:
            mu.interference_threshold = interferenceThreshold

    def add_macro_users(self, numberNewUsers, interferenceThreshold, int_dual):
        if numberNewUsers >0 :
            [self.macro_users.append(Macro_User(i, self, interferenceThreshold, int_dual)) for i in range(numberNewUsers)]
            self.update_macro_cells()
            self.setup_base_stations()

    def addFemtoBaseStation(self, num_femto_cells, max_users, max_antennas, powerVectorSetup = True, powerLimit=1,  random=True):
        if num_femto_cells >0 :
            if random ==False:
                [self.base_stations.append(Femto_Base_Station(i, self, max_users, max_antennas, powerVectorSetup, power_limit=powerLimit)) for i in range(num_femto_cells)]
            else:
                [self.base_stations.append(Femto_Base_Station(i, self, np.random.randint(1, max_users+1)
                                                              , np.random.randint(1, max_antennas+1),powerVectorSetup, power_limit=powerLimit))
             for i in range(num_femto_cells)]
            #   reinitialize all channel information
            self.update_macro_cells()
            self.setup_base_stations()


class Femto_Base_Station():
    def __init__(self, ID, network, num_femto_users, num_antenna, power_vector_setup,
                 pow_dual, pos_dual, power_limit, utility_function=np.sum):
        self.ID = ID
        self.users = []
        # Ensure there are always more antennas than users
        self.number_antennas = num_antenna
        self.macro_users = []
        self.network = network
        self.location = self.setup_location()
        self.coverage_size = np.array((5, 5))
        if power_vector_setup == False:
            self.beam_forming_matrix = cp.Variable((self.number_antennas, num_femto_users), complex=True)
        else:
            self.beam_forming_matrix = np.zeros((self.number_antennas, num_femto_users))
            self.power_vector = np.ones((num_femto_users))*(power_limit/(num_femto_users))
            # self.power_vector = np.zeros((num_femto_users))*(power_limit/(num_femto_users))

        self.sigma_square = 1e-3
        self.connect_users(num_femto_users, pos_dual)
        self.power_constraint = power_limit
        self.utility_function = utility_function
        self.H = None
        self.H_tilde = None
        self.power_vector_setup = power_vector_setup
        self.power_dual_variable = pow_dual

    #TODO type this parameter as macro user
    def setup_utility(self):
        self.utility_function = self.utility_function(self.get_user_sinr())
        pass

    def update_beamformer(self, optimize=False, channel_set=False, csi=False,  imperfectCsiNoisePower = 0 ):
        """
        For the power vector setup this function is used to update the zero-forcing pre-coder
        to the current down-link channel
        :return:
        """
        #   find zero-forcing matrix (should be a tall matrix in general)
        if optimize == True:
            self.beam_forming_matrix = self.optimize_beam_former(channel_set, imperfectCsiNoisePower)
            # check = self.H@self.beam_forming_matrix
            # pass
        if csi == True:
            self.beam_forming_matrix = self.optimize_beam_former_for_csi(imperfectCsiNoisePower)
        else:
            self.beam_forming_matrix = np.linalg.pinv(self.H + np.random.standard_normal(self.H.shape)*np.sqrt(imperfectCsiNoisePower))
            if np.any(np.isnan(self.beam_forming_matrix)):
                raise Exception("problem with inversion")

        #   normalize columns of the matrix
        for column in range(self.beam_forming_matrix.shape[1]):
            self.beam_forming_matrix[:, column] /= np.linalg.norm(self.beam_forming_matrix[:,column])

    def optimize_beam_former(self, set=False, imperfectCsiNoisePower=0):
        # Setup variables (beamformer)
        x = cp.Variable(self.beam_forming_matrix.shape)
        if set == True:
            # choose set that can be nulled given the remaining degrees of freedom.
            dof = self.number_antennas - len(self.users)
            correlations = np.linalg.norm(self.H_tilde@self.beam_forming_matrix, axis=1)
            check = np.argsort(correlations)
            macro_user_matrix = np.zeros((dof, self.number_antennas))
            for i in range(dof):
                arg = np.argmax(correlations)
                correlations[arg] = 0
                macro_user_matrix[i, :] = self.H_tilde[arg,:]
        else:
            macro_user_matrix = self.H_tilde
        constr = [(self.H+np.random.standard_normal(self.H.shape)*np.sqrt(imperfectCsiNoisePower))@x == np.eye(self.H.shape[0])]
        # Setup problem and solve
        utility = []
        utility += [cp.sum_squares(macro_user_matrix[m,:]@x) for m in range(macro_user_matrix.shape[0])]
        # test adding regularization term to increase user correlation
        # utility += [cp.norm2(self.H[i,:]@x) for i in range(self.H.shape[0])]
        prob = cp.Problem(cp.Minimize(cp.sum(utility)), constr)
        # prob = cp.Problem(cp.Minimize(cp.trace(cp.matmul(self.H_tilde@x, x.T@self.H_tilde.T))), constr)
        prob.solve()
        # Return optimial beamforming matrix
        beam_former = x.value
        return beam_former

    def optimize_beam_former_for_csi(self, imperfectCsiNoisePower=0):
        """
        Use the zero-forcing beamformer which minimizes the total leaked energy in the case of imperfect CSI.
        :param set:
        :return:
        """
        # Setup variables (beamformer)
        x = cp.Variable(self.beam_forming_matrix.shape)
        macro_user_matrix = self.H_tilde
        # Setup constraints (Zero-Forcing Constraint)
        # constr = [cp.trace(cp.matmul(self.H@x, x.T@self.H.T)) == 0]
        constr = [(self.H+np.random.standard_normal(self.H.shape)*np.sqrt(imperfectCsiNoisePower))@x == np.eye(self.H.shape[0])]
        # Setup problem and solve
        utility = []
        # utility += [cp.sum_squares(macro_user_matrix[m,:]@x) for m in range(macro_user_matrix.shape[0])]
        utility += [cp.norm2(x)]
        # test adding regularization term to increase user correlation
        # utility += [cp.norm2(self.H[i,:]@x) for i in range(self.H.shape[0])]
        prob = cp.Problem(cp.Minimize(cp.sum(utility)), constr)
        # prob = cp.Problem(cp.Minimize(cp.trace(cp.matmul(self.H_tilde@x, x.T@self.H_tilde.T))), constr)
        prob.solve()
        # Return optimial beamforming matrix
        beam_former = x.value
        return beam_former

    def reconize_macro_user(self, users):
        for macro_user in users:
            self.macro_users.append(macro_user)
            macro_user.add_interferer(self)

    def connect_users(self, num_femto_users, pos_dual):
        for ind in range(num_femto_users):
            new_user = Femto_User(ind, self.network, self, sigma_square=self.sigma_square)
            self.users.append(new_user)
        self.positivity_dual_variable = np.ones((len(self.users)))*pos_dual

    def setup_users(self):
        for user in self.users:
            user.setup_channels()
        self.get_user_channel_matrices()
        self.get_macro_channel_matrices()

    def get_user_channel_matrices(self):
        downlink = []
        for m_user in self.users:
            downlink.append(m_user.get_channel_for_base_station(self.ID))
        downlink = np.asarray(downlink)
        # H should be a fat matrix to which we will find the right psuedo inverse to
        self.H = downlink
        return downlink

    def get_macro_channel_matrices(self):
        downlink = []
        for m_user in self.macro_users:
            downlink.append(m_user.get_channel_for_base_station(self.ID))
        downlink = np.asarray(downlink)
        self.H_tilde = downlink
        return downlink

    def getID(self):
        return self.ID

    def get_location(self):
        return self.location

    def get_user_info(self, ID):
        return self.power_vector[ID], self.beam_forming_matrix[:, ID]

    def get_beamformer(self):
        return self.beam_forming_matrix

    def move_femto_users(self):
        for user in self.users:
            user.move()

    def setup_location(self):
        return np.array((np.random.randint(0,self.network.coverage_area[0]), np.random.randint(0,self.network.coverage_area[1])))

    def get_utility(self):
        utility = 0
        for user in self.users:
            utility += np.log(1+user.get_sinr())
        return utility

    def solve_local_opimization(self):
        for ind, element in enumerate(self.power_vector):
            c = 0
            user_i_channel = self.users[ind].get_channel_for_base_station(self.ID)
            for m_user in self.macro_users:
                macro_user_channel = m_user.get_channel_for_base_station(self.ID)
                c += m_user.get_dual_variable()*pow(np.linalg.norm(user_i_channel*macro_user_channel), 2)
            c += self.power_dual_variable
            c -= self.positivity_dual_variable[ind]
            #   prohibit negative powers
            updated_power = np.max((1/(c) - self.sigma_square, 0))
            # updated_power = 1/c - self.sigma_square
            if math.isnan(updated_power):
                raise Exception("problem with inversion")
            self.power_vector[ind] = updated_power

    def update_dual_variables(self, step, idx):
        #update scalar here
        self.power_dual_variable += step*(np.sum(self.power_vector) - self.power_constraint)
        self.power_dual_variable = np.max((0,self.power_dual_variable))
        if math.isnan(self.power_dual_variable):
            raise Exception("problem with inversion")

        #Update whole vector
        # self.positivity_dual_variable += pow(step,idx)*(-self.power_vector)
        self.positivity_dual_variable += step*(-self.power_vector)
        self.positivity_dual_variable = np.max((np.zeros(self.positivity_dual_variable.size), self.positivity_dual_variable), axis=0)
        if np.any(np.isnan(self.positivity_dual_variable)):
            raise Exception("problem with inversion")

    def get_user_locations(self):
        locations = []
        for user in self.users:
            locations.append(user.location)
        return np.asarray(locations)

    def change_num_antenna(self, num_antenna, optimize=False):
        if num_antenna >= len(self.users):
            self.number_antennas = num_antenna
            self.setup_users()
            self.reconize_macro_user()
            self.update_beamformer(optimize=optimize)

        else:
            raise Exception("cannot have more users than antenna")


class User:
    def __init__(self, ID, network):
        self.ID = ID
        self.uplink_channels = dict()
        self.downlink_channels = dict()
        self.location = (0, 0)
        self.network = network

    def get_channel_for_base_station(self, base_station_index):
        return self.downlink_channels.get(str(base_station_index))

    def add_base_station_antenna(self, num_antennas):
        for ind, item in enumerate(self.uplink_channels):
            self.downlink_channels[str(ind)] = 1

    def move(self):
        self.location = (np.random.randint(self.network.coverage_area[0]), np.random.randint(self.network.coverage_area[1]))
        #TODO update channel on move


class Macro_User(User):
    def __init__(self, ID, network,interference_threshold, dual):
        User.__init__(self, ID, network)
        self.interference = 0
        self.interference_threshold = interference_threshold
        self.dual_variable = dual
        self.move()


    def add_interferer(self, interferer :Femto_Base_Station):
        distance_to_base_station = interferer.get_location() - self.location + interferer.coverage_size*.001
        sqrt_gain = 1 / np.linalg.norm(distance_to_base_station)
        channel = np.random.randn(interferer.number_antennas) * sqrt_gain
        if np.any(channel == math.inf) or np.any(channel == - math.inf):
            raise Exception("infinite channel???")
        self.downlink_channels[str(interferer.ID)] = channel*sqrt_gain

    def update_dual_variables(self, step, idx):
        """
        Note that normally the values used here would depend on estimated SNR received at the macro user.
        :param step:
        :return:
        """
        total = 0
        for base_station in self.network.get_base_stations():
            for ind, power in enumerate(base_station.power_vector):
                channel = self.get_channel_for_base_station(base_station.ID)
                beamformer = base_station.beam_forming_matrix[:, ind]
                total += power*pow(np.linalg.norm(channel@beamformer),2)
                if math.isnan(power*pow(np.linalg.norm(channel@beamformer),2)):
                    raise Exception("problem with inversion")
        # self.dual_variable += pow(step,idx)*(total - self.interference_threshold)
        self.dual_variable += step*(total - self.interference_threshold)
        # if self.dual_variable < 0:
        #     print("check")
        self.dual_variable = np.max((0, self.dual_variable))
        if math.isnan(self.dual_variable):
            raise Exception("problem with inversion")
        self.interference = total

    def get_dual_variable(self):
        return self.dual_variable


class Femto_User(User):
    def __init__(self, ID, network, parent, sigma_square=1e-3):
        """
        For now assume that the femto users are only single antenna
        :param ID:
        :param parent:
        :param sigma_square:
        """
        User.__init__(self, ID, network)
        self.parent = parent
        self.power_from_base_station = 0
        self.interference = 0
        self.noise_power = sigma_square
        self.move()


    def setup_channels(self):
        for base_station in self.network.base_stations:
            distance_to_base_station = base_station.get_location()-self.location + base_station.coverage_size*.001
            sqrt_gain = 1/np.linalg.norm(distance_to_base_station)
            channel = np.random.randn(base_station.number_antennas)*sqrt_gain
            if np.any(channel == math.inf) or np.any(channel == - math.inf):
                raise Exception("infinite channel???")
            self.downlink_channels[str(base_station.ID)] = channel*sqrt_gain

    def update_power(self, power):
        self.power_from_base_station = power
        self.SINR = self.get_sinr()

    def get_sinr(self):
        power, beamformer = self.parent.get_user_info(self.ID)
        channel = self.downlink_channels[str(self.parent.getID())]
        # add interference to analyze for the case of imperfect CSI
        # self.SINR = power*pow(np.linalg.norm(channel@beamformer), 2)/(self.noise_power+self.interference)
        parent_beamformer = self.parent.get_beamformer()
        self.interference = 0
        for j in range(parent_beamformer.shape[1]):
            if j is not self.ID:
                self.interference += pow(np.linalg.norm(channel @ parent_beamformer[:,j]), 2)
        self.SINR = power*pow(np.linalg.norm(channel@beamformer), 2)/(self.noise_power+self.interference)
        return self.SINR

    def move(self):
        """
        Users can only move in the range of their current base-station (not allowing for users to move between at the moment)
        :return:
        """
        self.location = self.parent.get_location() + (np.random.randint(-self.parent.coverage_size[0], self.parent.coverage_size[0]),
                                                      np.random.randint(-self.parent.coverage_size[1], self.parent.coverage_size[1]))
        pass
