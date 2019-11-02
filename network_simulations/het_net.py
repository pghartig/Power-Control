import networkx as nx
import numpy as np

class het_network():
    def __init__(self, num_femto_cells, num_femto_users, num_macro_users):
        self.base_stations = []
        self.macro_users
        self.network = nx.DiGraph()

        # setup macro cell users
        self.network.add_nodes_from(np.arange(num_macro_users), type='macro_user')

        # setup femto base stations and users
        self.add_femto_cell_and_users(num_femto_cells, num_femto_users)

        # Connect femto base stations to macro users
        self.connect_macro_users()



    def connect_macro_users(self):
        test = list(self.network.nodes['type'])
        for node1 in list(self.network.nodes['type']):
            if node1.type == 'base_station':
                for node2 in list(self.network.nodes):
                    if node2.type == 'macro_user':
                        network.add_edge(node1, node2, gain = 1)

    def add_femto_cell_and_users(self, num_femto_cells, num_femto_users):
        for i in range(num_femto_cells):
            femto_cell = nx.DiGraph(name=i, type='base_station')
            femto_cell.add_node(i, type='base_station')
            for b in range(num_femto_users):
                self.add_femto_user(femto_cell, i, b)

            # self.network.add_node(femto_cell, femto_ID = i)
            self.network = nx.union(self.network, femto_cell,rename=('N-', 'C-'))   # add in this way to allow for intercell interference

    def add_femto_user(self, base_node: nx.DiGraph,ind,user_num):
        base_node.add_node(user_num, type='femto_user', ID = (ind, user_num))
        base_node.add_edge(0, user_num+1, type='channel', gain = 1)
        base_node.add_edge(user_num+1, 0, type='channel', gain = 1)

    def get_femto_cells(self):
        return None

    def allocate_power(self):
        return None


class femto_cell():
    def __init__(self, num_femto_users):
        self.users = []

class macro_user():

class femto_user():
