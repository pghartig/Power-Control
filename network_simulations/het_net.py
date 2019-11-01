import networkx as nx
import numpy as np

class network():
    def __init__(self, num_femto, num_femto_users, num_macro_users):
        graph = nx.DiGraph()
        nx.set_node_attributes(graph, 1, 'type')

        # setup femto base stations and users
        graph.add_nodes_from(np.arange())
        # setup macro cell users

    def allocate_power(self):
        return None

