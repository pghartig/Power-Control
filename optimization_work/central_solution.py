from network_simulations import het_net
"""
first setup the network according using the het_net class then consolidate all of the information from the network to solve the central optimization problem
"""

def test_central_optimization():
    network = het_net.het_network(10, 10, 5, 10, 1)
    network_information = network.get_network_channels()
    network.print_layout()

    # Setup Solver

    # Setup the Objective Function

    # Setup Constaints


    # Ensure that some QOS of service requirement is met TBD
    assert False
