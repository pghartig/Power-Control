from network_simulations import het_net

def test_basic_network_setup():
    network = het_net.het_network(10, 10)
    network.allocate_power()

    assert False