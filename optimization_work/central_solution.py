from network_simulations import het_net
import cvxpy as cp
"""
first setup the network according using the het_net class then consolidate all of the information from the network to solve the central optimization problem
"""

def test_central_optimization():
    network = het_net.Het_Network(10, 10, 5, 10, 1)
    # network.print_layout()

    # Setup Problem Variables
    network_information = network.get_network_channels()
    macro_user_matrices = network.get_macro_matrices()
    beam_forming_matrices = network.get_beam_formers()
    macro_thresholds = network.get_macro_thresholds()
    power_constraints = network.get_power_constaints()


    # Setup Constraints
    constraints = []
    # Power Constraints
    for ind, matrix in enumerate(beam_forming_matrices):
        constraints.append(cp.trace(matrix.H@matrix) <= power_constraints[str(ind)])
    # Zero forcing Constraints
    for ind, matrix in enumerate(beam_forming_matrices):
        channel_matrix = network_information[str(ind)]["femto"]
        for ind1 in range(matrix.shape[1]):
            for ind2 in range(channel_matrix.shape[0]):
                if ind1 != ind2:
                    test1= matrix[:,ind1]
                    test2 = channel_matrix[ind2,:]
                    constraints.append(matrix[:,ind1].T@channel_matrix[ind2,:] == 0)

    # Interference Constraints
    for matrix_ind in macro_user_matrices:
        matrix = macro_user_matrices[matrix_ind]
        total = 0
        for user in matrix:
            vector = matrix[user]
            user_power = beam_forming_matrices[int(user)]
            total += vector.T@user_power@user_power.H@cp.conj(vector)
        constraints.append(total <= macro_thresholds[matrix_ind])


    # Setup the Objective Function
    objective = cp.Minimize(network.central_utility_function)

    # Setup Solver
    prob = cp.Problem(objective, constraints)
    #TODO ensure variables can be complex
    prob.solve()

    # Ensure that some QOS of service requirement is met TBD
    assert False
