import cvxpy as cp
import numpy as np

def test_cvx():
    import cvxpy as cp
    import numpy as np

    # Generate a random SDP.
    n = 3
    p = 3
    np.random.seed(1)
    C = np.random.randn(n, n)
    A = []
    b = []
    for i in range(p):
        A.append(np.random.randn(n, n))
        b.append(np.random.randn())

    # Define and solve the CVXPY problem.
    # Create a symmetric matrix variable.
    X = cp.Variable((n, n), symmetric=True)
    # The operator >> denotes matrix inequality.
    constraints = [X >> 0]
    constraints += [
        cp.trace(A[i] @ X) == b[i] for i in range(p)
    ]
    prob = cp.Problem(cp.Minimize(cp.trace(C @ X)),
                      constraints)
    prob.solve()

    # Print result.
    print("The optimal value is", prob.value)
    print("A solution X is")
    print(X.value)