import cvxpy as cvx
import dccp
x = cvx.Variable(2)
y = cvx.Variable(2)
myprob = cvx.Problem(cvx.Maximize(cvx.norm(x-y, 2)), [0<=x, x<=1, 0<=y, y<=1])
print("problem is DCP:", myprob.is_dcp())   # false
print("problem is DCCP:", dccp.is_dccp(myprob))  # true
result = myprob.solve(method = 'dccp')
print("x =", x.value)
print("y =", y.value)
print("cost value =", result[0])