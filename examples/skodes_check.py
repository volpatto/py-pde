import numpy as np
from numpy import cos, sin, sqrt
from scikits.odes.odeint import odeint
from scikits.odes import ode

tout = np.linspace(0, 1)
initial_values = np.array([1])


def right_hand_side(t, y, ydot):
    ydot[0] = y[0]


# output = odeint(right_hand_side, tout, initial_values)
# print(output.values.y)

k = 4.0
m = 1.0
initx = [1, 0.1]


def rhseqn(t, x, xdot):
    # we create rhs equations for the problem
    xdot[0] = x[1]
    xdot[1] = - k / m * x[0]


solver = ode('cvode', rhseqn, old_api=False)
result = solver.solve([0., 1., 2.], initx)
print('   t        Solution          Exact')
print('------------------------------------')
for t, u in zip(result.values.t, result.values.y):
    print('%4.2f %15.6g %15.6g' % (
        t, u[0], initx[0] * cos(sqrt(k / m) * t) + initx[1] * sin(sqrt(k / m) * t) / sqrt(k / m)))
