"""
Solver comparison
=================

This example shows how to set up solvers explicitly and how to extract
diagnostic information.
"""

import pde
import os

# build_sundials = os.environ['SUNDIALS_INST']

# initialize the grid, an initial condition, and the PDE
grid = pde.UnitGrid([32, 32])
field = pde.ScalarField.random_uniform(grid, -1, 1)
eq = pde.DiffusionPDE()

# try the explicit solver
solver1 = pde.ExplicitSolver(eq)
controller1 = pde.Controller(solver1, t_range=1, tracker=None)
sol1 = controller1.run(field, dt=1e-3)
sol1.label = "explicit solver"
print("Diagnostic information from first run:")
print(controller1.diagnostics)
print()

# try the standard scipy solver
solver2 = pde.ScipySolver(eq)
controller2 = pde.Controller(solver2, t_range=1, tracker=None)
sol2 = controller2.run(field)
sol2.label = "scipy solver"
print("Diagnostic information from second run:")
print(controller2.diagnostics)
print()

# try the standard scikits.odes solver
solver3 = pde.SkOdesSolver(eq, method='dopri5')
controller3 = pde.Controller(solver3, t_range=1, tracker=None)
sol3 = controller3.run(field)
sol3.label = "scikits.odes solver"
print("Diagnostic information from third run:")
print(controller3.diagnostics)
print()

# plot both fields and give the deviation as the title
title = f"Deviation of explicit with scipy: {((sol1 - sol2)**2).average:.2g}"
pde.FieldCollection([sol1, sol2]).plot(title=title)

# # plot both fields and give the deviation as the title
# title = f"Deviation of explicit with scikits.odes: {((sol1 - sol3)**2).average:.2g}"
# pde.FieldCollection([sol1, sol3]).plot(title=title)
