"""
Defines a solver using :mod:`scikits.odes.odeint`
   
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
                Diego Volpatto <dtvolpatto@gmail.com>
"""

from typing import Callable

# from scipy import integrate
import numpy as np
from scikits.odes.odeint import odeint as integrate
from scikits.odes import ode

from ..fields.base import FieldBase
from ..pdes.base import PDEBase
from .base import SolverBase


class SkOdesSolver(SolverBase):
    """class for solving partial differential equations using scipy

    This class is a thin wrapper around :func:`scikits.odes.odeint.odeint`. In
    particular, it supports all the methods implemented by this function.
    """

    name = "skodes"

    def __init__(self, pde: PDEBase, backend: str = "auto", method: str = "cvode", **kwargs):
        r"""initialize the explicit solver

        Args:
            pde (:class:`~pde.pdes.base.PDEBase`):
                The instance describing the pde that needs to be solved
            backend (str):
                Determines how the function is created. Accepted  values are
                'numpy` and 'numba'. Alternatively, 'auto' lets the code decide
                for the most optimal backend.
            method (str):
                The method to integrate the problem from `scikits.odes.ode`.
            **kwargs:
                All extra arguments are forwarded to
                :func:`scikits.odes.odeint.odeint`.
        """
        super().__init__(pde)
        self.backend = backend
        self.method = method
        self.solver_params = kwargs

    def make_stepper(self, state: FieldBase, dt: float = None) -> Callable:
        """return a stepper function

        Args:
            state (:class:`~pde.fields.FieldBase`):
                An example for the state from which the grid and other
                information can be extracted
            dt (float):
                Initial time step for the simulation. If `None`, the solver will
                choose a suitable initial value

        Returns:
            Function that can be called to advance the `state` from time
            `t_start` to time `t_end`. The function call signature is
            `(state: numpy.ndarray, t_start: float, t_end: float)`
        """
        shape = state.data.shape
        self.info["dt"] = dt
        self.info["steps"] = 0
        self.info["stochastic"] = False

        # obtain function for evaluating the right hand side
        rhs = self._make_pde_rhs(state, backend=self.backend, allow_stochastic=False)

        def rhs_helper(t, state_flat, yout):
            """ helper function to provide the correct call convention """
            yout = rhs(state_flat.reshape(shape), t).flat

        def stepper(state, t_start, t_end):
            """use scikits.odes.odeint.odeint to advance `state` from `t_start` to
            `t_end`"""
            if dt is not None:
                self.solver_params["first_step"] = min(t_end - t_start, dt)

            t_points = np.array([t_start,  t_end])
            solver = ode(self.method, rhs_helper, old_api=False)
            solver.init_step(t_start, state.data.flat)
            sol = solver.step(t_end)
            # sol = solver.solve(t_points, state.data.flat)
            # sol = integrate(
            #     rhs_helper,
            #     tout=t_points,
            #     y0=state.data,
            #     method='bdf',
            #     **self.solver_params,
            # )
            self.info["steps"] += 1
            state.data.flat = sol.values.y
            return sol.values.t[0]

        if dt:
            self._logger.info(
                f"Initialized {self.__class__.__name__} stepper with dt=%g", dt
            )
        else:
            self._logger.info(f"Initialized {self.__class__.__name__} stepper")
        return stepper
