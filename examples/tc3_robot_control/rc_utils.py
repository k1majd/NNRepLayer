"""_summary_

Returns:
    _type_: _description_
"""
import numpy as np
from sympy import symbols, Matrix, sin, cos, lambdify

# define a class for the car control problem
class CarControlProblem:
    """_summary_"""

    # pylint: disable=too-many-instance-attributes

    def __init__(self) -> None:
        self.time_sample = 0.5
        self.dyn_approx_const = 0.01
        self.goal = np.array([0.0, 0.0, 0.2])
        self.init_x_interval = [(-5, -3), (-5, -3), (0, np.pi / 2)]
        self.lyp = None
        self.lyp_diff = None
        self.lyp_lf = None
        self.lyp_lg = None
        self.dynamic = None

    def initialize_symb_states(self):
        """_summary_"""
        # Define state, control, and goal symbols
        x_r1, x_r2, x_r3 = symbols("xr1 xr2 xr3")
        x_g1, x_g2, r_g = symbols("xg1 xg2 rg")
        u_r1, u_r2 = symbols("u1,u2")

        # Define vectors of state, control, and goal
        x_r_vec = Matrix([x_r1, x_r2, x_r3])  # robot state [x,y,theta]
        x_g_vec = Matrix([x_g1, x_g2, r_g])  # goal [x,y,r]
        u_r_vec = Matrix([u_r1, u_r2])  # control [v,w]

        # Construct lyapunov symbolic function
        lyp = (
            (x_r_vec[0] - x_g_vec[0]) ** 2
            + (x_r_vec[1] - x_g_vec[1]) ** 2
            - (self.dyn_approx_const) ** 2
        )
        lyp_diff = lyp.diff(Matrix([x_r_vec]))
        self.lyp = lambdify([x_r_vec, x_g_vec], lyp)
        self.lyp_lf = lambdify([x_r_vec, x_g_vec], lyp_diff.T * Matrix([0.0, 0.0, 0.0]))
        self.lyp_lg = lambdify(
            [x_r_vec, x_g_vec],
            lyp_diff.T
            * Matrix(
                [
                    [cos(x_r_vec[2]), -self.dyn_approx_const * sin(x_r_vec[2])],
                    [sin(x_r_vec[2]), self.dyn_approx_const * cos(x_r_vec[2])],
                    [0.0, 1.0],
                ]
            ),
        )

        # define system dynamic
        self.dynamic = lambdify(
            [x_r_vec, u_r_vec],
            Matrix([[cos(x_r_vec[2]), 0.0], [sin(x_r_vec[2]), 0.0], [0.0, 1.0]])
            * u_r_vec,
            "numpy",
        )
