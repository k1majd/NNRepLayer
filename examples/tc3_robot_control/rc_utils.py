"""_summary_

Returns:
    _type_: _description_
"""
import math
import numpy as np
from sympy import symbols, Matrix, sin, cos, lambdify

# define a class for the car control problem
class CarControlProblem:
    """_summary_

    Returns:
        _type_: _description_
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self) -> None:
        self.time_sample = 0.5
        self.dyn_approx_const = 0.01
        self.goal = np.array([0.0, 0.0, 0.2])
        self.init_state_set = [(-5, -3), (-5, -3), (0, np.pi / 2)]
        self.lyp = None
        self.lyp_lf = None
        self.lyp_lg = None
        self.dynamic = None
        self.controller_nn = None

    def generate_sample(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        state = np.zeros((3))
        state[0] = self.sample_interval(
            self.init_state_set[0][0], self.init_state_set[0][1], 300
        )
        state[1] = self.sample_interval(
            self.init_state_set[1][0], self.init_state_set[1][1], 300
        )
        state[2] = self.sample_interval(
            self.init_state_set[2][0], self.init_state_set[2][1], 6
        )
        return state, self.format_state2nn(state)

    def initialize_sym_states(self):
        """_summary_"""
        # Define state, control, and goal symbols
        s_r1, s_r2, s_r3 = symbols("sr1 sr2 sr3")
        s_g1, s_g2, r_g = symbols("sg1 sg2 rg")
        a_r1, a_r2 = symbols("a1,a2")

        # Define vectors of state, control, and goal
        state_vec = Matrix([s_r1, s_r2, s_r3])  # robot state [x,y,theta]
        goal_vec = Matrix([s_g1, s_g2, r_g])  # goal [x,y,r]
        action_vec = Matrix([a_r1, a_r2])  # action [v,w]

        # Construct lyapunov symbolic function
        lyp = (
            (state_vec[0] - goal_vec[0]) ** 2
            + (state_vec[1] - goal_vec[1]) ** 2
            - (self.dyn_approx_const) ** 2
        )
        lyp_diff = lyp.diff(Matrix([state_vec]))
        self.lyp = lambdify([state_vec, goal_vec], lyp)
        self.lyp_lf = lambdify(
            [state_vec, goal_vec], lyp_diff.T * Matrix([0.0, 0.0, 0.0])
        )
        self.lyp_lg = lambdify(
            [state_vec, goal_vec],
            lyp_diff.T
            * Matrix(
                [
                    [cos(state_vec[2]), -self.dyn_approx_const * sin(state_vec[2])],
                    [sin(state_vec[2]), self.dyn_approx_const * cos(state_vec[2])],
                    [0.0, 1.0],
                ]
            ),
        )

        # define system dynamic
        self.dynamic = lambdify(
            [state_vec, action_vec],
            Matrix([[cos(state_vec[2]), 0.0], [sin(state_vec[2]), 0.0], [0.0, 1.0]])
            * action_vec,
            "numpy",
        )

    def train_nn_controller(self, num_traj):
        traj_set = []
        traj_nn_set = []
        action_set = []

        # We use Dagger algorithm for training, ref: https://arxiv.org/pdf/2001.08088.pdf
        for i in range(num_traj):
            beta = (num_traj - i) / (num_traj)
            print("Cycle {}".format(i + 1))

            # generate a sample initial state in the initial set
            state, state_nn = self.generate_sample()

            state_traj = [state]
            state_traj_nn = [state_nn]
            action_traj = []

            while np.linalg.norm(state[0:2] - self.goal[0:2]) > self.goal[2] + 0.05:
                action_opt = self.give_control_opt(state)
                action_nn = self.give_control_nn(state_nn)
                state, state_nn = self.give_next_state(
                    beta * action_opt + (1 - beta) * action_nn, state
                )

                state_traj.append(state)
                state_traj_nn.append(state_nn)
                action_traj.append(action_opt)
            action_traj.append(action_opt)
            print("Trajectory {} is generated. Training time!!".format(i + 1))

            # convert the trajectories into np array
            state_traj = np.array(state_traj)
            state_traj_nn = np.array(state_traj_nn)
            action_traj = self.normalize_action(np.array(action_traj))

            # append the new trajectory into the list
            traj_set.append(state_traj)
            traj_nn_set.append(state_traj_nn)
            action_set.append(action_traj)

            # Train model for the generate sample
            his = self.controller_nn.fit(
                np.concatenate(traj_nn_set),
                np.concatenate(action_set),
                epochs=2,
                use_multiprocessing=True,
                verbose=0,
            )

            print("----------------------------------------")

    def give_control_opt(self, state):
        pass

    def give_control_nn(self, state_nn):
        pass

    def give_next_state(self, action, state):
        state = (self.time_sample * self.dynamic(state, action)).flatten() + state
        state[2] = self.pi_pi_adj(state[2])
        return state, self.format_state2nn(state)

    def normalize_action(self, action):
        return np.concatenate(
            (action, np.array([np.linspace(0.0, 1.0, num=action.shape[0])]).T), axis=1
        )

    @staticmethod
    def sample_interval(min_x, max_x, res=1):
        return (((max_x) - (min_x)) / res) * np.random.randint(0, res) + (min_x)

    @staticmethod
    def format_state2nn(state):
        state_nn = np.zeros((4))
        state_nn[0] = state[0]
        state_nn[1] = state[1]
        state_nn[2] = np.sin(state[2])
        state_nn[3] = np.cos(state[2])

        return state_nn

    @staticmethod
    def pi_pi_adj(theta):
        if theta > np.pi:
            temp = theta / (2 * np.pi) - 0.5
            k = np.ceil(temp)
            theta = theta - k * 2 * np.pi
        elif theta < -np.pi:
            temp = -theta / (2 * np.pi) - 0.5
            k = np.ceil(temp)
            theta = theta + k * 2 * np.pi
        return theta
