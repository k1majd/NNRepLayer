"""_summary_

Returns:
    _type_: _description_
"""
import os
import math
import numpy as np
from sympy import symbols, Matrix, sin, cos, lambdify
import cvxopt
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches


# define a class for the car control problem
class CarControlProblem:
    """_summary_

    Returns:
        _type_: _description_
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, nn_controller_arch) -> None:
        self.time_sample = 0.5
        self.dyn_approx_const = 0.01
        self.goal = np.array([0.0, 0.0, 0.2])
        self.init_state_set = [(-5, -3), (-5, -3), (0, np.pi / 2)]
        self.controller_nn = nn_controller_arch
        self.lyp = None
        self.lyp_lf = None
        self.lyp_lg = None
        self.dynamic = None
        self.controller_nn_train_hist = []

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

    def apply_dagger_learn(
        self, num_traj=100, train_epochs=100, batch_size_train=50, train2test_ratio=0.7
    ):
        """_summary_

        Args:
            num_traj (int, optional): _description_. Defaults to 100.
            train_epochs (int, optional): _description_. Defaults to 100.
            batch_size_train (int, optional): _description_. Defaults to 50.
            train2test_ratio (float, optional): _description_. Defaults to 0.7.

        Returns:
            _type_: _description_
        """

        ## generate testing dataset
        test_sets = self.give_test_dataset(
            int(num_traj * ((1 - train2test_ratio) / train2test_ratio))
        )
        print("----------------------------------------")
        print("Start Training!")

        traj_set = []
        traj_nn_set = []
        action_set = []

        # We use Dagger algorithm for training, ref: https://arxiv.org/pdf/2001.08088.pdf
        for i in range(num_traj):
            beta = (num_traj - i) / (num_traj)
            print(f"Cycle {i + 1}")

            # generate a sample initial state in the initial set
            state, state_nn = self.generate_sample()

            state_traj = [state]
            state_traj_nn = [state_nn]
            action_traj = []

            # generate the trajectory starting from the initial state where
            # action = beta * opt_action + (1-beta) * nn_action
            counter = 0
            while np.linalg.norm(state[0:2] - self.goal[0:2]) >= self.goal[2] + 0.05:
                action_opt = self.give_control_opt(state)
                action_nn = self.give_control_nn(state_nn)
                if action_nn[0] < 0:
                    action_nn[0] = 0
                state, state_nn = self.give_next_state(
                    beta * action_opt + (1 - beta) * action_nn[0:2], state
                )
                counter += 1
                state_traj.append(state)
                state_traj_nn.append(state_nn)
                action_traj.append(action_opt)
            action_traj.append(action_opt)
            print(
                f"Trajectory {i + 1} is generated in {counter} iterations. Training time!!"
            )

            # convert the trajectories into np array
            state_traj = np.array(state_traj)
            state_traj_nn = np.array(state_traj_nn)
            action_traj = self.normalize_action(np.array(action_traj))

            # append the new trajectory into the list
            traj_set.append(state_traj)
            traj_nn_set.append(state_traj_nn)
            action_set.append(action_traj)

            self.train_controller(
                [traj_nn_set, action_set], test_sets, train_epochs, batch_size_train
            )

            print("----------------------------------------")

        return [traj_set, traj_nn_set, action_set], test_sets

    def train_controller(self, train_set, test_set, train_epochs, batch_size_train):
        """_summary_

        Args:
            train_set (list): _description_
            test_set (list): _description_
        """
        ## define training callbacks:
        callback_reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=5, min_lr=0.0001
        )  # reduce learning rate
        callback_es = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )  # early stopping callback

        # Train model for the generated sample
        self.controller_nn_train_hist.append(
            self.controller_nn.fit(
                np.concatenate(train_set[0]),
                np.concatenate(train_set[1]),
                validation_data=(
                    np.concatenate(test_set[1]),
                    np.concatenate(test_set[2]),
                ),
                batch_size=batch_size_train,
                epochs=train_epochs,
                use_multiprocessing=True,
                verbose=1,
                callbacks=[callback_es, callback_reduce_lr],
            )
        )
        print("Model Loss + Accuracy on Train Data Set: ")
        self.controller_nn.evaluate(
            np.concatenate(train_set[0]), np.concatenate(train_set[1]), verbose=2
        )  # model evaluation
        print("Model Loss + Accuracy on Test Data Set: ")
        self.controller_nn.evaluate(
            np.concatenate(test_set[1]), np.concatenate(test_set[2]), verbose=2
        )  # model evaluation

    def give_test_dataset(self, size=35):
        """this method generates testing dataset

        Args:
            size (int, optional): number of trajectories in the test dataset. Defaults to 35.

        Returns:
            list: returns testing dataset
        """
        traj_set = []
        traj_nn_set = []
        action_set = []

        for i in range(size):
            print(f"Generate testing traj {i + 1}")
            state, state_nn = self.generate_sample()

            trace = self.give_single_trajectory(state, state_nn, control="opt")

            # append the new trajectory into the list
            traj_set.append(trace[0])
            traj_nn_set.append(trace[1])
            action_set.append(trace[2])

        return [traj_set, traj_nn_set, action_set]

    def give_single_trajectory(
        self,
        state,
        state_nn=None,
        control="opt",
        err=0.05,
        disp_thresh=0.005,
        err_check_iter=15,
        max_err_updates=5,
    ):
        """_summary_

        Args:
            state (ndarray): 1D array containing data with 'float' type
            state_nn (ndarray, optional):
            1D array containing data with 'float' type. Defaults to None.
            control (str, optional): string either "opt" or "nn". Defaults to "opt".
            err (float, optional): adjust distance to goal radius err. Defaults to 0.001.
            disp_thresh (float, optional): robot displacement threshold. Defaults to 0.005.
            err_check_iter (int, optional):
            min value of data points after which err update is checked. Defaults to 15.
            max_err_updates (int, optional):
            number of err updates after which the trajectory is terminated. Defaults to 5.

        Raises:
            NameError: if control type is not "opt" or "nn"

        Returns:
            list[ndarray]: a list of 2D arrays [state_traj, state_nn_traj, action_traj]
        """

        if state_nn is None:
            state_nn = self.format_state2nn(state)
        state_traj = [state]
        state_traj_nn = [state_nn]
        action_traj = []
        stop_cond = False
        if control == "nn":
            err_update = 0

        while (not stop_cond) and (
            np.linalg.norm(state[0:2] - self.goal[0:2]) >= self.goal[2] + err
        ):
            if control == "opt":
                action = self.give_control_opt(state)
                state, state_nn = self.give_next_state(action, state)
            elif control == "nn":
                action = self.give_control_nn(state_nn)
                state, state_nn = self.give_next_state(action[0:2], state)
                if len(state_traj) > err_check_iter:
                    old_state_dist2goal, _ = self.calc_distance_n_angle(state_traj[-5])
                    curr_state_dist2goal, _ = self.calc_distance_n_angle(state)
                    if (
                        curr_state_dist2goal >= old_state_dist2goal
                        or abs(curr_state_dist2goal - old_state_dist2goal) < disp_thresh
                    ):
                        err += 0.05
                        err_update += 1
                if err_update > max_err_updates:
                    stop_cond = True

            else:
                raise NameError(f"Controller name -{control}- is not found!")

            state_traj.append(state)
            state_traj_nn.append(state_nn)
            action_traj.append(action)
        action_traj.append(action)

        # convert the trajectories into np array
        state_traj = np.array(state_traj)
        state_traj_nn = np.array(state_traj_nn)
        if control == "opt":
            action_traj = self.normalize_action(np.array(action_traj))
        else:
            action_traj = np.array(action_traj)

        return [state_traj, state_traj_nn, action_traj]

    def give_control_opt(self, state, v_gain=1.2, w_gain=1):
        """_summary_

        Args:
            state (_type_): _description_
            v_gain (float, optional): _description_. Defaults to 1.2.
            w_gain (int, optional): _description_. Defaults to 1.

        Returns:
            _type_: _description_
        """
        # robot spec
        v_max = 1.0
        v_min = 0.0
        w_max = 0.5
        w_min = -0.5

        # OPT formulate: min u' H u + ff * u s.t. Au<=b
        ineq_mat = np.zeros((6, 3))
        ineq_vec = np.zeros((6, 1))

        ## lyapanov
        ineq_mat[0, 0:2] = self.lyp_lg(state, self.goal)
        ineq_mat[0, 2] = -1
        ineq_vec[0] = -self.lyp_lf(state, self.goal)
        # slack variabl of lyapanov
        ineq_mat[1, 2] = -1

        ## Control constraints
        ineq_mat[2, 0] = 1.0
        ineq_vec[2] = v_max
        ineq_mat[3, 0] = -1.0
        ineq_vec[3] = -v_min
        ineq_mat[4, 1] = 1.0
        ineq_vec[4] = w_max
        ineq_mat[5, 1] = -1.0
        ineq_vec[5] = -w_min

        # opt parameters
        dist_goal, theta_goal = self.calc_distance_n_angle(state)
        weight_mat = np.zeros((3, 3))
        weight_vec = np.zeros((3, 1))
        weight_mat[0, 0] = 2
        weight_mat[1, 1] = 2
        weight_mat[2, 2] = 0
        weight_vec[0] = -2 * np.exp(-v_gain / dist_goal) * v_max  # ref velocity
        weight_vec[1] = (
            -2 * w_gain * self.give_angular_diff(state[2], theta_goal)
        )  # ref steering rate
        weight_vec[2] = 2
        action = self.solve_qp(weight_mat, weight_vec, ineq_mat, ineq_vec)

        return action[0:2]

    def give_control_nn(self, state_nn):
        """_summary_

        Args:
            state_nn (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.controller_nn.predict(np.array([state_nn])).flatten()  # NN output

    def give_next_state(self, action, state):
        """_summary_

        Args:
            action (_type_): _description_
            state (_type_): _description_

        Returns:
            _type_: _description_
        """
        state = (self.time_sample * self.dynamic(state, action)).flatten() + state
        state[2] = self.adjust_pi_pi(state[2])
        return state, self.format_state2nn(state)

    def calc_distance_n_angle(self, state):
        """_summary_

        Args:
            state (_type_): _description_
            goal (_type_): _description_

        Returns:
            _type_: _description_
        """
        diff = self.goal - state
        dist = math.hypot(diff[0], diff[1])
        theta = math.atan2(diff[1], diff[0])
        return dist, theta

    def plot_history(self, path=os.getcwd() + "/history.eps"):

        """_summary_"""

        path = path.split(".")

        plt.rcParams["text.usetex"] = False
        mpl.style.use("seaborn")
        results_train_loss = []
        results_valid_loss = []
        results_train_acc = []
        results_valid_acc = []
        for his in self.controller_nn_train_hist:
            results_train_loss.extend(his.history["loss"])
            results_valid_loss.extend(his.history["val_loss"])
            results_train_acc.extend(his.history["accuracy"])
            results_valid_acc.extend(his.history["val_accuracy"])

        ## loss plotting
        plt.plot(results_train_loss, color="red", label="training loss")
        plt.plot(results_valid_loss, color="blue", label="validation loss")
        plt.title("Loss Function Output")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend(loc="upper left", frameon=False)
        plt.savefig(path[0] + "_loss." + path[1], format="eps")
        print(f"loss plot is stored in {path[0]}_loss.{path[1]}")
        # plt.show()

        ## accuracy plotting
        plt.plot(results_train_acc, color="red", label="training accuracy")
        plt.plot(results_valid_acc, color="blue", label="validation accuracy")
        plt.title("Accuracy Function Output")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.legend(loc="upper left", frameon=False)
        plt.savefig(path[0] + "_acc." + path[1], format="eps")
        print(f"accuracy plot is stored in {path[0]}_accuracy.{path[1]}")
        # plt.show()

    def plot_trajectory_sets(
        self,
        trajectories,
        path=os.getcwd() + "/trajVSref.eps",
        title="trained vs ref trajectories",
    ):
        # pylint: disable = "invalid-name"
        """_summary_

        Args:
            trajectories (_type_): _description_
            title (str, optional): _description_. Defaults to "trained vs ref trajectories".
        """
        plt.rcParams["text.usetex"] = False
        mpl.style.use("seaborn")
        left, bottom, width, height = (
            self.init_state_set[0][0],
            self.init_state_set[1][0],
            2,
            2,
        )
        rect = mpatches.Rectangle(
            (left, bottom),
            width,
            height,
            fill=True,
            color="grey",
            linewidth=2,
            label="Initial Set of States",
        )
        for traj in trajectories[0]:
            plt.plot(
                np.array(traj)[:, 0],
                np.array(traj)[:, 1],
                linestyle="--",
                linewidth=2.5,
                color="darkgray",
            )
            plt.arrow(
                np.array(traj)[10, 0],
                np.array(traj)[10, 1],
                np.array(traj)[11, 0] - np.array(traj)[10, 0],
                np.array(traj)[11, 1] - np.array(traj)[10, 1],
                shape="full",
                color="darkgray",
                lw=0,
                length_includes_head=True,
                head_width=0.15,
            )

        for traj in trajectories[1]:
            plt.plot(
                np.array(traj)[:, 0],
                np.array(traj)[:, 1],
                linewidth=2.5,
                color="dimgray",
            )
            plt.arrow(
                np.array(traj)[10, 0],
                np.array(traj)[10, 1],
                np.array(traj)[11, 0] - np.array(traj)[10, 0],
                np.array(traj)[11, 1] - np.array(traj)[10, 1],
                shape="full",
                color="dimgray",
                lw=0,
                length_includes_head=True,
                head_width=0.15,
            )

        plt.xlabel("x [m]", fontsize=20)
        plt.ylabel("y [m]", fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlim((-6, 1.2))
        plt.ylim((-5.1, 1.3))
        plt.title(title, fontsize=25)
        ax = plt.gca()
        ax.add_patch(rect)
        ax.set_facecolor("lavender")
        ax.add_patch(
            plt.Circle(
                (self.goal[0], self.goal[1]),
                self.goal[2],
                color="cornflowerblue",
                fill=True,
                label="Goal Set",
            )
        )
        ax.legend(loc="upper left", fontsize=20, frameon=False)
        plt.savefig(path, format="eps")
        print(f"trajVSref plot is stored in {path}")
        # plt.show()

    def visualize_ref_vs_nn(self, traj_set, path=os.getcwd() + "/trajVSref.eps"):
        """_summary_

        Args:
            traj_set (list): a list of trajectories
        """

        traj_set_nn = []

        counter = 1
        for traj in traj_set:
            # append the new trajectory into the list
            traj_set_nn.append(self.give_single_trajectory(traj[0], control="nn")[0])
            print(
                f"nn trajectory generated for test {counter} in {traj_set_nn[-1].shape[0]} iterations"
            )
            counter += 1

        self.plot_trajectory_sets([traj_set, traj_set_nn], path=path)

    @staticmethod
    def normalize_action(action):
        """_summary_

        Args:
            action (_type_): _description_

        Returns:
            _type_: _description_
        """
        return np.concatenate(
            (action, np.array([np.linspace(0.0, 1.0, num=action.shape[0])]).T), axis=1
        )

    @staticmethod
    def sample_interval(min_x, max_x, res=1):
        """_summary_

        Args:
            min_x (_type_): _description_
            max_x (_type_): _description_
            res (int, optional): _description_. Defaults to 1.

        Returns:
            _type_: _description_
        """
        return (((max_x) - (min_x)) / res) * np.random.randint(0, res) + (min_x)

    @staticmethod
    def format_state2nn(state):
        """_summary_

        Args:
            state (_type_): _description_

        Returns:
            _type_: _description_
        """
        state_nn = np.zeros((4))
        state_nn[0] = state[0]
        state_nn[1] = state[1]
        state_nn[2] = np.sin(state[2])
        state_nn[3] = np.cos(state[2])

        return state_nn

    @staticmethod
    def adjust_pi_pi(theta):
        """_summary_

        Args:
            theta (_type_): _description_

        Returns:
            _type_: _description_
        """
        if theta > np.pi:
            temp = theta / (2 * np.pi) - 0.5
            k = np.ceil(temp)
            theta = theta - k * 2 * np.pi
        elif theta < -np.pi:
            temp = -theta / (2 * np.pi) - 0.5
            k = np.ceil(temp)
            theta = theta + k * 2 * np.pi
        return theta

    @staticmethod
    def give_angular_diff(state1, state2):
        """_summary_

        Args:
            state1 (_type_): _description_
            state2 (_type_): _description_

        Returns:
            _type_: _description_
        """
        dist = state2 - state1
        return math.atan2(np.sin(dist), np.cos(dist))

    @staticmethod
    def solve_qp(P, q, G=None, h=None, A=None, b=None):
        # pylint: disable=invalid-name
        # pylint: disable=too-many-arguments

        """_summary_

        Args:
            P (_type_): _description_
            q (_type_): _description_
            G (_type_, optional): _description_. Defaults to None.
            h (_type_, optional): _description_. Defaults to None.
            A (_type_, optional): _description_. Defaults to None.
            b (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        P = 0.5 * (P + P.T)  # make sure P is symmetric
        args = [cvxopt.matrix(P), cvxopt.matrix(q)]
        if G is not None:
            args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
            if A is not None:
                args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
        cvxopt.solvers.options["show_progress"] = False
        cvxopt.solvers.options["maxiters"] = 100
        sol = cvxopt.solvers.qp(*args)
        if "optimal" not in sol["status"]:
            return None
        return np.array(sol["x"]).reshape((P.shape[1],))
