from dataclasses import dataclass, field
from typing import Union

import cvxpy as cvx
from dg_commons import PlayerName
from dg_commons.seq import DgSampledSequence
from dg_commons.sim.models.obstacles_dyn import DynObstacleState
from dg_commons.sim.models.spaceship import SpaceshipCommands, SpaceshipState
from dg_commons.sim.models.spaceship_structures import (
    SpaceshipGeometry,
    SpaceshipParameters,
)

from pdm4ar.exercises.ex11.discretization import *
from pdm4ar.exercises_def.ex11.utils_params import PlanetParams, SatelliteParams


@dataclass(frozen=True)
class SolverParameters:
    """
    Definition space for SCvx parameters in case SCvx algorithm is used.
    Parameters can be fine-tuned by the user.
    """

    # Cvxpy solver parameters
    solver: str = "ECOS"  # specify solver to use
    verbose_solver: bool = False  # if True, the optimization steps are shown
    max_iterations: int = 100  # max algorithm iterations

    # SCVX parameters (Add paper reference)
    lambda_nu: float = 1e5  # slack variable weight
    weight_p: NDArray = field(
        default_factory=lambda: 10 * np.array([[1.0]]).reshape((1, -1))
    )  # weight for final time

    tr_radius: float = 5  # initial trust region radius
    min_tr_radius: float = 1e-4  # min trust region radius
    max_tr_radius: float = 100  # max trust region radius
    rho_0: float = 0.0  # trust region 0
    rho_1: float = 0.25  # trust region 1
    rho_2: float = 0.9  # trust region 2
    alpha: float = 2.0  # div factor trust region update
    beta: float = 3.2  # mult factor trust region update

    # Discretization constants
    K: int = 50  # number of discretization steps
    N_sub: int = 5  # used inside ode solver inside discretization
    stop_crit: float = 1e-5  # Stopping criteria constant


class SpaceshipPlanner:
    """
    Feel free to change anything in this class.
    """

    planets: dict[PlayerName, PlanetParams]
    satellites: dict[PlayerName, SatelliteParams]
    spaceship: SpaceshipDyn
    sg: SpaceshipGeometry
    sp: SpaceshipParameters
    params: SolverParameters

    # Simpy variables
    x: spy.Matrix
    u: spy.Matrix
    p: spy.Matrix

    n_x: int
    n_u: int
    n_p: int

    X_bar: NDArray
    U_bar: NDArray
    p_bar: NDArray

    def __init__(
        self,
        planets: dict[PlayerName, PlanetParams],
        satellites: dict[PlayerName, SatelliteParams],
        sg: SpaceshipGeometry,
        sp: SpaceshipParameters,
    ):
        """
        Pass environment information to the planner.
        """
        self.planets = planets
        self.satellites = satellites
        self.sg = sg
        self.sp = sp

        # Solver Parameters
        self.params = SolverParameters()

        # Spaceship Dynamics
        self.spaceship = SpaceshipDyn(self.sg, self.sp)

        # Discretization Method
        # self.integrator = ZeroOrderHold(self.Spaceship, self.params.K, self.params.N_sub)
        self.integrator = FirstOrderHold(
            self.spaceship, self.params.K, self.params.N_sub
        )

        # Variables
        self.variables = self._get_variables()

        # Problem Parameters
        self.problem_parameters = self._get_problem_parameters()

        # We dont have any information about the init and goal state yet.
        # self.X_bar, self.U_bar, self.p_bar = self.initial_guess()

        # Constraints
        # constraints = self._get_constraints()

        # Objective
        # objective = self._get_objective()

        # Cvx Optimisation Problem
        # self.problem = cvx.Problem(objective, constraints)

    def compute_trajectory(
        self, init_state: SpaceshipState, goal_state: DynObstacleState
    ) -> tuple[DgSampledSequence[SpaceshipCommands], DgSampledSequence[SpaceshipState]]:
        """
        Compute a trajectory from init_state to goal_state.
        """
        self.init_state = init_state
        self.goal_state = goal_state

        self.X_bar, self.U_bar, self.p_bar = self.initial_guess()
        constraints = self._get_constraints()
        objective = self._get_objective()
        self.problem = cvx.Problem(objective, constraints)

        tr_radius = self.params.tr_radius

        # SCvx loop
        converged = False
        iteration = 0

        while not converged and iteration < self.params.max_iterations:
            # Store previous solution
            X_prev = self.X_bar.copy()

            # 1. Convexification step - linearize around current trajectory
            self._convexification()

            # 2. Solve convex subproblem
            try:
                status = self.problem.solve(
                    verbose=self.params.verbose_solver, solver=self.params.solver
                )

                if status not in ["optimal", "optimal_inaccurate"]:
                    print(f"Solver failed with status {self.problem.status}")
                    break

                # Extract solution
                self.X_bar = self.variables["X"].value
                self.U_bar = self.variables["U"].value
                self.p_bar = self.variables["p"].value

                # Check convergence
                state_change = np.linalg.norm(self.X_bar - X_prev)
                converged = state_change < self.params.stop_crit

                # Update trust region
                if not converged:
                    # Compare actual vs predicted improvement
                    actual_cost = self._evaluate_nonlinear_cost(
                        self.X_bar, self.U_bar, self.p_bar
                    )
                    predicted_cost = self.problem.value

                    rho = (actual_cost - predicted_cost) / predicted_cost

                    # Update trust region radius based on rho
                    if rho < self.params.rho_0:
                        tr_radius = tr_radius / self.params.alpha
                    elif rho >= self.params.rho_1:
                        tr_radius = tr_radius * self.params.beta

                    # Ensure trust region stays within bounds
                    tr_radius = np.clip(
                        tr_radius, self.params.min_tr_radius, self.params.max_tr_radius
                    )

            except cvx.SolverError:
                print(f"Solver failed at iteration {iteration}")
                break

            iteration += 1

        # Convert solution to time sequences
        tf = float(self.p_bar[0])
        ts = np.linspace(0, tf, self.params.K)

        # Create command sequence
        cmds_list = [
            SpaceshipCommands(float(self.U_bar[0, k]), float(self.U_bar[1, k]))
            for k in range(self.params.K)
        ]
        mycmds = DgSampledSequence[SpaceshipCommands](timestamps=ts, values=cmds_list)

        # Create state sequence
        states_list = [
            SpaceshipState(
                float(self.X_bar[0, k]),  # x
                float(self.X_bar[1, k]),  # y
                float(self.X_bar[2, k]),  # psi
                float(self.X_bar[3, k]),  # vx
                float(self.X_bar[4, k]),  # vy
                float(self.X_bar[5, k]),  # dpsi
                float(self.X_bar[6, k]),  # delta
                float(self.X_bar[7, k]),  # m
            )
            for k in range(self.params.K)
        ]
        mystates = DgSampledSequence[SpaceshipState](timestamps=ts, values=states_list)

        return mycmds, mystates

    def _evaluate_nonlinear_cost(self, X: NDArray, U: NDArray, p: NDArray) -> float:
        """
        Evaluate the true nonlinear cost of a trajectory
        """
        w_time = self.params.weight_p[0, 0]
        w_thrust = 0.1
        time_cost = w_time * p[0]
        thrust_cost = w_thrust * np.sum(np.square(U[0, :]))

        return float(time_cost + thrust_cost)

    def initial_guess(self) -> tuple[NDArray, NDArray, NDArray]:
        """
        Define initial guess for SCvx.
        """
        K = self.params.K

        X = np.zeros((self.spaceship.n_x, K))
        U = np.zeros((self.spaceship.n_u, K))
        p = np.zeros((self.spaceship.n_p))

        start_pos = np.array([self.init_state.x, self.init_state.y])
        goal_pos = np.array([self.goal_state.x, self.goal_state.y])
        distance = np.linalg.norm(goal_pos - start_pos)
        avg_speed = 2.0  # reasonable average speed in m/s
        p = np.array([max(distance / avg_speed, 5.0)])

        start = np.array(
            [
                self.init_state.x,
                self.init_state.y,
                self.init_state.psi,
                self.init_state.vx,
                self.init_state.vy,
                self.init_state.dpsi,
                self.init_state.delta,
                self.init_state.m,
            ]
        )

        goal = np.array(
            [
                self.goal_state.x,
                self.goal_state.y,
                self.goal_state.psi,
                self.goal_state.vx,
                self.goal_state.vy,
                0.0,  # dpsi
                0.0,  # delta
                start[7],  # keep same mass
            ]
        )

        # Linear interpolation for states
        for k in range(K):
            X[:, k] = start + (k / (K - 1)) * (goal - start)

        # Initial guess of zero controls
        # Could be improved by using nominal values
        U[0, :] = 1.0  # thrust
        U[1, :] = 0.0  # ddelta

        return X, U, p

    def _set_goal(self):
        """
        Sets goal for SCvx.
        """
        self.goal = cvx.Parameter((6, 1))
        pass

    def _get_variables(self) -> dict:
        """
        Define optimisation variables for SCvx.
        """
        variables = {
            "X": cvx.Variable((self.spaceship.n_x, self.params.K)),
            "U": cvx.Variable((self.spaceship.n_u, self.params.K)),
            "p": cvx.Variable(self.spaceship.n_p),
        }

        return variables

    def _get_problem_parameters(self) -> dict:
        """
        Define problem parameters for SCvx.
        """
        problem_parameters = {
            "init_state": cvx.Parameter(self.spaceship.n_x)
            # ...
        }

        return problem_parameters

    def _get_constraints(self) -> list[cvx.Constraint]:
        """
        Define constraints for SCvx.
        """
        X = self.variables["X"]
        U = self.variables["U"]
        p = self.variables["p"]
        K = self.params.K

        constraints = []

        slack_vars = {
            "pos": cvx.Variable(),
            "vel": cvx.Variable(),
            "dir": cvx.Variable(),
            "dyn": cvx.Variable(shape=(self.spaceship.n_x, K - 1)),
        }
        print(f"Initial state: {self.init_state}")
        print(f"Goal state: {self.goal_state}")

        # 1. Initial state constraint
        constraints.append(X[:, 0] == self.problem_parameters["init_state"])

        # 2. Final state constraints with tolerance
        pos_tol = 0.1  # meters
        vel_tol = 0.1  # m/s
        dir_tol = 0.1  # radians

        # Goal Position constraint
        constraints.append(
            cvx.norm(X[0:2, -1] - np.array([self.goal_state.x, self.goal_state.y]))
            <= pos_tol + slack_vars["pos"]
        )

        # Time constraint
        t_min = 0.0
        t_max = 5.0
        constraints += [p[0] >= t_min, p[0] <= t_max]

        # Goal Velocity constraint
        constraints.append(
            cvx.norm(X[3:5, -1] - np.array([self.goal_state.vx, self.goal_state.vy]))
            <= vel_tol + slack_vars["vel"]
        )

        # Direction constraint
        constraints.append(
            cvx.abs(X[2, -1] - self.goal_state.psi) <= dir_tol + slack_vars["dir"]
        )

        # 3. State and control bounds
        # Thrust limits
        F_min = self.sp.thrust_limits[0]
        F_max = self.sp.thrust_limits[1]
        constraints += [U[0, :] <= F_max, U[0, :] >= F_min]

        # Thrust angle rate limits
        delta_dot_min = self.sp.ddelta_limits[0]
        delta_dot_max = self.sp.ddelta_limits[1]
        constraints += [U[1, :] <= delta_dot_max, U[1, :] >= delta_dot_min]

        # Thrust angle limits
        delta_min = self.sp.delta_limits[0]
        delta_max = self.sp.delta_limits[1]
        constraints += [X[6, :] <= delta_max, X[6, :] >= delta_min]

        # Mass constraint
        constraints += [X[7, :] >= self.sp.m_v]

        # 4. Dynamics constraints (will be filled in by convexification)
        # These constraints ensure the trajectory follows the rocket's physics
        for k in range(K - 1):
            if isinstance(self.integrator, ZeroOrderHold):
                dx = (
                    self.integrator.A_bar[:, k].reshape(
                        self.spaceship.n_x, self.spaceship.n_x
                    )
                    @ X[:, k]
                    + self.integrator.B_bar[:, k].reshape(
                        self.spaceship.n_x, self.spaceship.n_u
                    )
                    @ U[:, k]
                    + self.integrator.F_bar[:, k].reshape(self.spaceship.n_x, 1) @ p
                    + self.integrator.r_bar[:, k]
                )
            else:  # FirstOrderHold
                dx = (
                    self.integrator.A_bar[:, k].reshape(
                        self.spaceship.n_x, self.spaceship.n_x
                    )
                    @ X[:, k]
                    + self.integrator.B_plus_bar[:, k].reshape(
                        self.spaceship.n_x, self.spaceship.n_u
                    )
                    @ U[:, k + 1]
                    + self.integrator.B_minus_bar[:, k].reshape(
                        self.spaceship.n_x, self.spaceship.n_u
                    )
                    @ U[:, k]
                    + self.integrator.F_bar[:, k].reshape(self.spaceship.n_x, 1) @ p
                    + self.integrator.r_bar[:, k]
                )
            constraints.append(X[:, k + 1] == dx + slack_vars["dyn"][:, k])

        # print(f"A_bar shape: {self.A_bar.shape}")
        # print(
        #     f"B_bar shape: {self.B_bar.shape if isinstance(self.integrator, ZeroOrderHold) else self.B_plus_bar.shape}"
        # )
        # print(f"F_bar shape: {self.F_bar.shape}")
        # print(f"r_bar shape: {self.r_bar.shape}")

        self.slack_cost = (
            cvx.sum(slack_vars["pos"])
            + cvx.sum(slack_vars["vel"])
            + cvx.sum(slack_vars["dir"])
            + cvx.sum(cvx.sum(slack_vars["dyn"]))
        )

        return constraints

    def _get_objective(self) -> Union[cvx.Minimize, cvx.Maximize]:
        """
        Define objective for SCvx.
        """
        # Example objective
        # X = self.variables["X"]
        U = self.variables["U"]
        p = self.variables["p"]  # final time

        # Weights for different objective terms
        w_time = self.params.weight_p[0, 0]  # weight for final time
        w_thrust = 0.1  # weight for thrust usage

        # 1. Minimize final time
        time_cost = w_time * p[0]

        # 2. Minimize thrust usage
        thrust_cost = w_thrust * cvx.sum(cvx.square(U[0, :]))

        # Total objective
        objective = time_cost + thrust_cost + 1e6 * self.slack_cost

        return cvx.Minimize(objective)

    def _convexification(self):
        """
        Perform convexification step, i.e. Linearization and Discretization
        and populate Problem Parameters.
        """
        # ZOH
        # A_bar, B_bar, F_bar, r_bar = self.integrator.calculate_discretization(self.X_bar, self.U_bar, self.p_bar)
        # FOH
        A_bar, B_plus_bar, B_minus_bar, F_bar, r_bar = (
            self.integrator.calculate_discretization(self.X_bar, self.U_bar, self.p_bar)
        )

        self.problem_parameters["init_state"].value = self.X_bar[:, 0]
        # ...

    def _check_convergence(self) -> bool:
        """
        Check convergence of SCvx.
        """

        pass

    def _update_trust_region(self):
        """
        Update trust region radius.
        """
        pass

    @staticmethod
    def _extract_seq_from_array() -> (
        tuple[DgSampledSequence[SpaceshipCommands], DgSampledSequence[SpaceshipState]]
    ):
        """
        Example of how to create a DgSampledSequence from numpy arrays and timestamps.
        """
        ts = (0, 1, 2, 3, 4)
        # in case my planner returns 3 numpy arrays
        F = np.array([0, 1, 2, 3, 4])
        ddelta = np.array([0, 0, 0, 0, 0])
        cmds_list = [SpaceshipCommands(f, dd) for f, dd in zip(F, ddelta)]
        mycmds = DgSampledSequence[SpaceshipCommands](timestamps=ts, values=cmds_list)

        # in case my state trajectory is in a 2d array
        npstates = np.random.rand(len(ts), 8)
        states = [SpaceshipState(*v) for v in npstates]
        mystates = DgSampledSequence[SpaceshipState](timestamps=ts, values=states)
        return mycmds, mystates
