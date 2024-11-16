import sympy as spy

from dg_commons.sim.models.spaceship_structures import (
    SpaceshipGeometry,
    SpaceshipParameters,
)


class SpaceshipDyn:
    sg: SpaceshipGeometry
    sp: SpaceshipParameters

    x: spy.Matrix
    u: spy.Matrix
    p: spy.Matrix

    n_x: int
    n_u: int
    n_p: int

    f: spy.Function
    A: spy.Function
    B: spy.Function
    F: spy.Function

    def __init__(self, sg: SpaceshipGeometry, sp: SpaceshipParameters):
        self.sg = sg
        self.sp = sp

        self.x = spy.Matrix(
            spy.symbols("x y psi vx vy dpsi delta m", real=True)
        )  # states
        self.u = spy.Matrix(spy.symbols("thrust ddelta", real=True))  # inputs
        self.p = spy.Matrix([spy.symbols("t_f", positive=True)])  # final time

        self.n_x = self.x.shape[0]  # number of states
        self.n_u = self.u.shape[0]  # number of inputs
        self.n_p = self.p.shape[0]

    def get_dynamics(
        self,
    ) -> tuple[spy.Function, spy.Function, spy.Function, spy.Function]:
        """
        Define dynamics and extract jacobians.
        Get dynamics for SCvx.
        0x 1y 2psi 3vx 4vy 5dpsi 6delta 7m
        """
        # Dynamics
        f = spy.zeros(self.n_x, 1)
        f[0] = self.x[3] * spy.cos(self.x[2]) - self.x[4] * spy.sin(self.x[2])  # dx/dt
        f[1] = self.x[3] * spy.sin(self.x[2]) + self.x[4] * spy.cos(self.x[2])  # dy/dt
        f[2] = self.x[5]  # dpsi/dt

        # Velocity dynamics
        f[3] = (1 / self.x[7]) * spy.cos(self.x[6]) * self.u[0] + self.x[5] * self.x[
            4
        ]  # dvx/dt
        f[4] = (1 / self.x[7]) * spy.sin(self.x[6]) * self.u[0] - self.x[5] * self.x[
            3
        ]  # dvy/dt
        # f[5] = -(self.sg.l_r / self.sg.I) * spy.sin(self.x[6]) * self.u[0]  # ddpsi/dt
        # I have no clue what I should be, i just took 1
        f[5] = -(self.sg.l_r / 1) * spy.sin(self.x[6]) * self.u[0]  # ddpsi/dt

        # Control dynamics
        f[6] = self.u[1]  # ddelta/dt

        # Mass dynamics
        f[7] = -self.sp.C_T * self.u[0]  # dm/dt

        A = f.jacobian(self.x)
        B = f.jacobian(self.u)
        F = f.jacobian(self.p)

        f_func = spy.lambdify((self.x, self.u, self.p), f, "numpy")
        A_func = spy.lambdify((self.x, self.u, self.p), A, "numpy")
        B_func = spy.lambdify((self.x, self.u, self.p), B, "numpy")
        F_func = spy.lambdify((self.x, self.u, self.p), F, "numpy")

        return f_func, A_func, B_func, F_func
