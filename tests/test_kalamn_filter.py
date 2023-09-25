from typing import Dict

import control
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng

import pysysid.kalman_filter as kf
import pysysid.pm2i as pm2i
import pysysid.signal_generators as sg


class Motor(pm2i.ProcessModelGenerator):
    """This class is used to simulate a motor's process model.

    Parameters
    ----------
    dt: float, optional
        If None, process model is supposed to be continuous.
        Else, specifies the sampling time for a discrete process model
    **kwargs: Dict[str, float]
        Process model parameters that define physical quantities within the
        motor.

    """

    def _compute_A(self, l21: float, l22: float, l31: float, l32: float) -> np.ndarray:
        return np.array(
            [
                [0, 0, 1],
                [0, l21, l22],
                [0, l31, l32],
            ]
        )

    def _compute_B(self, l23: float, l33: float) -> np.ndarray:
        return np.array(
            [
                [0, 0],
                [l23, 0],
                [0, l33],
            ]
        )

    def _compute_C(self):
        return np.array([[1, 0, 0], [0, 1, 0]])

    def _compute_M(self):
        return np.eye(2)

    def __init__(
        self, E_w: np.ndarray = None, E_v: np.ndarray = None, **kwargs: Dict[str, float]
    ):
        super().__init__(E_w=E_w, E_v=E_v, **kwargs)

        # good thing to extract params here so that error
        # is raised in constructor instead of when the simulation is running
        self.R = self.params["R"]
        self.L = self.params["L"]

        self.J = self.params["J"]
        self.B = self.params["B"]

        self.K = self.params["K"]

        # x = [
        # theta
        # i
        # omega
        # ]

        # u = [
        # v
        # tau_d
        # ]

        self.l21 = -self.R / self.L
        self.l22 = -self.K / self.L
        self.l23 = 1 / self.L

        self.l31 = self.K / self.J
        self.l32 = -self.B / self.J
        self.l33 = 1 / self.J

        self.mat_A = self._compute_A(
            l21=self.l21,
            l22=self.l22,
            l31=self.l31,
            l32=self.l32,
        )

        self.mat_B = self._compute_B(l23=self.l23, l33=self.l33)
        self.mat_L = self.mat_B
        self.mat_C = self._compute_C()
        self.mat_M = self._compute_M()
        self.mat_D = np.zeros((2, 2))

    def compute_state_derivative(
        self,
        t: float,
        total_state: np.ndarray,
        total_input: np.ndarray,
    ) -> np.ndarray:
        if self.E_w is None:
            return self.mat_A @ total_state + self.mat_B @ total_input

        u = total_input[:2, :]
        w = total_input[2:, :]
        return self.mat_A @ total_state + self.mat_B @ u + self.mat_L @ w

    def compute_output(
        self,
        t: float,
        total_state: np.ndarray,
        total_input: np.ndarray,
    ) -> np.ndarray:
        if self.E_v is None:
            return self.mat_C @ total_state
        u = total_input[:2, :]
        v = total_input[2:, :]
        return self.mat_C @ total_state + self.mat_M @ v

    def generate_process_model_to_integrate(
        self,
        rng: np.random.Generator = None,
    ) -> pm2i.ProcessModelToIntegrate:
        return pm2i.ProcessModelToIntegrate(
            nbr_states=3,
            nbr_inputs=2,
            nbr_outputs=2,
            fct_for_x_dot=self.compute_state_derivative,
            fct_for_y=self.compute_output,
            nbr_process_noise_inputs=2,
            nbr_measurement_noise_inputs=2,
            E_w=self.E_w,
            E_v=self.E_v,
            rng=rng,
        )


class MotorKalman(Motor):
    """This class is used to simulate a motor's process model, while estimating
    its parameters.

    """

    def __init__(self):
        pass

    def extract_x(self, total_state: np.ndarray) -> np.ndarray:
        return total_state[:3, :]

    def extract_l(self, total_state: np.ndarray) -> np.ndarray:
        return total_state[3:, 0]

    def compute_state_derivative(
        self,
        t: float,
        total_state: np.ndarray,
        total_input: np.ndarray,
    ):
        x = self.extract_x(total_state)
        l = self.extract_l(total_state)

        l21 = l[0]
        l22 = l[1]
        l23 = l[2]

        l31 = l[3]
        l32 = l[4]
        l33 = l[5]

        return np.vstack(
            (
                self._compute_A(l21, l22, l31, l32) @ x
                + self._compute_B(l23, l33) @ total_input,
                np.zeros((6, 1)),
            )
        )

    def compute_output(
        self, t: float, total_state: np.ndarray, total_input: np.ndarray
    ) -> np.ndarray:
        x = self.extract_x(total_state)

        return self._compute_C() @ x

    def compute_df_dx(
        self, t: float, total_state: np.ndarray, total_input: np.ndarray
    ) -> np.ndarray:
        x = self.extract_x(total_state)
        l = self.extract_l(total_state)

        theta = x[0, 0]
        i = x[1, 0]
        omega = x[2, 0]

        v = total_input[0, 0]
        tau_d = total_input[1, 0]

        l21 = l[0]
        l22 = l[1]
        l23 = l[2]

        l31 = l[3]
        l32 = l[4]
        l33 = l[5]

        M_kal = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [i, omega, v, 0, 0, 0],
                [0, 0, 0, i, omega, tau_d],
            ]
        )

        return np.block(
            [
                [self._compute_A(l21, l22, l31, l32), M_kal],
                [np.zeros((6, 9))],
            ]
        )

    def compute_df_du(
        self, t: float, total_state: np.ndarray, total_input: np.ndarray
    ) -> np.ndarray:
        l = self.extract_l(total_state)

        l21 = l[0]
        l22 = l[1]
        l23 = l[2]

        l31 = l[3]
        l32 = l[4]
        l33 = l[5]
        return np.vstack((self._compute_B(l23, l33), np.zeros((6, 2))))

    def compute_df_dw(
        self, t: float, total_state: np.ndarray, total_input: np.ndarray
    ) -> np.ndarray:
        return self.compute_df_du(t, total_state, total_input)

    def compute_dg_dx(
        self, t: float, total_state: np.ndarray, total_input: np.ndarray
    ) -> np.ndarray:
        return np.hstack((self._compute_C(), np.zeros((2, 6))))

    def compute_dg_dv(
        self, t: float, total_state: np.ndarray, total_input: np.ndarray
    ) -> np.ndarray:
        return self._compute_M()

    def generate_process_model_to_integrate(
        self,
    ) -> pm2i.ProcessModelToIntegrate:
        return pm2i.ProcessModelToIntegrate(
            nbr_states=9,  # 3 states + 6 parameters evaluated as states
            nbr_inputs=2,
            nbr_outputs=2,
            fct_for_x_dot=self.compute_state_derivative,
            fct_for_y=self.compute_output,
            df_dx=self.compute_df_dx,
            df_du=self.compute_df_du,
            df_dw=self.compute_df_dw,
            dg_dx=self.compute_dg_dx,
            dg_dv=self.compute_dg_dv,
            nbr_process_noise_inputs=2,
            nbr_measurement_noise_inputs=2,
        )


def retrieve_params(l: np.ndarray) -> Dict[str, float]:
    l21 = l[0]
    l22 = l[1]
    l23 = l[2]
    l31 = l[3]
    l32 = l[4]
    l33 = l[5]

    L = 1 / l23
    J = 1 / l33
    params = {
        "R": -l21 * L,
        "L": L,
        "J": J,
        "B": -l32 * J,
        "K": -l22 * L,
    }

    return params


class TestCEKF:
    """Used to test the CEKF class from kalman_filter.py."""

    def test_fit(self):
        """Used to test the fit method from the CEKf class."""

        assert True


if __name__ == "__main__":
    PLOTTING = True

    motor_params = {
        "R": 1.0,
        "L": 6e-4,
        "J": 8e-7,
        "B": 1.33e-5,
        "K": 2.38e-2,
    }

    E_w = np.diag([1e-5, 1e-9])
    E_v = np.diag([1e-12, 1e-9])

    x0 = np.array([0, 0, 0])
    params_x0 = np.array([-1e4, -1e2, 1e4, 1e5, -1e2, 1e7])

    E_x_0_kal = 0.0001 * np.eye(9)

    v_sg = sg.SquareGenerator(period=1, pulse_width=0.5, amplitude=1)

    tau_d_sg = sg.SineGenerator(frequency=0.3, amplitude=0.01, phase=0)

    input_gen = sg.InputGenerator([v_sg, tau_d_sg])

    motor = Motor(E_w=E_w, E_v=E_v, **motor_params)

    original_params = np.array(
        [motor.l21, motor.l22, motor.l23, motor.l31, motor.l32, motor.l33]
    )

    # motor_ss = control.StateSpace(motor.mat_A, motor.mat_B, motor.mat_C, motor.mat_D)
    # motor_poles = control.pole(motor_ss)

    # init_params = retrieve_params(params_x0)
    # motor_init = Motor(**init_params)
    # init_motor_ss = control.StateSpace(
    #     motor_init.mat_A, motor_init.mat_B, motor_init.mat_C, motor_init.mat_D
    # )
    # init_motor_poles = control.pole(init_motor_ss)

    print(f"Original params: {original_params}")
    print(f"Init params CEKF: {params_x0}")

    rng = default_rng(seed=1)

    motor_pm2i = motor.generate_process_model_to_integrate(rng)

    sol_t, sol_u, sol_x, sol_y = motor_pm2i.integrate(
        compute_u_from_t=input_gen.value_at_t, dt_data=0.01, t_end=2, x0=x0
    )

    if PLOTTING:
        fig, ax = plt.subplots(3, 2)

        v = sol_u[0, :]
        tau_d = sol_u[1, :]

        theta = sol_y[0, :]
        i = sol_y[1, :]
        omega = sol_x[2, :]

        ax[0][0].set_xlabel(r"$t$ (s)")
        ax[0][0].set_ylabel(r"$v(t)$ (V)")
        ax[0][0].plot(sol_t, v, label=r"Voltage Input", color="C0")
        ax[0][0].legend(loc="upper right")

        ax[1][0].set_xlabel(r"$t$ (s)")
        ax[1][0].set_ylabel(r"$\tau_{d}(t)$ (Nm)")
        ax[1][0].plot(sol_t, tau_d, label=r"Torque Disturbance", color="C0")
        ax[1][0].legend(loc="upper right")

        ax[0][1].set_xlabel(r"$t$ (s)")
        ax[0][1].set_ylabel(r"$\theta(t)$ (rad)")
        ax[0][1].plot(sol_t, theta, label=r"Angular Position", color="C0")
        ax[0][1].legend(loc="upper right")

        ax[1][1].set_xlabel(r"$t$ (s)")
        ax[1][1].set_ylabel(r"$i(t)$ (rad/s)")
        ax[1][1].plot(sol_t, i, label=r"Current", color="C0")
        ax[1][1].legend(loc="upper right")

        ax[2][1].set_xlabel(r"$t$ (s)")
        ax[2][1].set_ylabel(r"$\omega(t)$ (rad/s)")
        ax[2][1].plot(sol_t, omega, label=r"Angular Velocity", color="C0")
        ax[2][1].legend(loc="upper right")

        plt.show()

    X = np.block([[sol_t.reshape(1, len(sol_t)).T, sol_u.T]])

    y = sol_y.T

    A_con = np.hstack((np.zeros((6, 3)), np.diag([-1, -1, 1, 1, -1, 1])))
    b_con = np.zeros((6, 1))

    cekf_regressor = kf.CEKF(
        process_model=MotorKalman,
        physical_constraint_matrix_A=A_con,
        physical_constraint_vector_b=b_con,
        n_params=6,
    )

    x0_kal = np.hstack((x0, params_x0))

    cekf_regressor.fit(X, y, x0=x0_kal, E_x_0=E_x_0_kal, E_w=E_w, E_v=E_v)

    best_fit_params = cekf_regressor.optimal_parameters_

    if PLOTTING:
        sol_x_kal = cekf_regressor.x_arr_

        theta_kal = sol_x_kal[0, :]
        i_kal = sol_x_kal[1, :]
        omega_kal = sol_x_kal[2, :]

        ax[0][1].set_xlabel(r"$t$ (s)")
        ax[0][1].set_ylabel(r"$\theta(t)$ (rad)")
        ax[0][1].plot(sol_t, theta_kal, label=r"Estimated Angular Position", color="C2")
        ax[0][1].legend(loc="upper right")

        ax[1][1].set_xlabel(r"$t$ (s)")
        ax[1][1].set_ylabel(r"$i(t)$ (rad/s)")
        ax[1][1].plot(sol_t, i_kal, label=r"Estimated Current", color="C2")
        ax[1][1].legend(loc="upper right")

        ax[2][1].set_xlabel(r"$t$ (s)")
        ax[2][1].set_ylabel(r"$\omega(t)$ (rad/s)")
        ax[2][1].plot(sol_t, omega_kal, label=r"Estimated Angular Velocity", color="C2")
        ax[2][1].legend(loc="upper right")

    if PLOTTING:
        best_cekf_motor_params = retrieve_params(best_fit_params)

        print(f"{best_cekf_motor_params=}")

        motor_fit = Motor(**best_cekf_motor_params)
        motor_pm2i_fit = motor_fit.generate_process_model_to_integrate()

        sol_t_fit, sol_u_fit, sol_x_fit, sol_y_fit = motor_pm2i_fit.integrate(
            compute_u_from_t=input_gen.value_at_t, dt_data=0.01, t_end=2, x0=x0
        )

        v_fit = sol_u_fit[0, :]
        tau_d_fit = sol_u_fit[1, :]

        theta_fit = sol_y_fit[0, :]
        i_fit = sol_y_fit[1, :]
        omega_fit = sol_x_fit[2, :]

        ax[0][1].set_xlabel(r"$t$ (s)")
        ax[0][1].set_ylabel(r"$\theta(t)$ (rad)")
        ax[0][1].plot(
            sol_t_fit, theta_fit, label=r"Angular Position  - Best Fit", color="C1"
        )
        ax[0][1].legend(loc="upper right")

        ax[1][1].set_xlabel(r"$t$ (s)")
        ax[1][1].set_ylabel(r"$i(t)$ (rad/s)")
        ax[1][1].plot(sol_t, i_fit, label=r"Current - Best Fit", color="C1")
        ax[1][1].legend(loc="upper right")

        ax[1][1].set_xlabel(r"$t$ (s)")
        ax[1][1].set_ylabel(r"$\omega(t)$ (rad/s)")
        ax[1][1].plot(
            sol_t_fit, omega_fit, label=r"Angular Velocity  - Best Fit", color="C1"
        )
        ax[1][1].legend(loc="upper right")

        plt.show()
