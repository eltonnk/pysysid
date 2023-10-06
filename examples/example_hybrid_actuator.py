from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

import pysysid.genetic as genetic
import pysysid.pm2i as pm2i
import pysysid.signal_generators as sg


class HybridActuatorPMG(pm2i.ProcessModelGenerator):
    def __init__(self, dt: float = None, **kwargs: Dict[str, float]):
        super().__init__(dt, **kwargs)
        self.brake_alpha_0 = self.params["brake_alpha_0"]
        self.brake_alpha_1 = self.params["brake_alpha_1"]

        self.brake_A = self.params["brake_A"]

        self.brake_c_0 = self.params["brake_c_0"]
        self.brake_c_1 = self.params["brake_c_1"]

        self.brake_eta = self.params["brake_eta"]
        self.brake_gamma = self.params["brake_gamma"]
        self.brake_beta = self.params["brake_beta"]

        self.brake_J = self.params["brake_J"]

        self.brake_n = self.params["brake_n"]

        self.motor_K = self.params["motor_K"]

        self.motor_R = self.params["motor_R"]
        self.motor_L = self.params["motor_L"]

        self.motor_B = self.params["motor_B"]
        self.motor_J = self.params["motor_J"]

        self.gear_ratio_n = self.params["gear_ratio_n"]

        # process model constants
        self.l11 = -self.motor_R / self.motor_L
        self.l12 = self.gear_ratio_n * self.motor_K / self.motor_L
        self.l13 = 1 / self.motor_L

        self.l21 = -self.brake_eta

        self.J = self.gear_ratio_n**2 * self.motor_J + self.brake_J
        self.l31 = -self.gear_ratio_n * self.motor_K / self.J
        self.l32 = -self.gear_ratio_n**2 * self.motor_B / self.J
        self.l33 = -self.brake_alpha_0 / self.J
        self.l34 = -self.brake_alpha_1 / self.J
        self.l37 = -self.brake_c_0 / self.J
        self.l38 = -self.brake_c_1 / self.J
        self.l39 = 1 / self.J

        self.brake_n_1 = self.brake_n - 1

    def check_valid_brake_voltage(self, v_b: float):
        if v_b < 0:
            raise ValueError("Input voltage should always be positive for te brake.")

    def _omega_dot(self, total_state: np.ndarray, tau_u: float) -> float:
        # hybrid actuator process model
        i = total_state[0, 0]
        v_a = total_state[1, 0]
        omega = total_state[2, 0]
        z = total_state[3, 0]
        theta = total_state[4, 0]

        return (
            self.l31 * i
            + self.l32 * omega
            + (self.l33 + self.l34 * v_a) * z
            + (self.l37 + self.l38 * v_a) * omega
            + self.l39 * tau_u
        )

    def compute_state_derivative(
        self, total_state: np.ndarray, total_input: np.ndarray
    ) -> np.ndarray:
        # hybrid actuator process model
        i = total_state[0, 0]
        v_a = total_state[1, 0]
        omega = total_state[2, 0]
        z = total_state[3, 0]
        theta = total_state[4, 0]

        # Hybrid actuator inputs
        v_m = total_input[0, 0]
        v_b = total_input[1, 0]
        # Hand inputs
        tau_u = total_input[2, 0]

        self.check_valid_brake_voltage(v_b)

        abs_z = np.abs(z)
        abs_z_n_1 = np.power(abs_z, self.brake_n_1)
        abs_z_n = abs_z_n_1 * abs_z

        return np.array(
            [
                [self.l11 * i + self.l12 * omega + self.l13 * v_m],
                [self.l21 * (v_a - v_b)],
                [self._omega_dot(total_state, tau_u)],
                [
                    -self.brake_gamma * np.abs(omega) * z * abs_z_n_1
                    - self.brake_beta * omega * abs_z_n
                    + self.brake_A * omega
                ],
                [omega],
            ]
        )

    def _alpha_v_a(self, v_a: float) -> float:
        return self.brake_alpha_0 + self.brake_alpha_1 * v_a

    def _c_v_a(self, v_a: float) -> float:
        return self.brake_c_0 + self.brake_c_1 * v_a

    def compute_output(
        self, total_state: np.ndarray, total_input: np.ndarray
    ) -> np.ndarray:
        # hybrid actuator process model
        i = total_state[0, 0]
        v_a = total_state[1, 0]
        theta = total_state[4, 0]

        return np.array(
            [
                [i],
                [v_a],
                [theta],
            ]
        )

    def generate_process_model_to_integrate(
        self,
    ) -> pm2i.ProcessModelToIntegrate:
        ha_pm2i = pm2i.ProcessModelToIntegrate(
            nbr_states=5,
            nbr_inputs=3,
            nbr_outputs=3,
            fct_for_x_dot=self.compute_state_derivative,
            fct_for_y=self.compute_output,
        )

        return ha_pm2i


if __name__ == "__main__":
    motor_params = {
        "brake_alpha_0": 3.1428,
        "brake_alpha_1": 0.000576362,
        "brake_A": 12.1487,
        "brake_c_0": 0.3347,
        "brake_c_1": 0.5919,
        "brake_eta": 63.5591,
        "brake_gamma": 2.4246e-5,
        "brake_beta": 15.6237,
        "brake_J": 12e-7,
        "brake_n": 26,
        "motor_K": 2.38e-2,
        "motor_R": 1.0,
        "motor_L": 6e-4,
        "motor_B": 1.33e-5,
        "motor_J": 8e-7,
        "gear_ratio_n": 20,
    }

    # None if continous, float if discrete
    motor_dt = None
    x0 = None

    v_m_sg = sg.SquareGenerator(period=1, pulse_width=0.5, amplitude=1)

    v_a_sg = sg.SquareGenerator(period=1.5, pulse_width=0.5, amplitude=12, offset=12)

    tau_d_sg = sg.SineGenerator(frequency=0.3, amplitude=0.01, phase=0)

    input_gen = sg.InputGenerator([v_m_sg, v_a_sg, tau_d_sg])

    ha_pmg = HybridActuatorPMG(dt=motor_dt, **motor_params)

    ha_pm2i = ha_pmg.generate_process_model_to_integrate()

    sol_t, sol_u, _, sol_y = ha_pm2i.integrate(
        compute_u_from_t=input_gen.value_at_t, dt_data=0.01, t_end=2, x0=x0
    )

    fig, ax = plt.subplots(2, 2)

    v_m = sol_u[0, :]
    tau_d = sol_u[2, :]

    theta = sol_y[0, :]
    omega = sol_y[2, :]

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
    ax[1][1].set_ylabel(r"$\omega(t)$ (rad/s)")
    ax[1][1].plot(sol_t, omega, label=r"Angular Velocity", color="C0")
    ax[1][1].legend(loc="upper right")

    X = np.block([[sol_t.reshape(1, len(sol_t)).T, sol_u.T]])

    y = sol_y.T

    chromosome_parameter_ranges = {}

    range_var = 0.99
    for key, value in motor_params.items():
        chromosome_parameter_ranges[key] = (
            (1 - range_var) * value,
            (1 + range_var) * value,
        )

    genetic_algo_regressor = genetic.Genetic(
        process_model=Motor,
        dt=motor_dt,
        compute_u_from_t=input_gen.value_at_t,
        n_chromosomes=100,
        replace_with_best_ratio=0.01,
        can_terminate_after_index=10,
        ratio_max_error_for_termination=0.2,
        seed=2,
        chromosome_parameter_ranges=chromosome_parameter_ranges,
    )

    genetic_algo_regressor.fit(X, y, n_iter=50, x0=x0)

    best_fit_params = genetic_algo_regressor._elite_chromosome

    best_chromosome_motor_params = genetic_algo_regressor._gen_chromosome_dict(
        best_fit_params
    )

    print(f"{best_chromosome_motor_params=}")

    motor_fit = Motor(dt=motor_dt, **best_chromosome_motor_params)
    motor_pm2i_fit = motor_fit.generate_process_model_to_integrate()

    sol_t_fit, sol_u_fit, _, sol_y_fit = motor_pm2i_fit.integrate(
        compute_u_from_t=input_gen.value_at_t, dt_data=0.01, t_end=2, x0=x0
    )

    v_fit = sol_u_fit[0, :]
    tau_d_fit = sol_u_fit[1, :]

    theta_fit = sol_y_fit[0, :]
    omega_fit = sol_y_fit[2, :]

    ax[0][0].set_xlabel(r"$t$ (s)")
    ax[0][0].set_ylabel(r"$v(t)$ (V)")
    ax[0][0].plot(sol_t_fit, v_fit, label=r"Voltage Input - Best Fit", color="C1")
    ax[0][0].legend(loc="upper right")

    ax[1][0].set_xlabel(r"$t$ (s)")
    ax[1][0].set_ylabel(r"$\tau_{d}(t)$ (Nm)")
    ax[1][0].plot(
        sol_t_fit, tau_d_fit, label=r"Torque Disturbance - Best Fit", color="C1"
    )
    ax[1][0].legend(loc="upper right")

    ax[0][1].set_xlabel(r"$t$ (s)")
    ax[0][1].set_ylabel(r"$\theta(t)$ (rad)")
    ax[0][1].plot(
        sol_t_fit, theta_fit, label=r"Angular Position  - Best Fit", color="C1"
    )
    ax[0][1].legend(loc="upper right")

    ax[1][1].set_xlabel(r"$t$ (s)")
    ax[1][1].set_ylabel(r"$\omega(t)$ (rad/s)")
    ax[1][1].plot(
        sol_t_fit, omega_fit, label=r"Angular Velocity  - Best Fit", color="C1"
    )
    ax[1][1].legend(loc="upper right")

    plt.show()

    original_params = np.array(list(motor_params.values()))
