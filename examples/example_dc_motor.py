from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
# import yappi

import pysysid.genetic as genetic
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

    def __init__(self, dt: float = None, **kwargs: Dict[str, float]):
        super().__init__(dt, **kwargs)

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

        self.mat_A = np.array(
            [
                [-self.R / self.L, -self.K / self.L],
                [self.K / self.J, -self.B / self.J],
            ]
        )

        self.mat_B = np.array(
            [
                [1 / self.L, 0],
                [0, 1 / self.J],
            ]
        )

        # full state feedback for ease of debugging
        self.mat_C = np.eye(2)

        self.mat_D = np.zeros((2, 2))

    def compute_state_derivative(
        self,
        t: float,
        total_state: np.ndarray,
        total_input: np.ndarray,
    ) -> np.ndarray:
        x_dot = self.mat_A @ total_state + self.mat_B @ total_input

        return x_dot

    def compute_output(
        self,
        t: float,
        total_state: np.ndarray,
        total_input: np.ndarray,
    ) -> np.ndarray:
        return self.mat_C @ total_state + self.mat_D @ total_input

    def param_inequality_constraint(params: Dict[str, float]) -> np.ndarray:
        # all params should be positive. thus if chromosome is x, x_i >= 0.
        # since ineqaulity constraint should be of the form h(x) <= 0, we have

        h_x = []
        # all params should be positive. since inequality constraint should be
        # of the form h(x) <= 0, we multiply every constraint by -1
        for key, val in params.items():
            h_x.append(-1.0 * val)

        return h_x

    def generate_process_model_to_integrate(self) -> pm2i.ProcessModelToIntegrate:
        return pm2i.ProcessModelToIntegrate(
            nbr_states=2,
            nbr_inputs=2,
            nbr_outputs=2,
            fct_for_x_dot=self.compute_state_derivative,
            fct_for_y=self.compute_output,
        )


def _main():

    params = {
        "legend.fontsize": "x-large",
        #   'figure.figsize': (15, 5),
        "axes.labelsize": "x-large",
        "axes.titlesize": "x-large",
        "xtick.labelsize": "x-large",
        "ytick.labelsize": "x-large",
    }
    motor_params = {
        "R": 1.0,
        "L": 6e-4,
        "J": 8e-7,
        "B": 1.33e-5,
        "K": 2.38e-2,
    }

    # None if continous, float if discrete
    motor_dt = None
    x0 = None

    v_sg = sg.SquareGenerator(period=1, pulse_width=0.5, amplitude=1)

    tau_d_sg = sg.SineGenerator(frequency=0.3, amplitude=0.01, phase=0)

    input_gen_train = sg.InputGenerator([v_sg, tau_d_sg])

    motor = Motor(dt=motor_dt, **motor_params)

    motor_pm2i = motor.generate_process_model_to_integrate()

    sol_t, sol_u, _, sol_y = motor_pm2i.integrate(
        compute_u_from_t=input_gen_train.value_at_t, dt_data=0.01, t_end=2, x0=x0
    )

    fig, ax = plt.subplots(2, 2)

    v = sol_u[0, :]
    tau_d = sol_u[1, :]

    i_m = sol_y[0, :]
    omega = sol_y[1, :]


    ax[0][0].set_ylabel(r"$v(t)$ (V)")
    ax[0][0].plot(sol_t, v, color="C0")


    ax[1][0].set_xlabel(r"$t$ (s)")
    ax[1][0].set_ylabel(r"$\tau_{u}(t)$ (Nm)")
    ax[1][0].plot(sol_t, tau_d, color="C0")



    ax[0][1].set_ylabel(r"$i_m(t)$ (A)")
    ax[0][1].plot(sol_t, i_m, label=r"Simulated Motor Data", color="C0")
    

    ax[1][1].set_xlabel(r"$t$ (s)")
    ax[1][1].set_ylabel(r"$\omega(t)$ (rad/s)")
    ax[1][1].plot(sol_t, omega, color="C0")


    X = np.block([[sol_t.reshape(1, len(sol_t)).T, sol_u.T]])

    y = sol_y.T

    chromosome_parameter_ranges = {}

    range_var = 0.1
    for key, value in motor_params.items():
        chromosome_parameter_ranges[key] = (
            (1 - range_var) * value,
            (1 + range_var) * value,
        )

    genetic_algo_regressor = genetic.Genetic(
        process_model=Motor,
        dt=motor_dt,
        compute_u_from_t=input_gen_train.value_at_t,
        n_chromosomes=400,
        replace_with_best_ratio=0.01,
        can_terminate_after_index=10,
        ratio_max_error_for_termination=0.2,
        seed=2,
        chromosome_parameter_ranges=chromosome_parameter_ranges,
        n_jobs=8,
    )

    genetic_algo_regressor.fit(X, y, n_iter=50, x0=x0)


    # comparison identified data to training data 
    best_fit_params = genetic_algo_regressor._elite_chromosome

    best_chromosome_motor_params = genetic_algo_regressor._gen_chromosome_dict(
        best_fit_params
    )

    print(f"{best_chromosome_motor_params=}")

    motor_fit = Motor(dt=motor_dt, **best_chromosome_motor_params)
    motor_pm2i_fit = motor_fit.generate_process_model_to_integrate()

    sol_t_fit, sol_u_fit, _, sol_y_fit = motor_pm2i_fit.integrate(
        compute_u_from_t=input_gen_train.value_at_t, dt_data=0.01, t_end=2, x0=x0
    )

    v_fit = sol_u_fit[0, :]
    tau_d_fit = sol_u_fit[1, :]

    i_m_fit = sol_y_fit[0, :]
    omega_fit = sol_y_fit[1, :]


    ax[0][1].plot(
        sol_t_fit, i_m_fit, label=r"Identified Model Response", color="C1"
    )


    ax[1][1].plot(
        sol_t_fit, omega_fit, color="C1"
    )


    ax[0][1].legend(loc="upper right")
    for a in np.ravel(ax):
        a.grid(linestyle="--")
    

    # validation

    v_sg = sg.SquareGenerator(period=0.5, pulse_width=0.4, amplitude=1)

    tau_d_sg = sg.SineGenerator(frequency=0.4, amplitude=0.02, phase=0)

    input_gen_valid = sg.InputGenerator([v_sg, tau_d_sg])


    sol_t, sol_u, _, sol_y = motor_pm2i.integrate(
        compute_u_from_t=input_gen_valid.value_at_t, dt_data=0.01, t_end=2, x0=x0
    )

    fig2, ax2 = plt.subplots(2, 2)

    v = sol_u[0, :]
    tau_d = sol_u[1, :]

    i_m = sol_y[0, :]
    omega = sol_y[1, :]


    ax2[0][0].set_ylabel(r"$v(t)$ (V)")
    ax2[0][0].plot(sol_t, v, color="C0")


    ax2[1][0].set_xlabel(r"$t$ (s)")
    ax2[1][0].set_ylabel(r"$\tau_{u}(t)$ (Nm)")
    ax2[1][0].plot(sol_t, tau_d, color="C0")



    ax2[0][1].set_ylabel(r"$i_m(t)$ (A)")
    ax2[0][1].plot(sol_t, i_m, label=r"Simulated Motor Data", color="C0")
    

    ax2[1][1].set_xlabel(r"$t$ (s)")
    ax2[1][1].set_ylabel(r"$\omega(t)$ (rad/s)")
    ax2[1][1].plot(sol_t, omega, color="C0")


    sol_t_fit, sol_u_fit, _, sol_y_fit = motor_pm2i_fit.integrate(
        compute_u_from_t=input_gen_valid.value_at_t, dt_data=0.01, t_end=2, x0=x0
    )

    v_fit = sol_u_fit[0, :]
    tau_d_fit = sol_u_fit[1, :]

    i_m_fit = sol_y_fit[0, :]
    omega_fit = sol_y_fit[1, :]


    ax2[0][1].plot(
        sol_t_fit, i_m_fit, label=r"Identified Model Response", color="C1"
    )
 

    ax2[1][1].plot(
        sol_t_fit, omega_fit, color="C1"
    )


    ax2[0][1].legend(loc="upper right")
    for a in np.ravel(ax2):
        a.grid(linestyle="--")

    plt.show()

    original_params = np.array(list(motor_params.values()))


if __name__ == "__main__":
    # yappi.set_clock_type("cpu")
    # yappi.start()
    _main()

    # yappi.convert2pstats(yappi.get_func_stats()).dump_stats("stats/func_stats.prof")

    # yappi.convert2pstats(yappi.get_thread_stats()).dump_stats("stats/func_stats.prof")
