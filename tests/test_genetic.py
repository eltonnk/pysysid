from typing import Callable, Dict

import matplotlib.pyplot as plt
import numpy as np
import pytest

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
                [0, 0, 1],
                [0, -self.R / self.L, -self.K / self.L],
                [0, self.K / self.J, -self.B / self.J],
            ]
        )

        self.mat_B = np.array(
            [
                [0, 0],
                [1 / self.L, 0],
                [0, 1 / self.J],
            ]
        )

        # full state feedback for ease of debugging
        self.mat_C = np.eye(3, 3)

        self.mat_D = np.zeros((3, 2))

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

    def generate_process_model_to_integrate(self) -> pm2i.ProcessModelToIntegrate:
        return pm2i.ProcessModelToIntegrate(
            nbr_states=3,
            nbr_inputs=2,
            nbr_outputs=3,
            fct_for_x_dot=self.compute_state_derivative,
            fct_for_y=self.compute_output,
        )


def motor_inequality_constraint(chromosome: np.ndarray) -> np.ndarray:
    # all params should be positive. thus if chromosome is x, x_i > 0.
    # since ineqaulity constraint should be of the form h(x) < 0, we have

    return -1.0 * chromosome


@pytest.mark.parametrize(
    "og_params, input_gen, pmg_type, dt_data, t_end, range_var, inequality_constraint, n_chromosomes, replace_with_best_ratio, seed, n_iter",
    [
        (
            {
                "R": 1.0,
                "L": 6e-4,
                "J": 8e-7,
                "B": 1.33e-5,
                "K": 2.38e-2,
            },
            sg.InputGenerator(
                [
                    sg.SquareGenerator(period=1, pulse_width=0.5, amplitude=1),
                    sg.SineGenerator(frequency=0.3, amplitude=0.01, phase=0),
                ]
            ),
            Motor,
            0.01,
            2,
            0.99,
            motor_inequality_constraint,
            100,
            0.01,
            2,
            50,
        )
    ],
)
class TestGenetic:
    """Used to test the Genetic class from genetic.py."""

    def test_fit(
        self,
        og_params: Dict[str, float],
        input_gen: sg.InputGenerator,
        pmg_type: type[pm2i.ProcessModelGenerator],
        dt_data: float,
        t_end: float,
        range_var: float,
        inequality_constraint: Callable[[np.ndarray], np.ndarray],
        n_chromosomes: int,
        replace_with_best_ratio: float,
        seed: int,
        n_iter: int,
    ):
        """Used to test the fit method from the Genetic class."""

        # None if continous, float if discrete
        motor_dt = None
        x0 = None

        system_pmg = pmg_type(dt=motor_dt, **og_params)

        system_pm2i = system_pmg.generate_process_model_to_integrate()

        sol_t, sol_u, _, sol_y = system_pm2i.integrate(
            compute_u_from_t=input_gen.value_at_t, dt_data=dt_data, t_end=t_end, x0=x0
        )

        X = np.block([[sol_t.reshape(1, len(sol_t)).T, sol_u.T]])

        y = sol_y.T

        chromosome_parameter_ranges = {}

        for key, value in og_params.items():
            chromosome_parameter_ranges[key] = (
                (1 - range_var) * value,
                (1 + range_var) * value,
            )

        genetic_algo_regressor = genetic.Genetic(
            process_model=pmg_type,
            inequality_constraint=inequality_constraint,
            dt=motor_dt,
            compute_u_from_t=input_gen.value_at_t,
            n_chromosomes=n_chromosomes,
            replace_with_best_ratio=replace_with_best_ratio,
            can_terminate_after_index=10,
            ratio_max_error_for_termination=0.2,
            seed=seed,
            chromosome_parameter_ranges=chromosome_parameter_ranges,
        )

        genetic_algo_regressor.fit(X, y, n_iter=n_iter, x0=x0)

        best_fit_params = genetic_algo_regressor._elite_chromosome

        original_params = np.array(list(og_params.values()))

        np.testing.assert_allclose(best_fit_params, original_params, rtol=0, atol=0.02)
