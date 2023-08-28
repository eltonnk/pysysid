from typing import Callable, Dict

import numpy as np
import scipy.integrate as integrate


class ProcessModelToIntegrate:  # or, when shortened, pm2i
    nbr_states: int
    nbr_inputs: int
    nbr_outputs: int
    fct_for_x_dot: Callable[[float, np.ndarray, np.ndarray], np.ndarray]
    fct_for_y: Callable[[float, np.ndarray, np.ndarray], np.ndarray]

    def __init__(
        self, 
        nbr_states: int,
        nbr_inputs: int,
        nbr_outputs: int,
        fct_for_x_dot: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
        fct_for_y: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
    ):
        self.nbr_states: int = nbr_states
        self.nbr_inputs: int = nbr_inputs
        self.nbr_outputs: int = nbr_outputs
        self.fct_for_x_dot: Callable[[float, np.ndarray, np.ndarray], np.ndarray] = fct_for_x_dot
        self.fct_for_y: Callable[[float, np.ndarray, np.ndarray], np.ndarray] = fct_for_y

        self.ran_checks_for_compute_state_derivative: bool = False
        self.ran_checks_for_compute_output: bool = False

    def _check_valid_state(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] != self.nbr_states:
            raise ValueError(
                f"Did not provide correct number of states. System has {self.nbr_states} states, {x.shape} states were provided."
            )
        return np.reshape(x, (self.nbr_states, 1))

    def _check_valid_input(self, u: np.ndarray):
        if u.shape != (self.nbr_inputs, 1):
            raise ValueError(
                f"Did not provide correct number of inputs. System has {self.nbr_inputs} inputs, {u.shape} inputs were provided."
            )

    def _check_valid_output(self, y: np.ndarray):
        if y.shape != (self.nbr_outputs, 1):
            raise ValueError(
                f"Did not produce correct number of outputs. System has {self.nbr_outputs} outputs, {y.shape} outputs were provided."
            )

    def compute_state_derivative(self, t:float,  x: np.ndarray, u: np.ndarray):
        x = self._check_valid_state(x)
        if not self.ran_checks_for_compute_state_derivative:
            self._check_valid_input(u)
        x_dot = self.fct_for_x_dot(t, x, u)

        if not self.ran_checks_for_compute_state_derivative:
            self._check_valid_state(x_dot)
            self.ran_checks_for_compute_state_derivative = True
        return x_dot

    def compute_output(self, t:float, x: np.ndarray, u: np.ndarray):
        x = self._check_valid_state(x)
        if not self.ran_checks_for_compute_output:
            self._check_valid_input(u)

        y = self.fct_for_y(t, x, u)
        if not self.ran_checks_for_compute_output:
            self._check_valid_output(y)
            self.ran_checks_for_compute_output = True
        return y

    def generate_fction_to_integrate(
        self,
        compute_u_from_t: Callable[[float], np.ndarray],
    ):
        def fction_to_integrate(t: float, x: np.ndarray):
            u = compute_u_from_t(t)
            x_dot = self.compute_state_derivative(t, x, u)
            return x_dot.ravel()

        return fction_to_integrate

    def recreate_input_and_output(
        self,
        compute_u_from_t: Callable[[float], np.ndarray],
        t_arr: np.ndarray,
        x_arr: np.ndarray,
    ):
        u_arr = np.zeros(shape=(self.nbr_inputs, t_arr.shape[0]))
        y_arr = np.zeros(shape=(self.nbr_outputs, t_arr.shape[0]))

        for i, t in enumerate(t_arr):
            u = compute_u_from_t(t)
            u_arr[:, i] = u.ravel()
            y_arr[:, i] = self.compute_output(t, x_arr[:, i], u).ravel()

        return u_arr, y_arr

    def integrate(
        self,
        compute_u_from_t: Callable[[float], np.ndarray],
        dt_data: float,
        t_start: float = 0.0,
        t_end: float = 10.0,
        x0: np.ndarray = None,
    ):
        fction_to_integrate = self.generate_fction_to_integrate(compute_u_from_t)

        if x0 is None:
            x0 = np.zeros((self.nbr_states, 1)).ravel()

        t = np.arange(t_start, t_end, dt_data)
        # Find time-domain response by integrating the ODE
        sol = integrate.solve_ivp(
            fun=fction_to_integrate,
            t_span=(t_start, t_end),
            y0=x0,
            t_eval=t,
            rtol=1e-6,
            atol=1e-6,
            method="RK45",
        )

        sol_x = sol.y
        sol_t = sol.t

        sol_u, sol_y = self.recreate_input_and_output(compute_u_from_t, sol_t, sol_x)

        return sol_t, sol_u, sol_x, sol_y

class ProcessModelGenerator:
    def __init__(self, dt:float=None, **kwargs):
        self.dt = dt
        self.params: Dict[str, float] = kwargs

    def compute_state_derivative(
        self, t:float, total_state: np.ndarray, total_input: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError("Derived class should override this method")

    def compute_output(
        self, t:float, total_state: np.ndarray, total_input: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError("Derived class should override this method")

    def generate_process_model_to_integrate(
        self,
    ) -> ProcessModelToIntegrate:
        raise NotImplementedError("Derived class should override this method")
