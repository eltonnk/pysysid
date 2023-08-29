"""Module used by regressors to simulate identified systems.
"""
from typing import Callable, Dict, Tuple

import numpy as np
import scipy


class ProcessModelToIntegrate:  # or, when shortened, pm2i
    """This class uses the specified process model, which is entirely
    describded by the `fct_for_x_dot` and `fct_for_y` functions, to simulate
    an identified system.

    Parameters
    ----------
    nbr_states : int
        Used to verify that the state variable given to fct_for_x_dot and
        fct_for_y is the right dimension. Also used to verify that the state
        derivative variable returned by fct_for_x_dot is the right dimension.
    nbr_inputs : int
        Used to verify that the input variable given to fct_for_x_dot and
        fct_for_y is the right dimension.
    nbr_outputs : int
        Used to verify that the output variable returned by fct_for_y is the
        right dimension.
    fct_for_x_dot : Callable[[float, np.ndarray, np.ndarray], np.ndarray]
        Computes the system's state derivative at time t given the state and
        input at that same time.
    fct_for_y : Callable[[float, np.ndarray, np.ndarray], np.ndarray]
        Computes the system's output at time t given the state and
        input at that same time.
    """

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
        self.fct_for_x_dot: Callable[
            [float, np.ndarray, np.ndarray], np.ndarray
        ] = fct_for_x_dot
        self.fct_for_y: Callable[
            [float, np.ndarray, np.ndarray], np.ndarray
        ] = fct_for_y

        self.ran_checks_for_compute_state_derivative: bool = False
        self.ran_checks_for_compute_output: bool = False

    def _check_valid_state(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] != self.nbr_states:
            raise ValueError(
                f"""Did not provide correct number of states. System has 
                {self.nbr_states} states, {x.shape} states were provided."""
            )
        return np.reshape(x, (self.nbr_states, 1))

    def _check_valid_input(self, u: np.ndarray):
        if u.shape != (self.nbr_inputs, 1):
            raise ValueError(
                f"""Did not provide correct number of inputs. System has 
                {self.nbr_inputs} inputs, {u.shape} inputs were provided."""
            )

    def _check_valid_output(self, y: np.ndarray):
        if y.shape != (self.nbr_outputs, 1):
            raise ValueError(
                f"""Did not produce correct number of outputs. System has 
                {self.nbr_outputs} outputs, {y.shape} outputs were provided."""
            )

    def compute_state_derivative(
        self,
        t: float,
        x: np.ndarray,
        u: np.ndarray,
    ) -> np.ndarray:
        """Checks that input, state and state derivate passed to and returned by
        `self.fct_for_x_dot` have the right dimensions. If so, returns
        the state derivative returned by `self.fct_for_x_dot`.

        Parameters
        ----------
        t : float
            Time at which the state derivative should be computed.
        x : np.ndarray
            The system's state at time `t`
        u : np.ndarray
            The system's input at time `t`

        Returns
        -------
        np.ndarray
            The system's state derivative at time `t`
        """
        x = self._check_valid_state(x)
        if not self.ran_checks_for_compute_state_derivative:
            self._check_valid_input(u)
        x_dot = self.fct_for_x_dot(t, x, u)

        if not self.ran_checks_for_compute_state_derivative:
            self._check_valid_state(x_dot)
            self.ran_checks_for_compute_state_derivative = True
        return x_dot

    def compute_output(
        self,
        t: float,
        x: np.ndarray,
        u: np.ndarray,
    ) -> np.ndarray:
        """Checks that input, state and output passed to and returned by
        `self.fct_for_y` have the right dimensions. If so, returns
        the output returned by `self.fct_for_y`.

        Parameters
        ----------
        t : float
            Time at which the output should be computed.
        x : np.ndarray
            The system's state at time `t`
        u : np.ndarray
            The system's input at time `t`

        Returns
        -------
        np.ndarray
            The system's output at time `t`
        """
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
    ) -> Callable[[float, np.ndarray], np.ndarray]:
        """Uses the `compute_u_from_t` function to generate a predetermined
        signal at time t, the system's input, which is passed to the system's
        process model function to compute the state derivative.

        This makes it so `scipy.integrate.solve_ivp` which can only compute
        states for an homogenous system ( x_dot + f(t,x) = 0 ), can now compute
        states fo a non-homogenous system ( x_dot + f(t,x) = u(t) ).

        Parameters
        ----------
        compute_u_from_t : Callable[[float], np.ndarray]
            Function used to compute the input signal at time t.

        Returns
        -------
        Callable[[float, np.ndarray], np.ndarray]
            Function that computes the input signal and then the state
            derivative.
        """

        def fction_to_integrate(t: float, x: np.ndarray) -> np.ndarray:
            u = compute_u_from_t(t)
            x_dot = self.compute_state_derivative(t, x, u)
            return x_dot.ravel()

        return fction_to_integrate

    def recreate_input_and_output(
        self,
        compute_u_from_t: Callable[[float], np.ndarray],
        t_arr: np.ndarray,
        x_arr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Uses states computed by `scipy.integrate.solve_ivp` to compute
        the system's output, which is a function of input and outputs.
        To do so, the argument `compute_u_from_t` should be the same
        function that was passed to `self.generate_fction_to_integrate`.

        Parameters
        ----------
        compute_u_from_t : Callable[[float], np.ndarray]
            Function used to compute the input signal at time t.
        t_arr : np.ndarray
            Timeseries with times (in seconds) at which input and output
            should be computed
        x_arr : np.ndarray
            States for each time value in the `t_arr` timeseries

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Input and Output timeseries
        """
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Simulates the system's response to a given input signal.

        Parameters
        ----------
        compute_u_from_t : Callable[[float], np.ndarray]
            Function used to compute the system's input signal at time t
        dt_data : float
            Interval of time at which the system's response should be evaluated
        t_start : float, optional
            Time at which to start the simulation, by default 0.0
        t_end : float, optional
            Time at which to end the simulation, by default 10.0
        x0 : np.ndarray, optional
            Values for each of the system's states at `t_start`, by default None

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Time, input, state and output timeseries produced by the simulation
        """
        fction_to_integrate = self.generate_fction_to_integrate(compute_u_from_t)

        if x0 is None:
            x0 = np.zeros((self.nbr_states, 1)).ravel()

        t = np.arange(t_start, t_end, dt_data)
        # Find time-domain response by integrating the ODE
        sol = scipy.integrate.solve_ivp(
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

        sol_u, sol_y = self.recreate_input_and_output(
            compute_u_from_t,
            sol_t,
            sol_x,
        )

        return sol_t, sol_u, sol_x, sol_y


class ProcessModelGenerator:
    """Abstract class that represents a system's process model, and uses this
    specification to produce the object, an instance of
    `ProcessModelToIntegrate`, which is used to simulate the response of this
    system to a given input signal, and is also used to verify correct state,
    input and output dimensions.

    Parameters
    ----------
    dt: float, optional
        If None, process model is supposed to be continuous.
        Else, specifies the sampling time for a discrete process model
    **kwargs: Dict[str, float]
        Process model parameters that define physical quantities within the
        system.
    """

    def __init__(self, dt: float = None, **kwargs: Dict[str, float]):
        self.dt = dt
        self.params: Dict[str, float] = kwargs

    def compute_state_derivative(
        self, t: float, total_state: np.ndarray, total_input: np.ndarray
    ) -> np.ndarray:
        """This method should define how the system's state derivative
        is computed.

        Parameters
        ----------
        t : float
            Time at which the state derivative should be computed.
        total_state : np.ndarray
            The system's state at time `t`
        total_input : np.ndarray
            The system's input at time `t`

        Returns
        -------
        np.ndarray
            The system's state derivative at time `t`

        """
        raise NotImplementedError("Derived class should override this method")

    def compute_output(
        self, t: float, total_state: np.ndarray, total_input: np.ndarray
    ) -> np.ndarray:
        """This method should define how the system's output
        is computed.

        Parameters
        ----------
        t : float
            Time at which the output should be computed.
        x : np.ndarray
            The system's state at time `t`
        u : np.ndarray
            The system's input at time `t`

        Returns
        -------
        np.ndarray
            The system's output at time `t`
        """
        raise NotImplementedError("Derived class should override this method")

    def generate_process_model_to_integrate(
        self,
    ) -> ProcessModelToIntegrate:
        """This method should pass the `compute_state_derivative` and
        `compute_output` methods to the `ProcessModelToIntegrate` constructor,
        as well as the appropriate input, output and state dimensions.

        Returns
        -------
        ProcessModelToIntegrate
            Object used to simulate the response of the process model
            defined by this instance of the `ProcessModelGenerator` class to a
            given output signal.

        """
        raise NotImplementedError("Derived class should override this method")
