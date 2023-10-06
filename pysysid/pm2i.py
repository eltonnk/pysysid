"""Module used by regressors to simulate identified systems.
"""
from typing import Callable, Dict, Tuple

import numpy as np
import scipy
from numpy.random import default_rng


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
    nbr_process_noise_inputs: int, optional
        Used to verify dimensions of E_w amd df_dw
    nbr_measurement_noise_inputs: int, optional
        Used to verify dimensions of E_v and dg_dv
    E_w : np.ndarrray, optional
        Square covariance matrix of the zero mean process noise.
        Process noise will be added when computing state derivatives with
        `NoisyProcessModelToIntegrate`.
        The last x inputs in the input vector `total_input` when calling
        `compute_state_derivative` will be considered noise
        inputs if E_w is not None, with x being the number of rows in the E_w
        matrix.
    E_v : np.ndarray, optional
        Square covariance matrix of the zero mean measurement noise.
        Measurement noise will be added when computing outputs with
        `NoisyProcessModelToIntegrate`.
        The last x inputs in the input vector `total_input` when calling
        `compute_output` will be considered noise
        inputs if E_v is not None, with x being the number of rows in the E_v
        matrix.
    rng : np.random.Generator
        Random number generator. Should be provided when E_v or E_w is not None,
        to generate the process and measurements noises. If not provided,
        this class will create its own random number generator. This makes it so
        the user can't provide a seed to the random number generator and thus
        the results when integrating states and outputs will not be repeatable.
    """

    def __init__(
        self,
        nbr_states: int,
        nbr_inputs: int,
        nbr_outputs: int,
        fct_for_x_dot: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
        fct_for_y: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
        df_dx: Callable[[float, np.ndarray, np.ndarray], np.ndarray] = None,
        df_du: Callable[[float, np.ndarray, np.ndarray], np.ndarray] = None,
        df_dw: Callable[[float, np.ndarray, np.ndarray], np.ndarray] = None,
        dg_dx: Callable[[float, np.ndarray, np.ndarray], np.ndarray] = None,
        dg_dv: Callable[[float, np.ndarray, np.ndarray], np.ndarray] = None,
        nbr_process_noise_inputs: int = 0,
        nbr_measurement_noise_inputs: int = 0,
        E_w: np.ndarray = None,
        E_v: np.ndarray = None,
        rng: np.random.Generator = None,
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

        self.df_dx = df_dx
        self.df_dw = df_dw
        self.df_du = df_du
        self.dg_dx = dg_dx
        self.dg_dv = dg_dv

        self.nbr_process_noise_inputs = nbr_process_noise_inputs
        self.nbr_measurement_noise_inputs = nbr_measurement_noise_inputs

        self.E_w = None
        if E_w is not None:
            self.E_w = self._sanity_check_covariance_matrix(
                E_w, self.nbr_process_noise_inputs
            )
            self.mean_w = np.zeros((self.nbr_process_noise_inputs))
        self.E_v = None
        if E_v is not None:
            self.E_v = self._sanity_check_covariance_matrix(
                E_v, self.nbr_measurement_noise_inputs
            )
            self.mean_v = np.zeros((self.nbr_measurement_noise_inputs))

        self.rng = None
        if E_w is not None or E_v is not None:
            if rng:
                self.rng = rng
            else:
                self.rng = default_rng()

        self.ran_checks_for_compute_state_derivative: bool = False
        self.ran_checks_for_compute_output: bool = False
        self.ran_checks_df_dx: bool = False
        self.ran_checks_df_du: bool = False
        self.ran_checks_df_dw: bool = False
        self.ran_checks_dg_dx: bool = False
        self.ran_checks_dg_dv: bool = False

    def _sanity_check_covariance_matrix(
        self, cov_matrix: np.ndarray, size: int
    ) -> np.ndarray:
        shape = cov_matrix.shape
        dimension = len(shape)
        if dimension != 2:
            raise ValueError(
                f"""Covariance matrix should be a 2-dimensional 
                array. Current dimension is {dimension}"""
            )
        rows = shape[0]
        cols = shape[1]
        if rows != cols:
            raise ValueError(
                f"""Covariance matrix is not square. Number of 
                rows: {rows}. Number of columns: {cols}."""
            )

        if size != rows:
            raise ValueError(
                f"""Covariance matrix size does not match dimension of noise input:
                Matrix size: ({rows, cols}). Noise input size: ({size}, 1)."""
            )

        return cov_matrix

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

    def _check_valid_matrix(self, mat: np.ndarray, n_rows: int, n_cols: int, name: str):
        shape = mat.shape
        if shape != (n_rows, n_cols):
            raise ValueError(
                f"""Did not produce correct shape for matrix {name}. Shape 
                should be ({n_rows, n_cols}), but is actually {shape}."""
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

        if self.E_w is not None:
            process_noise = self._sample_process_noise(t)
            u = np.vstack((u, process_noise.reshape(self.nbr_process_noise_inputs, 1)))

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

        if self.E_v is not None:
            measurement_noise = self.rng.multivariate_normal(self.mean_v, self.E_v)
            u = np.vstack(
                (u, measurement_noise.reshape(self.nbr_measurement_noise_inputs, 1))
            )

        y = self.fct_for_y(t, x, u)
        if not self.ran_checks_for_compute_output:
            self._check_valid_output(y)
            self.ran_checks_for_compute_output = True
        return y

    def compute_df_dx(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        x = self._check_valid_state(x)
        if not self.ran_checks_df_dx:
            self._check_valid_input(u)

        df_dx = self.df_dx(t, x, u)
        if not self.ran_checks_df_dx:
            self._check_valid_matrix(df_dx, self.nbr_states, self.nbr_states, "df_dx")
            self.ran_checks_df_dx = True

        return df_dx

    def compute_df_du(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        x = self._check_valid_state(x)
        if not self.ran_checks_df_du:
            self._check_valid_input(u)

        df_du = self.df_du(t, x, u)
        if not self.ran_checks_df_du:
            self._check_valid_matrix(df_du, self.nbr_states, self.nbr_inputs, "df_du")
            self.ran_checks_df_du = True

        return df_du

    def compute_df_dw(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        x = self._check_valid_state(x)
        if not self.ran_checks_df_dw:
            self._check_valid_input(u)

        df_dw = self.df_dw(t, x, u)
        if not self.ran_checks_df_dw:
            self._check_valid_matrix(
                df_dw, self.nbr_states, self.nbr_process_noise_inputs, "df_dw"
            )
            self.ran_checks_df_dw = True

        return df_dw

    def compute_dg_dx(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        x = self._check_valid_state(x)
        if not self.ran_checks_dg_dx:
            self._check_valid_input(u)

        dg_dx = self.dg_dx(t, x, u)
        if not self.ran_checks_dg_dx:
            self._check_valid_matrix(dg_dx, self.nbr_outputs, self.nbr_states, "dg_dx")
            self.ran_checks_dg_dx = True

        return dg_dx

    def compute_dg_dv(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        x = self._check_valid_state(x)
        if not self.ran_checks_dg_dv:
            self._check_valid_input(u)

        dg_dv = self.dg_dv(t, x, u)
        if not self.ran_checks_dg_dv:
            self._check_valid_matrix(
                dg_dv, self.nbr_outputs, self.nbr_measurement_noise_inputs, "dg_dv"
            )
            self.ran_checks_dg_dv = True

        return dg_dv

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

    def _generate_sampled_process_noise(self, dt_data: float, t_arr: np.ndarray):
        nbr_samples = len(t_arr)
        self.process_noise_arr = self.rng.multivariate_normal(
            self.mean_w, self.E_w, size=nbr_samples
        )
        self.process_noise_arr = self.process_noise_arr.T

        def _sample_process_noise(t: float):
            index = int(t / dt_data)
            try:
                process_noise = self.process_noise_arr[:, index]
            except IndexError as e:
                process_noise = self.process_noise_arr[:, -1]

            return process_noise

        self._sample_process_noise = _sample_process_noise

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
        t = np.arange(t_start, t_end, dt_data)

        if self.rng:
            self._generate_sampled_process_noise(dt_data, t)

        fction_to_integrate = self.generate_fction_to_integrate(compute_u_from_t)

        if x0 is None:
            x0 = np.zeros((self.nbr_states, 1)).ravel()

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


def discretize_process_model_linearized_around_x(
    pm2i: ProcessModelToIntegrate,
    dt_data: float,
    t: float,
    x: np.ndarray,
    u: np.ndarray,
    E_w_c: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    A_c = pm2i.df_dx(t, x, u)
    B_c = pm2i.df_du(t, x, u)
    L_c = pm2i.df_dw(t, x, u)

    Q_c = E_w_c

    xdim = pm2i.nbr_states
    udim = pm2i.nbr_inputs

    zero_state_state = np.zeros((xdim, xdim))
    zero_state_input = np.zeros((xdim, udim))
    zero_input_state = zero_state_input.T
    zero_input_input = np.zeros((udim, udim))

    LQLT = L_c @ Q_c @ L_c.T

    Theta = np.block(
        [
            [A_c, LQLT, zero_state_state, zero_state_input],
            [zero_state_state, -A_c.T, zero_state_state, zero_state_input],
            [zero_state_state, zero_state_state, A_c, B_c],
            [zero_input_state, zero_input_state, zero_input_state, zero_input_input],
        ]
    )

    condTheta = np.linalg.cond(Theta)
    normTheta = np.linalg.norm(Theta)
    # normInvTheta = np.linalg.norm(np.linalg.inv(Theta)) # inverse of A doesn't exist

    Psi = scipy.linalg.expm(Theta * dt_data)

    A_d = Psi[0:xdim, 0:xdim]
    B_d = Psi[2 * xdim : 3 * xdim, 3 * xdim : 3 * xdim + udim]
    Q_d = Psi[0:xdim, xdim : 2 * xdim] @ A_d.T

    return A_d, B_d, Q_d


class ProcessModelGenerator:
    """Abstract class that represents a system's process model, and uses this
    specification to produce the object, an instance of
    `ProcessModelToIntegrate`, which is used to simulate the response of this
    system to a given input signal, and is also used to verify correct state,
    input and output dimensions.

    Parameters
    ----------
    dt : float, optional
        If None, process model is supposed to be continuous.
        Else, specifies the sampling time for a discrete process model
    E_w : np.ndarrray, optional
        Square covariance matrix of the zero mean process noise.
        Process noise will be added when computing state derivatives with
        `NoisyProcessModelToIntegrate`.
        The last x inputs in the input vector `total_input` when calling
        `compute_state_derivative` will be considered noise
        inputs if E_w is not None, with x being the number of rows in the E_w
        matrix.
    E_v : np.ndarray, optional
        Square covariance matrix of the zero mean measurement noise.
        Measurement noise will be added when computing outputs with
        `NoisyProcessModelToIntegrate`.
        The last x inputs in the input vector `total_input` when calling
        `compute_output` will be considered noise
        inputs if E_v is not None, with x being the number of rows in the E_v
        matrix.
    **kwargs : Dict[str, float]
        Process model parameters that define physical quantities within the
        system.
    """

    def __init__(
        self,
        dt: float = None,
        E_w: np.ndarray = None,
        E_v: np.ndarray = None,
        **kwargs: Dict[str, float],
    ):
        self.dt = dt
        self.E_w = E_w
        self.E_v = E_v
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
        total_state : np.ndarray
            The system's state at time `t`
        total_input : np.ndarray
            The system's input at time `t`

        Returns
        -------
        np.ndarray
            The system's output at time `t`
        """
        raise NotImplementedError("Derived class should override this method")

    @staticmethod
    def param_inequality_constraint(params: np.ndarray) -> np.ndarray:
        """Returns the value of inequality constraint function h(x).
        h(x) is such that the inequality h(x) <= 0 should be true
        for all possible values of x, with x being the array `params`.

        Parameters
        ----------
        params : np.ndarray
            Array of parameters. Should be in the same order as the parameters
            in the `kwargs` dict in the derived class constructor.

        Returns
        -------
        np.ndarray
            The constraint's function h(x)'s value given x.
        """
        raise NotImplementedError("Derived class should override this method")

    @staticmethod
    def param_equality_constraint(params: np.ndarray) -> np.ndarray:
        """Returns the value of inequality constraint function g(x).
        g(x) is such that the inequality g(x) = 0 should be true
        for all possible values of x, with x being the array `params`.

        Parameters
        ----------
        params : np.ndarray
            Array of parameters. Should be in the same order as the parameters
            in the `kwargs` dict in the derived class constructor.

        Returns
        -------
        np.ndarray
            The constraint function g(x)'s value given x.

        """
        raise NotImplementedError("Derived class should override this method")

    def generate_process_model_to_integrate(
        self,
        rng: np.random.Generator = None,
    ) -> ProcessModelToIntegrate:
        """This method should pass the `compute_state_derivative` and
        `compute_output` methods to the `ProcessModelToIntegrate` constructor,
        as well as the appropriate input, output and state dimensions.

        Parameters
        ----------
        rng : np.random.Generator
            Random number generator. Should be provided when E_v or E_w is not None,
            to generate the process and measurements noises. If not provided,
            this class will create its own random number generator. This makes it so
            the user can't provide a seed to the random number generator and thus
            the results when integrating states and outputs will not be repeatable.


        Returns
        -------
        ProcessModelToIntegrate
            Object used to simulate the response of the process model
            defined by this instance of the `ProcessModelGenerator` class to a
            given output signal.

        """
        raise NotImplementedError("Derived class should override this method")

    def compute_df_dx(
        self, t: float, total_state: np.ndarray, total_input: np.ndarray
    ) -> np.ndarray:
        """Derivate of the process vector function f (such that x_dot = f(x,u,w), with
        x the state, u the input and w the process noise) relative to the state vector.

        Parameters
        ----------
        t : float
            Time at which the derivative should be computed.
        total_state : np.ndarray
            The state around which the derivative should be computed at time `t`
        total_input : np.ndarray
            The input around which the derivative should be computed at time `t`

        Returns
        -------
        np.ndarray
            The derivative at time `t`

        """

        raise NotImplementedError("Derived class should override this method")

    def compute_df_du(
        self, t: float, total_state: np.ndarray, total_input: np.ndarray
    ) -> np.ndarray:
        """Derivate of the process vector function f (such that x_dot = f(x,u,w), with
        x the state, u the input and w the process noise) relative to the input vector.

        Parameters
        ----------
        t : float
            Time at which the derivative should be computed.
        total_state : np.ndarray
            The state around which the derivative should be computed at time `t`
        total_input : np.ndarray
            The input around which the derivative should be computed at time `t`

        Returns
        -------
        np.ndarray
            The derivative at time `t`

        """

        raise NotImplementedError("Derived class should override this method")

    def compute_df_dw(
        self, t: float, total_state: np.ndarray, total_input: np.ndarray
    ) -> np.ndarray:
        """Derivate of the process vector function f (such that x_dot = f(x,u,w), with
        x the state, u the input and w the process noise) relative to the process noise vector.

        Parameters
        ----------
        t : float
            Time at which the derivative should be computed.
        total_state : np.ndarray
            The state around which the derivative should be computed at time `t`
        total_input : np.ndarray
            The input around which the derivative should be computed at time `t`

        Returns
        -------
        np.ndarray
            The derivative at time `t`

        """

        raise NotImplementedError("Derived class should override this method")

    def compute_dg_dx(
        self, t: float, total_state: np.ndarray, total_input: np.ndarray
    ) -> np.ndarray:
        """Derivate of the output vector function g (such that y = g(x,u,v), with
        x the state, u the input and v the measurment noise) relative to the state vector.

        Parameters
        ----------
        t : float
            Time at which the derivative should be computed.
        total_state : np.ndarray
            The state around which the derivative should be computed at time `t`
        total_input : np.ndarray
            The input around which the derivative should be computed at time `t`

        Returns
        -------
        np.ndarray
            The derivative at time `t`

        """
        raise NotImplementedError("Derived class should override this method")

    def compute_dg_dv(
        self, t: float, total_state: np.ndarray, total_input: np.ndarray
    ) -> np.ndarray:
        """Derivate of the output vector function g (such that y = g(x,u,v), with
        x the state, u the input and v the measurment noise) relative to the measurement noise vector.

        Parameters
        ----------
        t : float
            Time at which the derivative should be computed.
        total_state : np.ndarray
            The state around which the derivative should be computed at time `t`
        total_input : np.ndarray
            The input around which the derivative should be computed at time `t`

        Returns
        -------
        np.ndarray
            The derivative at time `t`

        """
        raise NotImplementedError("Derived class should override this method")
