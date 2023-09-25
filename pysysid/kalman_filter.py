"""Constrained Extended Kalamn Filter (CEKF) system identification methods."""


from typing import Tuple

import numpy as np
import sklearn.base

from . import pm2i, util


class CEKF(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    def __init__(
        self,
        process_model: type[pm2i.ProcessModelGenerator] = None,
        physical_constraint_matrix_A: np.ndarray = None,
        physical_constraint_vector_b: np.ndarray = None,
        n_params: int = 1,
        n_jobs=None,
    ) -> None:
        """Instantiate :class:`CEKF`.

        The process model is for now assumed to be continuous.

        References
        ----------
        https://journals.sagepub.com/doi/full/10.1177/1475921720929434

        Parameters
        ----------
        process_model : type[pm2i.ProcessModelGenerator]
            pm2i.ProcessModelGenerator derived type that will be used to compute
            the state derivative (x_dot = f(x,u)) and output (y = g(x,u)) of the
            system  to be identified at time `t`.
            Should also define
            The last `n_params` states in the state vector of `process_model`
            should be states representing the physical parameters that need to
            be identified with the CEXP. Their derivative, as defined in the
            `compute_state_derivative` of the `ProcessModelGenerator` derived
            class of `process_model` should always be equal to zero.
        """
        self.process_model = process_model
        self.physical_constraint_matrix_A = physical_constraint_matrix_A
        self.physical_constraint_vector_b = physical_constraint_vector_b
        self.n_params = n_params
        self.n_jobs = n_jobs

    def _measurement_innovation(
        self, y_k: np.ndarray, t: float, x_hat_k_given_km1: np.ndarray, u_k: np.ndarray
    ) -> np.ndarray:
        y_hat_k_given_km1 = self.pm2i_.compute_output(t, x_hat_k_given_km1, u_k)
        r_k = y_k - y_hat_k_given_km1
        return r_k

    def _compute_output_derivatives_around(
        self,
        t: float,
        x_hat_k_given_km1: np.ndarray,
        u_k: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        H_x_k = self.pm2i_.compute_dg_dx(t, x_hat_k_given_km1, u_k)
        H_v_k = self.pm2i_.compute_dg_dv(t, x_hat_k_given_km1, u_k)

        return H_x_k, H_v_k

    def _innovation_covariance_matrix(
        self,
        E_x_k_given_km1: np.ndarray,
        H_x_k: np.ndarray,
        H_v_k: np.ndarray,
    ) -> np.ndarray:
        E_y_k = H_x_k @ E_x_k_given_km1 @ H_x_k.T + H_v_k @ self.E_v_ @ H_v_k.T

        return E_y_k

    def _kalman_gain(
        self, E_x_k_given_km1: np.ndarray, H_x_k: np.ndarray, E_y_k: np.ndarray
    ) -> np.ndarray:
        L_tilde_k = np.linalg.solve(E_y_k.T, H_x_k @ E_x_k_given_km1.T).T
        return L_tilde_k

    def _state_estimate_for_measurement(
        self, x_hat_k_given_km1: np.ndarray, L_tilde_k: np.ndarray, r_k: np.ndarray
    ) -> np.ndarray:
        x_tilde_k_given_k = x_hat_k_given_km1 + L_tilde_k @ r_k

        return x_tilde_k_given_k

    def _check_constraints(
        self,
        x_tilde_k_given_k: np.ndarray,
        L_tilde_k: np.ndarray,
        x_hat_k_given_km1: np.ndarray,
        r_k: np.ndarray,
        E_y_k: np.ndarray,
    ) -> np.ndarray:
        index_inactive_constraints = (
            self.physical_constraint_matrix_A @ x_tilde_k_given_k
            - self.physical_constraint_vector_b
            > 0
        )
        if np.all(index_inactive_constraints):
            L_k = L_tilde_k
            x_hat_k_given_k = x_tilde_k_given_k
            return L_k, x_hat_k_given_k

        index_active_constraints = np.logical_not(index_inactive_constraints)
        index_active_constraints = index_active_constraints.reshape(
            (index_active_constraints.shape[0])
        )

        A_a = self.physical_constraint_matrix_A[index_active_constraints, :]
        b_a = self.physical_constraint_vector_b[index_active_constraints, :]

        A_a_T = A_a.T

        temp1 = np.linalg.solve(E_y_k.T, r_k).T

        temp2 = np.linalg.solve(A_a @ A_a_T, A_a @ x_tilde_k_given_k - b_a)

        L_k = L_tilde_k - (1 / temp1 @ r_k) * A_a_T @ temp2 @ temp1

        x_hat_k_given_k = x_hat_k_given_km1 + L_k @ r_k

        return L_k, x_hat_k_given_k

    def _state_covariance_for_measurement(
        self,
        L_k: np.ndarray,
        H_x_k: np.ndarray,
        E_x_k_given_km1: np.ndarray,
        H_v_k: np.ndarray,
    ):
        size_I = L_k.shape[0]
        temp1 = np.eye(size_I) - L_k @ H_x_k
        temp2 = L_k @ H_v_k
        E_x_k_given_k = temp1 @ E_x_k_given_km1 @ temp1.T + temp2 @ self.E_v_ @ temp2.T

        return E_x_k_given_k

    def _measurement_update(
        self,
        y_k: np.ndarray,
        t: float,
        x_hat_k_given_km1: np.ndarray,
        u_k: np.ndarray,
        E_x_k_given_km1: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r_k = self._measurement_innovation(y_k, t, x_hat_k_given_km1, u_k)
        H_x_k, H_v_k = self._compute_output_derivatives_around(
            t, x_hat_k_given_km1, u_k
        )
        E_y_k = self._innovation_covariance_matrix(E_x_k_given_km1, H_x_k, H_v_k)
        L_tilde_k = self._kalman_gain(E_x_k_given_km1, H_x_k, E_y_k)
        x_tilde_k_given_k = self._state_estimate_for_measurement(
            x_hat_k_given_km1, L_tilde_k, r_k
        )
        L_k, x_hat_k_given_k = self._check_constraints(
            x_tilde_k_given_k, L_tilde_k, x_hat_k_given_km1, r_k, E_y_k
        )
        E_x_k_given_k = self._state_covariance_for_measurement(
            L_k, H_x_k, E_x_k_given_km1, H_v_k
        )

        return x_hat_k_given_k, E_x_k_given_k

    def _state_estimate_for_time_update(
        self, x_hat_k_given_k: np.ndarray, dt_data: float, t: float, u_k: np.ndarray
    ):
        # TODO: figure better method to do integral
        # can't use data at k+1 (y_kp1 and u_kp1) since y _kp1 does not give us the state
        integral = dt_data * self.pm2i_.compute_state_derivative(
            t, x_hat_k_given_k, u_k
        )
        x_hat_kp1_given_k = x_hat_k_given_k + integral

        return x_hat_kp1_given_k

    def _state_covariance_for_time_update(
        self,
        E_x_k_given_k: np.ndarray,
        dt_data: float,
        t: float,
        x_hat_k_given_k: np.ndarray,
        u_k: np.ndarray,
    ):
        # slight deviation from
        # https://journals.sagepub.com/doi/full/10.1177/1475921720929434
        # here, using van loan's method as shown in
        # https://ieeexplore.ieee.org/document/1101743
        # to discretize the linearized process model
        phi_x_k, _, E_w_d = pm2i.discretize_process_model_linearized_around_x(
            self.pm2i_, dt_data, t, x_hat_k_given_k, u_k, self.E_w_
        )

        E_kp1_given_k = phi_x_k @ E_x_k_given_k @ phi_x_k.T + E_w_d

        return E_kp1_given_k

    def _time_update(
        self,
        t: float,
        x_hat_k_given_k: np.ndarray,
        u_k: np.ndarray,
        E_x_k_given_k: np.ndarray,
        dt_data: float,
    ):
        x_hat_kp1_given_k = self._state_estimate_for_time_update(
            x_hat_k_given_k, dt_data, t, u_k
        )

        E_kp1_given_k = self._state_covariance_for_time_update(
            E_x_k_given_k,
            dt_data,
            t,
            x_hat_k_given_k,
            u_k,
        )
        return x_hat_kp1_given_k, E_kp1_given_k

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        x0: np.ndarray = None,
        E_x_0: np.ndarray = None,
        E_w: np.ndarray = None,
        E_v: np.ndarray = None,
    ) -> "CEKF":
        # TODO: check that process model includes process model and measurement
        # model derivates (ex. df_df, dg_dx, etc)
        # TODO: check that both constraint A and b are defined if one of the two
        # is defined
        pmg = self.process_model()
        self.pm2i_: pm2i.ProcessModelToIntegrate = (
            pmg.generate_process_model_to_integrate()
        )
        self.E_v_ = E_v
        self.E_w_ = E_w

        X_t, X_u = util.split_time_input(X)
        dt_data = X_t[1] - X_t[0]

        # not same convetnion for learning algorithms and systems engineering
        T = X_t.T
        U = X_u.T
        Y = y.T

        del X_t, X_u

        if x0 is None:
            x0 = np.zeros((self.pm2i_.nbr_states, 1))
        else:
            if len(x0) != self.pm2i_.nbr_states:
                raise ValueError(
                    "x0 must have same number of values as the number of states in process_model"
                )
            x0 = x0.reshape((self.pm2i_.nbr_states, 1))

        x_hat_k_given_km1 = x0
        E_x_k_given_km1 = E_x_0

        self.x_arr_ = np.zeros(((self.pm2i_.nbr_states, len(T))))
        last_print_at = 0.0
        for i, t in enumerate(T):
            if t - last_print_at >= 1:
                print(f"Estimating state at timestep {t}...")
                last_print_at = t
            u_k = U[:, i].reshape((self.pm2i_.nbr_inputs, 1))
            y_k = Y[:, i].reshape((self.pm2i_.nbr_outputs, 1))

            x_hat_k_given_k, E_x_k_given_k = self._time_update(
                t, x_hat_k_given_km1, u_k, E_x_k_given_km1, dt_data
            )
            x_hat_kp1_given_k, E_kp1_given_k = self._measurement_update(
                y_k, t, x_hat_k_given_k, u_k, E_x_k_given_k
            )

            self.x_arr_[:, i] = x_hat_kp1_given_k.reshape((self.pm2i_.nbr_states,))

            x_hat_k_given_km1 = x_hat_kp1_given_k
            E_x_k_given_km1 = E_kp1_given_k

        self.optimal_parameters_ = x_hat_k_given_km1[-self.n_params :, 0]
