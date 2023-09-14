"""Constrained Extended Kalamn Filter (CEKF) system identification methods."""


from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import sklearn.base
from numpy.random import default_rng

from . import pm2i, util


class CEKF(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    def __init__(
        self,
        process_model: type[pm2i.ProcessModelGenerator] = None,
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
            pm2i.ProcessModelGenerator derived type that will be used, when its
            constructor is called with a set of identified paramaters,
            to simulate the response of the identified system.
        """
        self.process_model = process_model
        self.n_params = n_params
        self.n_jobs = n_jobs

    def _measurement_innovation(
        self, y_k: np.ndarray, t: float, x_hat_k_given_km1: np.ndarray, u_k: np.ndarray
    ) -> np.ndarray:
        y_hat_k_given_km1 = self.pm2i_.compute_output(t, x_hat_k_given_km1, u_k)
        r_k = y_k - y_hat_k_given_km1
        return r_k

    def _innovation_covariance_matrix(
        self,
        E_x_k_given_km1: np.ndarray,
        t: float,
        x_hat_k_given_km1: np.ndarray,
        u_k: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        H_x_k = self.pm2i_.compute_dg_dx(t, x_hat_k_given_km1, u_k)
        H_v_k = self.pm2i_.compute_dg_dv(t, x_hat_k_given_km1, u_k)

        E_y_k = H_x_k @ E_x_k_given_km1 @ H_x_k.T + H_v_k @ self.E_v_ @ H_v_k.T

        return E_y_k, H_x_k

    def _kalman_gain(
        E_x_k_given_km1: np.ndarray, H_x_k: np.ndarray, E_y_k: np.ndarray
    ) -> np.ndarray:
        L_tilde_k = np.linalg.solve(E_y_k.T, H_x_k @ E_x_k_given_km1.T).T
        return L_tilde_k

    def _state_estimate_for_measurement(
        x_hat_k_given_km1: np.ndarray, L_tilde_k: np.ndarray, r_k: np.ndarray
    ) -> np.ndarray:
        x_tilde_k_given_k = x_hat_k_given_km1 + L_tilde_k @ r_k

        return x_tilde_k_given_k

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_iter: int = 0,
        x0: np.ndarray = None,
        E_w_0: np.ndarray = None,
        E_v: np.ndarray = None,
    ) -> "CEKF":
        # TODO: check that process model has noise matrices
        self.pm2i_: pm2i.ProcessModelToIntegrate = (
            self.process_model.generate_process_model_to_integrate()
        )
        self.E_v_ = E_v

        X_t, X_u = util.split_time_input(X)
        self.dt_data_ = X_t[1] - X_t[0]
