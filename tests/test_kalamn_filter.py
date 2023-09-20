from typing import Dict

import numpy as np

import pysysid.pm2i as pm2i


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
        return np.eye(3, 3)

    def _compute_M(self):
        return np.eye(3, 3)

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

        self.mat_A = self._compute_A(
            l21=-self.R / self.L,
            l22=-self.self.K / self.L,
            l31=self.K / self.J,
            l32=-self.B / self.J,
        )

        self.mat_B = self._compute_B(l23=1 / self.L, l33=1 / self.J)
        self.mat_L = self.mat_B
        self.mat_C = self._compute_C()
        self.mat_M = self._compute_M()

    def compute_state_derivative(
        self,
        t: float,
        total_state: np.ndarray,
        total_input: np.ndarray,
    ) -> np.ndarray:
        u = total_input[:2, :]
        w = total_input[2:, :]

        x_dot = self.mat_A @ total_state + self.mat_B @ u + self.mat_L @ w

        return x_dot

    def compute_output(
        self,
        t: float,
        total_state: np.ndarray,
        total_input: np.ndarray,
    ) -> np.ndarray:
        u = total_input[:2, :]
        v = total_input[2:, :]
        return self.mat_C @ total_state + self.mat_M @ v

    def generate_process_model_to_integrate(self) -> pm2i.ProcessModelToIntegrate:
        return pm2i.ProcessModelToIntegrate(
            nbr_states=3,
            nbr_inputs=2,
            nbr_outputs=3,
            fct_for_x_dot=self.compute_state_derivative,
            fct_for_y=self.compute_output,
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

        l31 = l[4]
        l32 = l[5]
        l33 = l[6]

        return np.vstack(
            (
                self._compute_A(l21, l22, l31, l32) @ x
                + self._compute_B(l23, l33) @ total_input,
                np.zeros(6, 1),
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

        theta = total_state[0, 0]
        i = total_state[1, 0]
        omega = total_state[2, 0]

        v = total_input[0, 0]
        tau_d = total_state[1, 0]

        l21 = l[0]
        l22 = l[1]
        l23 = l[2]

        l31 = l[4]
        l32 = l[5]
        l33 = l[6]

        M_kal = np.array(
            [[0, 0, 0, 0, 0, 0], [i, omega, v, 0, 0, 0], [0, 0, 0, i, omega, tau_d]]
        )

        return np.block(
            [[self._compute_A(l21, l22, l31, l32), M_kal], [np.zeros(6, 9)]]
        )

    def compute_df_du(
        self, t: float, total_state: np.ndarray, total_input: np.ndarray
    ) -> np.ndarray:
        l = self.extract_l(total_state)

        l21 = l[0]
        l22 = l[1]
        l23 = l[2]

        l31 = l[4]
        l32 = l[5]
        l33 = l[6]
        return np.vstack((self._compute_B(l23, l33), np.zeros(6, 2)))

    def compute_df_dw(
        self, t: float, total_state: np.ndarray, total_input: np.ndarray
    ) -> np.ndarray:
        return self.compute_df_du(t, total_state, total_input)

    def compute_dg_dx(
        self, t: float, total_state: np.ndarray, total_input: np.ndarray
    ) -> np.ndarray:
        return np.hstack((self._compute_C(), np.zeros(3, 6)))

    def compute_dg_dv(
        self, t: float, total_state: np.ndarray, total_input: np.ndarray
    ) -> np.ndarray:
        return self._compute_M()
