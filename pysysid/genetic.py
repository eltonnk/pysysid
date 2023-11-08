"""Genetic algorithm system identification methods."""


from time import time
from typing import Any, Callable, Dict, List, Tuple, Union

import joblib
import numpy as np
import sklearn.base
from numpy.random import default_rng

from . import pm2i, util


class Genetic(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    """Genetic algorithm model.

    References
    ----------

    A. Dupuis, M. Ghribi and A. Kaddouri, "Multiobjective genetic estimation of
    DC motor parameters and load torque," 2004 IEEE International Conference on
    Industrial Technology, 2004. IEEE ICIT '04., Hammamet, Tunisia, 2004,
    pp. 1511-1514 Vol. 3, doi: 10.1109/ICIT.2004.1490788.
    (https://ieeexplore.ieee.org/document/1490788)

    N.M. Kwok, Q.P. Ha, M.T. Nguyen, J. Li, B. Samali, Bouc–Wen model parameter
    identification for a MR fluid damper using computationally efficient GA, ISA
    Transactions, Volume 46, Issue 2, 2007, Pages 167-179, ISSN 0019-0578,
    https://doi.org/10.1016/j.isatra.2006.08.005.
    (https://www.sciencedirect.com/science/article/pii/S0019057807000237)

    R. Farmani and J. A. Wright, "Self-adaptive fitness formulation for
    constrained optimization," in IEEE Transactions on Evolutionary Computation,
    vol. 7, no. 5, pp. 445-455, Oct. 2003, doi: 10.1109/TEVC.2003.817236.
    (https://ieeexplore.ieee.org/abstract/document/1237163)


    """

    # Array check parameters for :func:`fit` when ``X`` and ``y` are given
    _check_X_y_params: Dict[str, Any] = {
        "multi_output": True,
        "y_numeric": True,
    }

    # Array check parameters for :func:`predict` and :func:`fit` when only
    # ``X`` is given
    _check_array_params: Dict[str, Any] = {
        "dtype": "numeric",
    }

    def __init__(
        self,
        process_model: type[pm2i.ProcessModelGenerator] = None,
        equality_constraint_tolerance: float = 0.0001,
        dt: float = None,
        compute_u_from_t: Callable[[float], np.ndarray] = None,
        n_chromosomes: int = 2,
        replace_with_best_ratio: float = 0.01,
        can_terminate_after_index: int = 2,
        ratio_max_error_for_termination: float = 0.2,
        seed: int = None,
        chromosome_parameter_ranges: Dict[str, Tuple[float, float]] = None,
        integration_method: str = "RK45",
        integration_timeout: float = 1000.0,
        n_jobs=None,
    ) -> None:
        """Instantiate :class:`Genetic`.

        Parameters
        ----------
        process_model : type[pm2i.ProcessModelGenerator]
            pm2i.ProcessModelGenerator derived type that will be used, when its
            constructor is called with a set of identified paramaters,
            to simulate the response of the identified system.
        equality_constraint_tolerance : float
            Set of parametrs is considered valid according to equality constraint
            h(x) if |h(x)| <= equality_constraint_tolerance, since exact equality
            is almost impossible to attain in an optimization setting.
            The equality contraint should be defined in `process_model` as the
            `param_equality_constraint` method.
        dt: float, optional
            If None, process model is supposed to be continuous.
            Else, specifies the sampling time for a discrete process model
        compute_u_from_t: Callable[[float], np.ndarray], optional
            Function used to compute the system's input signal at time t.
            TODO : If None, the input signal at time t will be interpolated from
            the input array X when using `fit` and `predict`
        n_chromosomes: int
            Number of sets of system parameters that will be generated randomly
            when `fit` is called. Each of these sets of parameters will evolve
            at each iteration of the genetic algorithm in `fit`, so that they
            produce a simualted system response which is close to the provided
            data.
        seed: int
            TODO
        chromosome_parameter_ranges : Dict[str, Tuple[float, float]]
            TODO
        """
        self.process_model = process_model
        self._equality_constraint_tolerance = equality_constraint_tolerance
        self.dt = dt
        self.compute_u_from_t = compute_u_from_t
        self.n_chromosomes = n_chromosomes
        self.replace_with_best_ratio = replace_with_best_ratio
        self.can_terminate_after_index = can_terminate_after_index
        self.ratio_max_error_for_termination = ratio_max_error_for_termination
        self.seed = seed
        self.chromosome_parameter_ranges = chromosome_parameter_ranges
        self.n_jobs = n_jobs
        self.integration_method = integration_method
        self.integration_timeout = integration_timeout

    def _validate_chromosome_parameter_range(self):
        for param_name, param_range in self.chromosome_parameter_ranges.items():
            param_min, param_max = param_range
            if param_min > param_max:
                raise ValueError(
                    f"""Parameter '{param_name}' in default 
                    chromosome_parameter_ranges has bigger minimum value than 
                    maximum value. Plaese specify minimum parameter value, 
                    then maximum parameter value."""
                )

    def _validate_replacement_ratio(self):
        if self.replace_with_best_ratio < 0 or self.replace_with_best_ratio > 1:
            raise ValueError(
                f"""Constructor argument 'replace_with_best_ratio'
                must have a must have a value in the [0.0, 1.0] interval. 
                Current value is {self.replace_with_best_ratio}."""
            )
        # TODO: raise warning if the replacement ratio is over 20, 30 , 50% ?
        # (don't want to replace too many chromosomes by best one or else
        # convergence slows down)

    def _validate_ratio_max_error_for_termination(self):
        if (
            self.ratio_max_error_for_termination < 0
            or self.ratio_max_error_for_termination > 1
        ):
            raise ValueError(
                f"""Constructor argument 'ratio_max_error_for_termination'
                must have a must have a value in the [0.0, 1.0] interval. 
                Current value is {self.ratio_max_error_for_termination}."""
            )

        # TODO: raise warning if the ratio is over  50%, 70%, 90% ?
        # (dont want to stop if error has not gone down)

    def _validate_integration_method(self):
        list_integration_methods = ["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"]

        if self.integration_method not in list_integration_methods:
            raise ValueError(
                f"""Choose constructor parameter `integration_method` from this 
                list: {list_integration_methods}. See `scipy.integrate.solve_ivp` 
                documentation for info on these methods."""
            )

        if self.integration_method != "RK45" and self.integration_method != "Radau":
            raise NotImplementedError(
                """Only `integration_method`s implemented from 
                `scipy.integrate.solve_ivp` are `RK45` and `Radau`."""
            )

        list_jac_req_methods = ["Radau", "BDF", "LSODA"]
        has_jac_method = True

        fake_params_dict = {}

        for key in list(self.chromosome_parameter_ranges.keys()):
            fake_params_dict[key] = 1.0

        fake_pmg = self.process_model(**fake_params_dict)
        fake_pm2i = fake_pmg.generate_process_model_to_integrate()

        try:
            fake_pmg.compute_df_dx(
                0,
                np.zeros((fake_pm2i.nbr_states)),
                np.zeros((fake_pm2i.nbr_inputs)),
            )
        except NotImplementedError:
            has_jac_method = False

        if self.integration_method in list_jac_req_methods and not has_jac_method:
            raise ValueError(
                f"""Constructor paramater `process_model` must be a derived class
                of the `ProcessModelGenerator class, with the `compute_df_dx` 
                method derived and defined, since the chosen 
                `integration_method` is {self.integration_method}. See the 
                `scipy.integrate.solve_ivp` documentation for more info."""
            )

    def _initialize_replacement_step_variables(self):
        self._n_chromosomes_to_replace = int(
            self.n_chromosomes * self.replace_with_best_ratio
        )

    def _initialize_termination_checking_variables(self, n_iter: int):
        self._g_index = None
        self._elite_chromosome_error_list = np.zeros((n_iter))

    def _initialize_simulation_flags(self):
        self._chromosomes_to_be_simulated = np.full((self.n_chromosomes), True)

    def _initialize_constraints(self):
        fake_params_array = np.zeros(
            (len(list(self.chromosome_parameter_ranges.keys())))
        )

        self._inequality_constraint = None
        try:
            self.process_model.param_inequality_constraint(fake_params_array)
            self._inequality_constraint = self.process_model.param_inequality_constraint
        except NotImplementedError:
            self._inequality_constraint = None

        self._equality_constraint = None
        try:
            self.process_model.param_equality_constraint(fake_params_array)
            self._equality_constraint = self.process_model.param_equality_constraint
        except NotImplementedError:
            self._equality_constraint = None

    def _initialize_chromosomes(
        self, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        n_params = len(self.chromosome_parameter_ranges.keys())
        # Columns: chromosomes
        # Rows: parameters
        chromosomes: np.ndarray = np.zeros((n_params, self.n_chromosomes))

        # Initialize all parameters in all chromosomes randomly, according to approximate
        # range of values we think the parameters should fall in
        for i, param_range in enumerate(
            list(self.chromosome_parameter_ranges.values())
        ):
            param_min, param_max = param_range
            chromosomes[i, :] = rng.uniform(
                low=param_min, high=param_max, size=self.n_chromosomes
            )

        param_std_deviation = np.std(chromosomes, axis=1)
        return chromosomes, param_std_deviation

    def _gen_chromosome_dict(self, param_array: np.ndarray) -> Dict[str, float]:
        chromosome_dict = {}
        for param_name, param_value in zip(
            list(self.chromosome_parameter_ranges.keys()), param_array
        ):
            chromosome_dict[param_name] = param_value

        return chromosome_dict

    def _preprocessing_input_output_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float, int, np.ndarray]:
        X_t, X_u = util.split_time_input(X)
        dt_data = X_t[1] - X_t[0]

        n_outputs = y.shape[1]
        output_inverse_delta_list = np.max(y, axis=0)
        output_inverse_delta_list = 1 / output_inverse_delta_list

        return X_t, X_u, dt_data, n_outputs, output_inverse_delta_list

    def _simulate_chromosome_trajectory(
        self, chromosome_dict: Dict[str, float], dt_data: float, X_t: np.ndarray
    ) -> np.ndarray:
        pro_model_gen_j: pm2i.ProcessModelGenerator = self.process_model(
            dt=self.dt, **chromosome_dict
        )
        pm2i_j = pro_model_gen_j.generate_process_model_to_integrate()

        _, _, _, sol_y = pm2i_j.integrate(
            compute_u_from_t=self.compute_u_from_t,
            dt_data=dt_data,
            t_start=X_t[0],
            t_end=X_t[-1] + dt_data,
            x0=self.x0_,
            method=self.integration_method,
            timeout=self.integration_timeout,
        )

        return sol_y.T

    def _compute_chromosome_mean_error(
        self,
        y: np.ndarray,
        sol_y: np.ndarray,
        n_outputs: int,
        output_inverse_delta_list: np.ndarray,
    ) -> float:
        # TODO: deal with unstable error, create a self.is_unstable_response_
        # parameter in self.fit so that we can instead only compare the
        if sol_y.shape[0] < y.shape[0]:
            # Numerical integration blew up, and scipy.integrate cuts the
            # simulation short when state values are +/- inf. The parameters
            #  in this specific chromosome produce an unstable system
            return np.nan

        normalized_square_error = np.zeros(y.shape)

        for i in range(n_outputs):
            square = np.square(sol_y[:, i] - y[:, i])
            inverse_delta = output_inverse_delta_list[i]
            # the error for each output is normalized so that no output error
            # supersedes another output in the euclidean norm
            # see https://ieeexplore.ieee.org/document/1490788
            normalized_square_error[:, i] = square * inverse_delta

        normalized_square_error = np.linalg.norm(normalized_square_error, axis=1)
        return np.mean(normalized_square_error)

    def _evaluate_chromosome_infeasibility(
        self, chromosomes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        infeasability = np.zeros((self.n_chromosomes))

        if (
            self._inequality_constraint is not None
            and self._equality_constraint is not None
        ):
            c = [
                [],
                [],
            ]
        else:
            c = [[]]

        for j in range(self.n_chromosomes):
            if self._inequality_constraint is not None:
                c_ineq_j = self._inequality_constraint(chromosomes[:, j])
                # infeasiblity is necessarily positive since its either zero or
                # a positive value ( h(x) <0)
                c_ineq_j = np.clip(c_ineq_j, a_min=0.0, a_max=None)
                c[0].append(c_ineq_j.reshape((len(c_ineq_j), 1)))
            if self._equality_constraint is not None:
                c_eq_j = self._equality_constraint(chromosomes[:, j])
                c_eq_j = np.abs(c_eq_j) - self._equality_constraint_tolerance
                c_eq_j = np.clip(c_eq_j, a_min=0.0, a_max=None)
                c_eq_j = c_eq_j.reshape((len(c_eq_j), 1))
                if self._inequality_constraint is not None:
                    c[1].append(c_eq_j)
                else:
                    c[0].append(c_eq_j)

        c = np.block(c)

        # we find maximum value for each equality and inequality constraint
        # values throughout all chromosomes in this generation so that we can normalize
        # each constraint individually and thus no constraint is prioritized
        # over another
        c_max = np.max(c, axis=1).reshape((c.shape[0], 1))

        index_non_zero_x_max = (c_max != 0.0).reshape((c.shape[0]))
        # if every constraint is respected, we set all infeaiblity values to
        # zero, and infeasibility indexes should all be false
        if np.all(np.logical_not(index_non_zero_x_max)):
            return infeasability, infeasability > 0.0

        normalized_c = np.divide(
            c[index_non_zero_x_max, :], c_max[index_non_zero_x_max]
        )
        infeasability = np.mean(normalized_c, axis=0)

        infeasability_indexes = infeasability > 0.0

        return infeasability, infeasability_indexes

    def _find_best_solution(
        self,
        infeasibility: np.ndarray,
        infeasability_indexes: np.ndarray,
        mean_error_per_chromosome: np.ndarray,
    ) -> Tuple[float, float, bool]:
        # best individual is the one with lowest infeasibility if
        # all chromosomes are infeasible

        population_contains_feasible_solutions = True
        if np.all(infeasability_indexes):
            best_index = np.argmin(infeasibility)

            population_contains_feasible_solutions = False

            best_error = mean_error_per_chromosome[best_index]
            best_infeasibility = infeasibility[best_index]
            return (
                best_error,
                best_infeasibility,
                population_contains_feasible_solutions,
            )

        # if not all infeasible, is chosen in the feasible solutions
        indexes_feasible_solutions = np.logical_not(infeasability_indexes)

        feasible_errors = mean_error_per_chromosome[indexes_feasible_solutions]

        # the best individual is then the a feasible chromosome with the lowest
        # error
        best_index_feasible_chromosome = np.argmin(feasible_errors)

        best_error = feasible_errors[best_index_feasible_chromosome]
        best_infeasibility = 0.0

        return (
            best_error,
            best_infeasibility,
            population_contains_feasible_solutions,
        )

    def _find_worst_infeasible_solution(
        self,
        infeasibility: np.ndarray,
        infeasability_indexes: np.ndarray,
        mean_error_per_chromosome: np.ndarray,
        population_contains_feasible_solutions: bool,
    ) -> Tuple[float, float, bool]:
        apply_first_penalty = True
        # if all solutions infeasible, the worst solution is naturally the one
        # with the worst infeasibility
        if not population_contains_feasible_solutions:
            worst_index = np.argmax(infeasibility)
            worst_error = mean_error_per_chromosome[worst_index]
            worst_infeasibility = infeasibility[worst_index]
            return worst_error, worst_infeasibility, apply_first_penalty

        # not all solutions are infeasible
        indexes_feasible_solutions = np.logical_not(infeasability_indexes)

        feasible_errors = mean_error_per_chromosome[indexes_feasible_solutions]
        infeasible_errors = mean_error_per_chromosome[infeasability_indexes]

        min_feasable_error = np.min(feasible_errors)
        min_infeasable_error = np.min(infeasible_errors)

        if min_infeasable_error < min_feasable_error:
            indexes_possible_worst_solutions = infeasible_errors < min_feasable_error
            infeasiblity_infeasable_solutions = infeasibility[infeasability_indexes]
            infeasiblity_possible_worst_solutions = infeasiblity_infeasable_solutions[
                indexes_possible_worst_solutions
            ]
            index_worst_solution = np.argmax(infeasiblity_possible_worst_solutions)
            worst_infeasibility = infeasiblity_possible_worst_solutions[
                index_worst_solution
            ]

            infeasible_errors = mean_error_per_chromosome[infeasability_indexes]
            possible_worst_errors = infeasible_errors[indexes_possible_worst_solutions]
            worst_error = possible_worst_errors[index_worst_solution]

            return worst_error, worst_infeasibility, apply_first_penalty

        infeasiblity_infeasable_solutions = infeasibility[infeasability_indexes]
        index_worst_solution = np.argmax(infeasiblity_infeasable_solutions)
        worst_infeasibility = infeasiblity_infeasable_solutions[index_worst_solution]

        infeasible_errors = mean_error_per_chromosome[infeasability_indexes]

        worst_error = infeasible_errors[index_worst_solution]

        apply_first_penalty = False

        return worst_error, worst_infeasibility, apply_first_penalty

    def _find_highest_obj_func_value(
        self, mean_error_per_chromosome: np.ndarray
    ) -> np.ndarray:
        return np.max(mean_error_per_chromosome)

    def _compute_infeasibility_scaling(
        self,
        infeasiblity: np.ndarray,
        infeasability_indexes: np.ndarray,
        best_infeasibility: float,
        worst_infeasibility: float,
    ) -> np.ndarray:
        divisor = worst_infeasibility - best_infeasibility

        infeasiblity_scaling = (
            infeasiblity[infeasability_indexes] - best_infeasibility
        ) / divisor

        return infeasiblity_scaling

    def _apply_first_penalty(
        self,
        mean_error_per_chromosome: np.ndarray,
        infeasiblity_scaling: np.ndarray,
        infeasability_indexes: np.ndarray,
        best_error: float,
        worst_error: float,
    ) -> np.ndarray:
        penalized_mean_error_per_chromosome = mean_error_per_chromosome

        penalized_mean_error_per_chromosome[
            infeasability_indexes
        ] = penalized_mean_error_per_chromosome[
            infeasability_indexes
        ] + infeasiblity_scaling * (
            best_error - worst_error
        )

        return penalized_mean_error_per_chromosome

    def _evaluate_second_penalty_scaling_factor(
        self, best_error: float, worst_error: float, highest_error: float
    ):
        # TODO : error in paper in this spot... conditions are unclear.
        # check if this works as intended... maybe analyse signs are correct?
        if worst_error < best_error:
            return (highest_error - best_error) / best_error
        elif worst_error > best_error:
            return (highest_error - worst_error) / worst_error
        else:
            return 0.0

    def _apply_second_penalty(
        self,
        mean_error_per_chromosome: np.ndarray,
        infeasability_indexes: np.ndarray,
        infeasiblity_scaling: np.ndarray,
        gamma: float,
    ) -> np.ndarray:
        penalized_mean_error_per_chromosome = mean_error_per_chromosome

        infeasible_errors = penalized_mean_error_per_chromosome[infeasability_indexes]

        # It is explained in https://ieeexplore.ieee.org/abstract/document/1237163
        # that the value of this magic number does not have an effect on the
        # performance of the Genetic Algo, as long as the exponetial term
        # is there to reduce change in objectoive function value, or error,
        # for chromosomes with low infeasiblity

        # TODO: add this as an hyper-param?
        exp_weight_param = 2.0

        exponential_weighting = (
            np.exp(exp_weight_param * infeasiblity_scaling) - 1.0
        ) / (np.exp(2.0) - 1.0)

        penalized_mean_error_per_chromosome[infeasability_indexes] = (
            infeasible_errors
            + gamma * np.abs(infeasible_errors) * exponential_weighting
        )
        return penalized_mean_error_per_chromosome

    def _adaptatively_penalize_constraint_violations(
        self, mean_error_per_chromosome: np.ndarray, chromosomes: np.ndarray
    ):
        # from: https://ieeexplore.ieee.org/abstract/document/1237163

        penalized_mean_error_per_chromosome = mean_error_per_chromosome

        # If no constraints, can't penalize error
        if self._inequality_constraint is None and self._equality_constraint is None:
            return mean_error_per_chromosome

        infeasability, infeasability_indexes = self._evaluate_chromosome_infeasibility(
            chromosomes
        )

        # if all solutions are feasible, then no need to penalize constraint
        # violations as there are None
        if np.all(infeasability == 0.0):
            return mean_error_per_chromosome

        # identify bounding solutions
        # X downward hat
        (
            best_error,
            best_infeasibility,
            population_contains_feasible_solutions,
        ) = self._find_best_solution(
            infeasability, infeasability_indexes, mean_error_per_chromosome
        )

        # X hat
        (
            worst_error,
            worst_infeasibility,
            apply_first_penalty,
        ) = self._find_worst_infeasible_solution(
            infeasability,
            infeasability_indexes,
            mean_error_per_chromosome,
            population_contains_feasible_solutions,
        )

        # X downward round hat
        highest_error = self._find_highest_obj_func_value(mean_error_per_chromosome)

        infeasiblity_scaling = self._compute_infeasibility_scaling(
            infeasability,
            infeasability_indexes,
            best_infeasibility,
            worst_infeasibility,
        )

        if apply_first_penalty and best_error > worst_error:
            penalized_mean_error_per_chromosome = self._apply_first_penalty(
                mean_error_per_chromosome,
                infeasiblity_scaling,
                infeasability_indexes,
                best_error,
                worst_error,
            )

        gamma = self._evaluate_second_penalty_scaling_factor(
            best_error, worst_error, highest_error
        )

        penalized_mean_error_per_chromosome = self._apply_second_penalty(
            penalized_mean_error_per_chromosome,
            infeasability_indexes,
            infeasiblity_scaling,
            gamma,
        )

        return penalized_mean_error_per_chromosome

    def _compute_fitness_per_chromosome_from_error(
        self,
        mean_error_per_chromosome: np.ndarray,
    ):
        fitness_per_chromosome = np.zeros((self.n_chromosomes))

        indexes_non_nan_errors = ~np.isnan(mean_error_per_chromosome)

        # Normalize error to get fitness
        inverse_max_error = 1 / np.max(
            mean_error_per_chromosome[indexes_non_nan_errors]
        )
        fitness_per_chromosome[indexes_non_nan_errors] = (
            -1 * mean_error_per_chromosome[indexes_non_nan_errors] + 1
        ) * inverse_max_error

        min_fitness = np.min(fitness_per_chromosome[indexes_non_nan_errors])
        fitness_per_chromosome[indexes_non_nan_errors] = (
            fitness_per_chromosome[indexes_non_nan_errors] - min_fitness
        )

        inverse_max_fitness = np.max(fitness_per_chromosome[indexes_non_nan_errors])
        fitness_per_chromosome[indexes_non_nan_errors] = (
            fitness_per_chromosome[indexes_non_nan_errors] * inverse_max_fitness
        )

        indexes_nan_errors = np.isnan(mean_error_per_chromosome)
        # replace all nan errors (caused by unstable plants) by zero fitnesses
        fitness_per_chromosome[indexes_nan_errors] = np.zeros(
            (np.count_nonzero(indexes_nan_errors))
        )

        return fitness_per_chromosome

    def _sim_and_compute_error(
        self,
        chromosome: np.ndarray,
        dt_data: float,
        X_t: np.ndarray,
        y: np.ndarray,
        n_outputs: int,
        output_inverse_delta_list: np.ndarray,
    ):
        chromosome_j_dict = self._gen_chromosome_dict(chromosome)
        sol_y = self._simulate_chromosome_trajectory(
            chromosome_dict=chromosome_j_dict, dt_data=dt_data, X_t=X_t
        )

        return self._compute_chromosome_mean_error(
            y,
            sol_y,
            n_outputs,
            output_inverse_delta_list,
        )

    def _evaluate_chromosome_fitness(
        self,
        chromosomes: np.ndarray,
        dt_data: float,
        X_t: np.ndarray,
        n_outputs: int,
        y: np.ndarray,
        output_inverse_delta_list: List[float],
        mean_error_per_chromosome_no_penalization: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean_error_per_chromosome = mean_error_per_chromosome_no_penalization

        if not self.n_jobs:
            for j in range(self.n_chromosomes):
                # Simulate model

                if self._chromosomes_to_be_simulated[j]:
                    mean_error_per_chromosome[j] = self._sim_and_compute_error(
                        chromosome=chromosomes[:, j],
                        dt_data=dt_data,
                        X_t=X_t,
                        y=y,
                        n_outputs=n_outputs,
                        output_inverse_delta_list=output_inverse_delta_list,
                    )
        else:
            # HACK: can't pass the method directly to joblib.Parall as the class
            # instance `self` would not be serialized
            _sim_and_comp_err = lambda instnc, chrm, dt_d, X_t, y, n_out, oidl: instnc._sim_and_compute_error(
                chromosome=chrm,
                dt_data=dt_d,
                X_t=X_t,
                y=y,
                n_outputs=n_out,
                output_inverse_delta_list=oidl,
            )

            # Simulates the system represented by every chromosome that changed
            # (specified by true values in self._chromosomes_to_be_simulated)

            # HACK: next 2 expressions are wridd
            indexes_chrmsms_2_b_simed = np.array(
                np.where(self._chromosomes_to_be_simulated)
            )
            indexes_chrmsms_2_b_simed = indexes_chrmsms_2_b_simed.flatten()
            new_sim_mean_error = joblib.Parallel(n_jobs=self.n_jobs, max_nbytes=1e3)(
                joblib.delayed(_sim_and_comp_err)(
                    self,
                    chromosomes[:, j],
                    dt_data,
                    X_t,
                    y,
                    n_outputs,
                    output_inverse_delta_list,
                )
                for j in indexes_chrmsms_2_b_simed
            )

            mean_error_per_chromosome[
                self._chromosomes_to_be_simulated
            ] = new_sim_mean_error

        mean_error_per_chromosome_no_penalization = mean_error_per_chromosome

        mean_error_per_chromosome = self._adaptatively_penalize_constraint_violations(
            mean_error_per_chromosome, chromosomes
        )

        fitness_per_chromosome = self._compute_fitness_per_chromosome_from_error(
            mean_error_per_chromosome
        )

        return (
            fitness_per_chromosome,
            mean_error_per_chromosome,
            mean_error_per_chromosome_no_penalization,
        )

    def _crossover_chromosomes(
        self,
        chromosomes: np.ndarray,
        fitness_per_chromosome: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        r = rng.uniform(low=0, high=1)
        for _ in range(self.n_chromosomes):
            i = rng.integers(low=0, high=self.n_chromosomes)
            j = i
            while j == i:
                j = rng.integers(low=0, high=self.n_chromosomes)
            if fitness_per_chromosome[i] > fitness_per_chromosome[j]:
                chromosomes[:, j] = r * chromosomes[:, i] + (1 - r) * chromosomes[:, j]
                self._chromosomes_to_be_simulated[j] = True
            elif fitness_per_chromosome[i] < fitness_per_chromosome[j]:
                chromosomes[:, i] = r * chromosomes[:, j] + (1 - r) * chromosomes[:, i]
                self._chromosomes_to_be_simulated[i] = True

        return chromosomes

    def _mutate_chromosomes(
        self,
        chromosomes: np.ndarray,
        fitness_per_chromosome: np.ndarray,
        param_std_deviation: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        n_params = chromosomes.shape[0]

        r = rng.uniform(low=0, high=1)
        for j in range(self.n_chromosomes):
            if fitness_per_chromosome[j] < r:
                for i in range(n_params):
                    chromosomes[i, j] = chromosomes[i, j] + rng.normal(
                        loc=0, scale=param_std_deviation[i]
                    )
                    self._chromosomes_to_be_simulated[j] = True

        return chromosomes

    def _select_elite_chromosome(
        self,
        chromosomes: np.ndarray,
        mean_error_per_chromosome: np.ndarray,
        mean_error_per_chromosome_no_penalization: np.ndarray,
        generation_index: int,
    ):
        non_nan_error_indexes = ~np.isnan(mean_error_per_chromosome)
        valid_errors = mean_error_per_chromosome[non_nan_error_indexes]
        index_elite_chromosome = np.argmin(valid_errors)
        new_elite_chromosome_error = valid_errors[index_elite_chromosome]

        if (
            generation_index == 0
            or new_elite_chromosome_error < self._elite_chromosome_error
        ):
            self._elite_chromosome_error = new_elite_chromosome_error
            valid_errors_no_penal = mean_error_per_chromosome_no_penalization[
                non_nan_error_indexes
            ]
            self._elite_chromosome_error_no_penal = valid_errors_no_penal[
                index_elite_chromosome
            ]
            valid_chromosomes = chromosomes[:, non_nan_error_indexes]
            self._elite_chromosome = valid_chromosomes[:, index_elite_chromosome]

        # This list lets us see progress of chromosome error
        self._elite_chromosome_error_list[
            generation_index
        ] = self._elite_chromosome_error

    def _replace_some_chromosomes_with_elite(
        self,
        chromosomes: np.ndarray,
        mean_error_per_chromosome_no_penalization: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        for _ in range(self._n_chromosomes_to_replace):
            k = rng.integers(low=0, high=self.n_chromosomes)
            chromosomes[:, k] = self._elite_chromosome
            mean_error_per_chromosome_no_penalization[
                k
            ] = self._elite_chromosome_error_no_penal
        return chromosomes, mean_error_per_chromosome_no_penalization

    def _check_for_termination_condition(
        self,
        generation_index: int,
    ):
        if generation_index == 0:
            # don't want to stop on first generation
            return False

        # want to check if the chromosome error is going down
        if self._g_index is None:
            # store generation index when error first goes down
            if (
                self._elite_chromosome_error_list[-1]
                < self._elite_chromosome_error_list[-2]
            ):
                self._g_index = generation_index
            else:
                return False

        # want to wait a few generations after error first goes down for the chromosome error
        # to settle into an exponential distribution
        if generation_index - self._g_index <= self.can_terminate_after_index:
            return False

        error_running_mean = np.mean(self._elite_chromosome_error_list[self._g_index :])

        if error_running_mean <= self.ratio_max_error_for_termination * np.max(
            self._elite_chromosome_error_list
        ):
            return True

        return False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_iter: int = 0,
        x0: np.ndarray = None,
    ) -> "Genetic":
        """Fit the model.

        This algorithm is based on the algorithm found in
        https://www.sciencedirect.com/science/article/pii/S0019057807000237

        Parameters
        ----------
        X : np.ndarray
            Input Data
        y : np.ndarray
            Output Data
        n_iter: np.ndarray
            Maximum number of chromosome generations before the algorithm stops,
            which is equivalent to the maximum number of iterations.
            The real number of iterations migh be smaller if the termination
            condition is reached before `n_iter` iterations.
        x0: np.ndarray
            Initial state of the system being identified. Should be an array with
            same dimension as the array returned by the
            `compute_state_derivative` of the `self.process_model` class.
            In some cases can be inferred from first datapoints in `X` and `y`.

        Returns
        -------
        Genetic
            Instance of itself.

        Raises
        -----
        ValueError
            If constructor or fit parameters are incorrect.
        """

        self.x0_ = x0

        self._validate_chromosome_parameter_range()
        self._validate_replacement_ratio()
        self._validate_ratio_max_error_for_termination()
        self._validate_integration_method()

        # Random number generator. Can specify seed for reproducibility.
        rng = default_rng(self.seed)

        self._initialize_replacement_step_variables()
        self._initialize_termination_checking_variables(n_iter)
        self._initialize_simulation_flags()
        self._initialize_constraints()

        chromosomes, param_std_deviation = self._initialize_chromosomes(rng)

        (
            X_t,
            X_u,
            dt_data,
            n_outputs,
            output_inverse_delta_list,
        ) = self._preprocessing_input_output_data(X, y)

        mean_error_per_chromosome_no_penalization = np.zeros((self.n_chromosomes))

        # TODO: handle if self.process_model is None
        # TODO: handle continuous/discrete sim if self.dt is None or not
        # TODO: check wheter dt = dt_data if dt is not None
        # TODO: handle input fction as an interpolation of X_u if no access to
        # fction from which u is computed (so if self.compute_u_from_t is None)
        # TODO: take inspiration from sklearn https://github.com/scikit-learn/scikit-learn/blob/2a2772a87b6c772dc3b8292bcffb990ce27515a8/sklearn/neighbors/_base.py#L879
        # or https://scikit-learn.org/stable/computing/parallelism.html#parallelism
        # for how to use multiple threads when n_jobs is not None

        for generation_index in range(n_iter):
            start_time = time()
            # simulate and evaluate fitness
            (
                fitness_per_chromosome,
                mean_error_per_chromosome,
                mean_error_per_chromosome_no_penalization,
            ) = self._evaluate_chromosome_fitness(
                chromosomes=chromosomes,
                dt_data=dt_data,
                X_t=X_t,
                n_outputs=n_outputs,
                y=y,
                output_inverse_delta_list=output_inverse_delta_list,
                mean_error_per_chromosome_no_penalization=mean_error_per_chromosome_no_penalization,
            )

            # we will only simulate trajectory for chromosomes that have changed in crossover, mutate or replacement steps
            self._chromosomes_to_be_simulated = np.full((self.n_chromosomes), False)

            # cross over
            chromosomes = self._crossover_chromosomes(
                chromosomes, fitness_per_chromosome, rng
            )
            # mutation
            chromosomes = self._mutate_chromosomes(
                chromosomes, fitness_per_chromosome, param_std_deviation, rng
            )
            # elitism
            self._select_elite_chromosome(
                chromosomes,
                mean_error_per_chromosome,
                mean_error_per_chromosome_no_penalization,
                generation_index,
            )
            # replacement
            (
                chromosomes,
                mean_error_per_chromosome_no_penalization,
            ) = self._replace_some_chromosomes_with_elite(
                chromosomes, mean_error_per_chromosome_no_penalization, rng
            )

            print(
                f"Generation {generation_index} | Current Error: {self._elite_chromosome_error} | Time elapsed: {time() - start_time}"
            )

            # termination
            if self._check_for_termination_condition(generation_index):
                return self

        return self

    def predict(
        self,
        X: Union[np.ndarray, Tuple[np.ndarray, Callable[[float], np.ndarray]]],  # HACK
    ) -> np.ndarray:
        """

        Parameters
        ----------
        X : np.ndarray
            Data matrix.

        Returns
        -------
        np.ndarray
            Predicted data matrix.
        """
        if isinstance(X, np.ndarray):
            X_t, X_u = util.split_time_input(X)
            dt_data = X_t[1] - X_t[0]
            raise NotImplementedError(
                "Need to implement interpolation function to compute input at times not specified in X_t"
            )
        else:
            X_t, compute_u_from_t = X
            dt_data = X_t[1] - X_t[0]

            elite_chromosome_dict = self._gen_chromosome_dict(self._elite_chromosome)

            sol_y = self._simulate_chromosome_trajectory(
                chromosome_dict=elite_chromosome_dict, dt_data=dt_data, X_t=X_t
            )

        return sol_y

    # Extra estimator tags
    # https://scikit-learn.org/stable/developers/develop.html#estimator-tags
    def _more_tags(self):
        return {
            "multioutput": True,
            "multioutput_only": True,
        }
