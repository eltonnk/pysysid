"""Genetic algorithm system identification methods."""

from typing import Any, Dict, Tuple, Callable, List

import numpy as np
import sklearn.base
import util
from numpy.random import default_rng
import pm2i


class Genetic(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    """Genetic algorithm model."""

    # Array check parameters for :func:`fit` when ``X`` and ``y` are given
    _check_X_y_params: Dict[str, Any] = {
        'multi_output': True,
        'y_numeric': True,
    }

    # Array check parameters for :func:`predict` and :func:`fit` when only
    # ``X`` is given
    _check_array_params: Dict[str, Any] = {
        'dtype': 'numeric',
    }

    def __init__(
        self,
        dyn_model: type[pm2i.ProcessModelGenerator] = None,
        dt: float = None,
        compute_u_from_t: Callable[[float], np.ndarray] = None,
        x0: np.ndarray = None,
        n_chromosomes: int = 2,
        replace_with_best_ratio: float = 0.01,
        can_terminate_after_index: int = 2,
        ratio_max_error_for_termination:  float= 0.2,
        seed: int = None,
        chromosome_parameter_ranges: Dict[str, Tuple[float, float]] = None,
        n_jobs=None,
    ) -> None:
        """Instantiate :class:`Genetic`.

        Parameters
        ----------
        dyn_model : type[DynamicModel]
            TODO
        dt: float
            TODO
        compute_u_from_t: Callable[[float], np.ndarray]
            TODO
        x0: np.ndarray 
            TODO
        n_chromosomes: int
            TODO
        seed: int
            TODO
        chromosome_parameter_ranges : Dict[str, Tuple[float, float]]
            TODO
        """
        self.dyn_model = dyn_model
        self.dt = dt
        self.compute_u_from_t = compute_u_from_t
        self.x0 = x0
        self.n_chromosomes = n_chromosomes
        self.replace_with_best_ratio = replace_with_best_ratio
        self.can_terminate_after_index = can_terminate_after_index
        self.ratio_max_error_for_termination = ratio_max_error_for_termination
        self.seed = seed
        self.chromosome_parameter_ranges = chromosome_parameter_ranges
        self.n_jobs = n_jobs


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

    def _validate_replacement_ratio(self):
        if self.ratio_max_error_for_termination < 0 or self.ratio_max_error_for_termination > 1:
            raise ValueError(
                f"""Constructor argument 'ratio_max_error_for_termination'
                must have a must have a value in the [0.0, 1.0] interval. 
                Current value is {self.ratio_max_error_for_termination}."""
            )
        
        # TODO: raise warning if the replacement ratio is over  50%, 70%, 90% ? 
        # (dont want to stop if error has not gone down)
        
    def _initialize_replacement_step_variables(self):
        self._n_chromosomes_to_replace = int(self.n_chromosomes * self.replace_with_best_ratio)

    def _initialize_termination_checking_variables(self, n_iter:int):
        self._g_index = None
        self._elite_chromosome_error_list = np.zeros((n_iter))
        
            
    def _initialize_chromosomes(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        n_params = len(self.chromosome_parameter_ranges.keys())
        # Columns: chromosomes
        # Rows: parameters
        chromosomes: np.ndarray = np.zeros((n_params, self.n_chromosomes))

        # Initialize all parameters in all chromosomes randomly, according to approximate
        # range of values we think the parameters should fall in
        for i, param_range in enumerate(list(self.chromosome_parameter_ranges.values())):
            param_min, param_max = param_range
            chromosomes[i, :] = rng.uniform(low=param_min, high=param_max, size=self.n_chromosomes)


        param_std_deviation = np.std(chromosomes, axis=1)
        return chromosomes, param_std_deviation

    def _gen_chromosome_dict(self, param_array: np.ndarray) -> Dict[str, float]:
        chromosome_dict = {}
        for param_name, param_value in zip(list(self.chromosome_parameter_ranges.keys()), param_array):
            chromosome_dict[param_name] = param_value

        return chromosome_dict
    
    def _preprocessing_input_output_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float, int, List[float]]:
        X_t, X_u = util.split_time_input(X)
        dt_data = X_t[1] - X_t[0]

        n_outputs = y.shape[1]
        output_inverse_delta_list = []
        for i in range(n_outputs):
            out_max = np.max(y[:, i])
            out_min = np.min(y[:, i])
            out_delta = out_max - out_min
            inverse_out_delta = 1/ out_delta
            output_inverse_delta_list.append(inverse_out_delta)

        return X_t , X_u , dt_data, n_outputs, output_inverse_delta_list
    
    def _simulate_chromosome_trajectory(self, chromosome_dict: Dict[str, float], dt_data:float, X_t:np.ndarray) -> np.ndarray:
        pro_model_gen_j: pm2i.ProcessModelGenerator = self.dyn_model(dt=self.dt, **chromosome_dict)
        pm2i_j = pro_model_gen_j.generate_process_model_to_integrate()
        # TODO How will initial conditions work? here x0 is defined as zero, should be passed as a hyperparameter
        _, _, _, sol_y = pm2i_j.integrate(
            compute_u_from_t=self.compute_u_from_t, 
            dt_data=dt_data, 
            t_start=X_t[0], 
            t_end=X_t[-1]+dt_data, 
            x0=self.x0
        )

        return sol_y
    
    def _evaluate_chromosome_fitness(
            self, 
            chromosomes: np.ndarray,
            dt_data: float, 
            X_t: np.ndarray, 
            n_outputs: int,
            y: np.ndarray,
            output_inverse_delta_list: List[float],
        ) -> Tuple[np.ndarray, np.ndarray]:

        fitness_per_chromosome = np.zeros((self.n_chromosomes))
        mean_error_per_chromosome = np.zeros((self.n_chromosomes))
        for j in range(self.n_chromosomes):
            # Simulate model 
            chromosome_j_dict = self._gen_chromosome_dict(chromosomes[:, j])
            sol_y = self._simulate_chromosome_trajectory(
                chromosome_dict=chromosome_j_dict,
                dt_data=dt_data,
                X_t=X_t
            )

            # TODO: verification sol_y has same shape as y
            normalized_square_error = np.zeros(y.shape)

            for i in range(n_outputs):
                normalized_square_error[:,i] = np.squared(sol_y[:, i] - y[:, i]) * output_inverse_delta_list[i]

            normalized_square_error = np.linalg.norm(normalized_square_error, axis=1)
            mean_error_per_chromosome[j]= np.mean(normalized_square_error)

        # Normalize error to get fitness
        inverse_max_error = 1 / np.max(mean_error_per_chromosome)
        fitness_per_chromosome =  (-1*mean_error_per_chromosome + 1) * inverse_max_error

        min_fitness = np.min(fitness_per_chromosome)
        fitness_per_chromosome = fitness_per_chromosome - min_fitness
        
        inverse_max_fitness = np.max(fitness_per_chromosome)
        fitness_per_chromosome = fitness_per_chromosome * inverse_max_fitness

        return fitness_per_chromosome, mean_error_per_chromosome
    

    def _crossover_chromosomes(
            self, 
            chromosomes: np.ndarray, 
            fitness_per_chromosome: np.ndarray, 
            rng: np.random.Generator
        ) -> np.ndarray:

        r = rng.uniform(low=0, high=1)
        for _ in range(self.n_chromosomes):
            i = rng.integers(low=0, high=self.n_chromosomes)
            j = i
            while j == i:
                j = rng.integers(low=0, high=self.n_chromosomes)
            if fitness_per_chromosome[i] > fitness_per_chromosome[j]:
                chromosomes[:, j] = r* chromosomes[:, i] + (1 - r) * chromosomes[:, j]
            elif fitness_per_chromosome[i] < fitness_per_chromosome[j]:
                chromosomes[:, i] = r* chromosomes[:, j] + (1 - r) * chromosomes[:, i]

        return chromosomes
    
    def _mutate_chromosomes(
            self, 
            chromosomes: np.ndarray, 
            fitness_per_chromosome: np.ndarray, 
            param_std_deviation: np.ndarray,
            rng: np.random.Generator
    ) -> np.ndarray:
        n_params = chromosomes.shape[0]
        
        r = rng.uniform(low=0, high=1)
        for j in range(self.n_chromosomes):
            if fitness_per_chromosome[j] < r:
                for i in range(n_params):
                    chromosomes[i, j] = chromosomes[i, j] + rng.normal(loc=0, scale=param_std_deviation[i])


        return chromosomes
    
    def _select_elite_chromosome(
            self,
            chromosomes: np.ndarray,
            mean_error_per_chromosome: np.ndarray,
            generation_index: int
    ):
        if generation_index == 0:
            self._elite_chromosome_error = np.min(mean_error_per_chromosome)
            index_elite_chromosome = np.argmin(mean_error_per_chromosome)
            self._elite_chromosome = chromosomes[:, index_elite_chromosome]
            return
        
        generational_min_chromosome_error = np.min(mean_error_per_chromosome)

        if generational_min_chromosome_error < self._elite_chromosome_error:
            self._elite_chromosome_error = generational_min_chromosome_error
            index_elite_chromosome = np.argmin(mean_error_per_chromosome)
            self._elite_chromosome = chromosomes[:, index_elite_chromosome]
        
        # This list lets us see progress of chromosome error
        self._elite_chromosome_error_list[generation_index] = self._elite_chromosome_error
            
    def _replace_some_chromosomes_with_elite(
            self, 
            chromosomes: np.ndarray, 
            rng: np.random.Generator
        ) -> np.ndarray:
        for _ in range(self._n_chromosomes_to_replace):
            k = rng.integers(low=0, high=self.n_chromosomes)
            chromosomes[:, k] = self._elite_chromosome

        return chromosomes
    
    def _check_for_termination_condition(
            self,
            generation_index : int,
        ):
            if generation_index == 0:
                # don't want to stop on first generation
                return False
            
            # want to check if the chromosome error is going down
            if self._g_index is None:
                # store generation index when error first goes down
                if self._elite_chromosome_error_list[-1] < self._elite_chromosome_error_list[-2]:
                    self._g_index = generation_index
                else:
                    return False
                
            # want to wait a few generations after error first goes down for the chromosome error
            # to settle into an exponential distribution
            if generation_index - self._g_index <= self.can_terminate_after_index:
                return False
            
            error_running_mean = np.mean(self._elite_chromosome_error_list[self._g_index:])

            if error_running_mean <= self.ratio_max_error_for_termination * np.max(self._elite_chromosome_error_list):
                return True

            return False


    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_iter: int = 0,
    ) -> 'Genetic':
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

        Returns
        -------
        Genetic
            Instance of itself.

        Raises
        -----
        ValueError
            If constructor or fit parameters are incorrect.
        """
        
        self._validate_chromosome_parameter_range()
        self._validate_replacement_ratio()
        self._validate_replacement_ratio()

        # Random number generator. Can specify seed for reproducibility.
        rng = default_rng(self.seed)

        self._initialize_replacement_step_variables()
        self._initialize_termination_checking_variables(n_iter)

        chromosomes, param_std_deviation = self._initialize_chromosomes(rng)

        (
            X_t, 
            X_u, 
            dt_data, 
            n_outputs, 
            output_inverse_delta_list
        ) = self._preprocessing_input_output_data(X, y)

        # TODO: handle if self.dyn_model is None
        # TODO: handle continuous/discrete sim if self.dt is None or not
        # TODO: check wheter dt = dt_data if dt is not None
        # TODO: handle input fction as an interpolation of X_u if no access to 
        # fction from which u is computed (so if self.compute_u_from_t is None)
        # TODO: take inspiration from sklearn https://github.com/scikit-learn/scikit-learn/blob/2a2772a87b6c772dc3b8292bcffb990ce27515a8/sklearn/neighbors/_base.py#L879 
        # or https://scikit-learn.org/stable/computing/parallelism.html#parallelism
        # for how to use multiple threads when n_jobs is not None
        
        for generation_index in range(n_iter):
            # simulate and evaluate fitness
            fitness_per_chromosome, mean_error_per_chromosome = self._evaluate_chromosome_fitness(
                chromosomes=chromosomes,
                dt_data=dt_data,
                X_t=X_t,
                n_outputs=n_outputs,
                y=y,
                output_inverse_delta_list=output_inverse_delta_list,
            )
            
            # cross over
            chromosomes = self._crossover_chromosomes(chromosomes, fitness_per_chromosome, rng)
            # mutation
            chromosomes = self._mutate_chromosomes(chromosomes, fitness_per_chromosome, param_std_deviation, rng)
            # elitism
            self._select_elite_chromosome(chromosomes, mean_error_per_chromosome, generation_index)
            # replacement
            chromosomes = self._replace_some_chromosomes_with_elite(chromosomes, rng)
            # termination
            if self._check_for_termination_condition(generation_index):
                return self
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Perform a single-step prediction for each state in each episode.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.

        Returns
        -------
        np.ndarray
            Predicted data matrix.
        """
        # TODO WIll single-step prediction work here?
        # TODO How will initial conditions work?
        pass

    # Extra estimator tags
    # https://scikit-learn.org/stable/developers/develop.html#estimator-tags
    def _more_tags(self):
        return {
            'multioutput': True,
            'multioutput_only': True,
        }
