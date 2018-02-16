from __future__ import absolute_import, division, print_function
import numpy as np, sys
from abc import abstractmethod, ABCMeta


class HyperparameterBackend(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, grid):
        """
        Parameters
        ----------
        grid: dict
            Keys are hyperparameter names and values are either
            a single (min, max) tuple for single value parameters
            or a tuple of (min, max) tuples for tuple-valued parameters.
        """
        pass

    @abstractmethod
    def get_next_hyperparameters(self):
        pass

    @abstractmethod
    def record_result(self, hyperparam_dict, score):
        """
        Parameters
        ----------
        hyperparam_dict: dict
            hyperparameter names as keys and values as values.
        score: int or float
            The result, or metric value, of using the hyparameters.
        """
        pass


class RandomSearch(HyperparameterBackend):
    def __init__(self, grid):
        self.grid = grid

    def get_next_hyperparameters(self):
        return [np.random.uniform(start, end) for start, end in self.grid]

    def record_result(self, hyperparam_dict, score):
        pass  # Random search doesn't base its decisions on the results of previous trials


class MOESearch(HyperparameterBackend):
    def __init__(self, grid):
        if sys.version_info[0] == 2:
            from httplib import BadStatusLine
            from moe.easy_interface.experiment import Experiment
            from moe.easy_interface.simple_endpoint import gp_next_points
            from moe.optimal_learning.python.data_containers import SamplePoint
        else:
            raise RuntimeError("MOESearch requires Python2!")
        self.experiment = Experiment(grid)

    def get_next_hyperparameters(self):
        try:
            return gp_next_points(self.experiment)[0]
        except BadStatusLine:
            raise RuntimeError('MOE server is not running!')

    def record_result(self, hyperparam_dict, score):
        self.experiment.historical_data.append_sample_points(
            [SamplePoint(point=hyperparam_dict.values(), value=score)])


class HyperparameterSearcher(object):
    def __init__(self, model_class, fixed_hyperparameters, grid, X_train, y_train, validation_data,
                 metric='auPRC', maximize=True, backend=RandomSearch):
        self.model_class = model_class
        self.fixed_hyperparameters = fixed_hyperparameters
        self.grid = grid
        self.X_train = X_train
        self.y_train = y_train
        self.validation_data = validation_data
        self.metric = metric
        self.maximize = maximize
        self.best_score = 0
        self.best_model = self.best_hyperparameters = None
        # Some hyperparameters have multiple elements, and we need backend to treat each of them
        # as a separate dimension, so unpack them here.
        backend_grid = [bounds for value in grid.values()
                        for bounds in (value if isinstance(value[0], (list, tuple, np.ndarray))
                                       else (value,))]
        self.backend = backend(backend_grid)

    def search(self, num_hyperparameter_trials):
        for trial in range(num_hyperparameter_trials):
            # Select next hyperparameters with MOE, rounding hyperparameters that are integers
            # and re-packing multi-element hyperparameters
            raw_hyperparameters = self.backend.get_next_hyperparameters()
            hyperparameters = {}
            i = 0
            for name, bounds in self.grid.items():
                if isinstance(bounds[0], (list, tuple, np.ndarray)):
                    # Multi-element hyperparameter
                    hyperparameters[name] = raw_hyperparameters[i : i + len(bounds)]
                    if isinstance(bounds[0][0], int):
                        hyperparameters[name] = np.rint(hyperparameters[name]).astype(int)
                    i += len(bounds)
                else:
                    hyperparameters[name] = raw_hyperparameters[i]
                    if isinstance(bounds[0], int):
                        hyperparameters[name] = int(round(hyperparameters[name]))
                    i += 1
            assert i == len(raw_hyperparameters)
            # Try these hyperparameters
            model = self.model_class(**{key: value
                                        for dictionary in (hyperparameters, self.fixed_hyperparameters)
                                        for key, value in dictionary.items()})
            model.train(self.X_train, self.y_train, validation_data=self.validation_data)
            print(self.validation_data)
            task_scores = model.score(self.validation_data[0], self.validation_data[1], self.metric)
            score = task_scores.mean()  # mean across tasks
            # Record hyperparameters and validation loss
            self.backend.record_result(hyperparameters, score)
            # If these hyperparameters were the best so far, store this model
            if self.maximize == (score > self.best_score):
                self.best_score = score
                self.best_model = model
                self.best_hyperparameters = hyperparameters
