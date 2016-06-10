from __future__ import absolute_import, division, print_function
import sys
if sys.version_info[0] == 3:
    exit('Hyperparameter search with MOE requires Python 2.')
from ConfigParser import _Chainmap as ChainMap
from moe.easy_interface.experiment import Experiment
from moe.easy_interface.simple_endpoint import gp_next_points
from moe.optimal_learning.python.data_containers import SamplePoint


class HyperparameterSearcher(object):
    def __init__(self, model_class, fixed_hyperparameters, grid, X_train, y_train, validation_data,
                 metric='auPRG', maximize=True):
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
        self.experiment = Experiment(grid.values())

    def search(self, num_hyperparameter_trials):
        for trial in range(num_hyperparameter_trials):
            # Select next hyperparameters with MOE, rounding hyperparameters that are integers
            raw_hyperparameters = gp_next_points(self.experiment)[0]
            hyperparameters = {name: int(round(value)) if isinstance(bounds[0], int) else value
                               for (name, bounds), value in
                               zip(self.grid.items(), raw_hyperparameters)}
            # Try these hyperparameters
            model = self.model_class(**ChainMap(hyperparameters, self.fixed_hyperparameters))
            model.train(self.X_train, self.y_train, validation_data=self.validation_data)
            score = model.score(self.metric, *self.validation_data)
            # Record hyperparameters and validation loss
            self.experiment.historical_data.append_sample_points(
                [SamplePoint(point=hyperparameters.values(), value=score)])
            # If these hyperparameters were the best so far, store this model
            if self.maximize == (score > self.best_score):
                self.best_score = score
                self.best_model = model
                self.best_hyperparameters = hyperparameters