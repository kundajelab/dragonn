from __future__ import division

import random
random.seed(1)
import numpy as np
np.random.seed(1)

from collections import namedtuple, defaultdict, OrderedDict
from simdna import simulations
from simdna.synthetic import StringEmbeddable
from dragonn.utils import get_motif_scores, one_hot_encode
from sklearn.model_selection import train_test_split 

Data = namedtuple('Data', ('X_train', 'X_valid', 'X_test',
                           'train_embeddings', 'valid_embeddings', 'test_embeddings',
                           'y_train', 'y_valid', 'y_test',
                           'motif_names'))


def get_available_simulations():
    return [function_name for function_name in dir(simulations)
            if "simulate" in function_name]


def print_available_simulations():
    for function_name in get_available_simulations():
        print(function_name)


def get_simulation_function(simulation_name):
    if simulation_name in get_available_simulations():
        return getattr(simulations, simulation_name)
    else:
        print("%s is not available. Available simulations are:" % (simulation_name))
        print_available_simulations()


def print_simulation_info(simulation_name):
    simulation_function = get_simulation_function(simulation_name)
    if simulation_function is not None:
        print(simulation_function.__doc__)


def get_simulation_data(simulation_name, simulation_parameters,
                        test_set_size=4000, validation_set_size=3200):
    simulation_function = get_simulation_function(simulation_name)
    sequences, y, embeddings = simulation_function(**simulation_parameters)
    if simulation_name == "simulate_heterodimer_grammar":
        motif_names = [simulation_parameters["motif1"],
                       simulation_parameters["motif2"]]
    elif simulation_name == "simulate_multi_motif_embedding":
        motif_names = simulation_parameters["motif_names"]
    else:
        motif_names = [simulation_parameters["motif_name"]]

    train_sequences, test_sequences, train_embeddings, test_embeddings, y_train, y_test = \
        train_test_split(sequences, embeddings, y, test_size=test_set_size)
    train_sequences, valid_sequences, train_embeddings, valid_embeddings, y_train, y_valid = \
        train_test_split(train_sequences, train_embeddings, y_train, test_size=validation_set_size)
    X_train = one_hot_encode(train_sequences)
    X_valid = one_hot_encode(valid_sequences)
    X_test = one_hot_encode(test_sequences)

    return Data(X_train, X_valid, X_test, train_embeddings, valid_embeddings, test_embeddings,
                y_train, y_valid, y_test, motif_names)


