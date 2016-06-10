from __future__ import absolute_import, division, print_function
import numpy as np, random
np.random.seed(1)
random.seed(1)
from sklearn.cross_validation import train_test_split
from dragonn.simulations import (
    simulate_single_motif_detection,
    simulate_motif_counting,
    simulate_motif_density_localization,
    simulate_multi_motif_embedding,
    simulate_differential_accessibility
)
from dragonn.models import SequenceDNN, MotifScoreRNN, gkmSVM, SVC
from dragonn.utils import one_hot_encode, get_motif_scores, reverse_complement

# Settings

seq_length = 500
num_sequences = 8000
num_positives = 4000
num_negatives = num_sequences - num_positives
GC_fraction = 0.4
test_fraction = 0.2
validation_fraction = 0.2
do_hyperparameter_search = False
num_hyperparameter_trials = 100
use_deep_CNN = False
use_RNN = False

print('Generating sequences...')

sequences, labels = simulate_single_motif_detection(
    'SPI1_disc1', seq_length, num_positives, num_negatives, GC_fraction)

print('One-hot encoding sequences...')

encoded_sequences = one_hot_encode(sequences)

print('Getting motif scores...')

motif_scores = get_motif_scores(encoded_sequences, motif_names=['SPI1_disc1'])

print('Partitioning data into training, validation and test sets...')

X_train, X_test, y_train, y_test = train_test_split(encoded_sequences, labels, test_size=test_fraction)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=validation_fraction)

print('Adding reverse complements...')

X_train = np.concatenate((X_train, reverse_complement(X_train)))
y_train = np.concatenate((y_train, y_train))
random_order = np.arange(len(X_train))
np.random.shuffle(random_order)
X_train = X_train[random_order]
y_train = y_train[random_order]

# Build model, train and test

if not do_hyperparameter_search:
    hyperparameters = {'seq_length': seq_length, 'use_deep_CNN': use_deep_CNN, 'use_RNN': use_RNN,
                       'num_filters': 45, 'pool_width': 25, 'conv_width': 10, 'L1': 0, 'dropout': 0.2}
    if use_deep_CNN:
        hyperparameters.update({'num_filters_2': 50, 'conv_width_2': 8,
                                'num_filters_3': 50, 'conv_width_3': 5})
    if use_RNN:
        hyperparameters.update({'GRU_size': 35, 'TDD_size': 45})
    model = SequenceDNN(**hyperparameters)
    model.train(X_train, y_train, validation_data=(X_valid, y_valid))
    print('Test results: {}'.format(model.test(X_test, y_test)))

else:
    print('Starting hyperparameter search...')
    from hyperparameter_search import HyperparameterSearcher
    fixed_hyperparameters = {'seq_length': seq_length, 'use_deep_CNN': use_deep_CNN, 'use_RNN': use_RNN}
    grid = {'num_filters': (5, 100), 'pool_width': (5, 40), 'conv_width': (6, 20), 'dropout': (0, 0.5)}
    if use_deep_CNN:
        grid.update({'num_filters_2': (5, 100), 'conv_width_2': (6, 20),
                     'num_filters_3': (5, 100), 'conv_width_3': (6, 20),
        })
    if use_RNN:
        grid.update({'GRU_size': (10, 50), 'TDD_size': (20, 60)})

    searcher = HyperparameterSearcher(SequenceDNN, fixed_hyperparameters, grid, X_train, y_train,
                                      validation_data=(X_valid, y_valid))
    searcher.search(num_hyperparameter_trials)
    print('Best hyperparameters: {}'.format(searcher.best_hyperparameters))
    print('Test results: {}'.format(searcher.best_model.test(X_test, y_test)))
