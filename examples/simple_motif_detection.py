from __future__ import absolute_import, division, print_function
import numpy as np, random
np.random.seed(1)
random.seed(1)
from dragonn.models import SequenceDNN
from simdna.simulations import simulate_single_motif_detection
from dragonn.utils import one_hot_encode, get_motif_scores, reverse_complement
try:
    from sklearn.model_selection import train_test_split  # sklearn >= 0.18
except ImportError:
    from sklearn.cross_validation import train_test_split  # sklearn < 0.18
import sys

# Settings

seq_length = 500
num_sequences = 8000
num_positives = 4000
num_negatives = num_sequences - num_positives
GC_fraction = 0.4
test_fraction = 0.2
validation_fraction = 0.2
do_hyperparameter_search = False
num_hyperparameter_trials = 50
num_epochs = 100
use_deep_CNN = False
use_RNN = False

print('Generating sequences...')

sequences, labels, embeddings = simulate_single_motif_detection(
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

print('Randomly splitting data into training and test sets...')

random_order = np.arange(len(X_train))
np.random.shuffle(random_order)
X_train = X_train[random_order]
y_train = y_train[random_order]

# Build and train model

if not do_hyperparameter_search:
    hyperparameters = {'seq_length': seq_length, 'use_RNN': use_RNN,
                       'num_filters': (45,), 'pool_width': 25, 'conv_width': (10,),
                       'L1': 0, 'dropout': 0.2, 'num_epochs': num_epochs}
    if use_deep_CNN:
        hyperparameters.update({'num_filters': (45, 50, 50), 'conv_width': (10, 8, 5)})
    if use_RNN:
        hyperparameters.update({'GRU_size': 35, 'TDD_size': 45})
    model = SequenceDNN(**hyperparameters)
    model.train(X_train, y_train, validation_data=(X_valid, y_valid),
                save_best_model_to_prefix='best_model')

else:
    print('Starting hyperparameter search...')
    from dragonn.hyperparameter_search import HyperparameterSearcher, RandomSearch
    fixed_hyperparameters = {'seq_length': seq_length, 'use_RNN': use_RNN, 'num_epochs': num_epochs}
    grid = {'num_filters': ((5, 100),), 'pool_width': (5, 40),
            'conv_width': ((6, 20),), 'dropout': (0, 0.5)}
    if use_deep_CNN:
        grid.update({'num_filters': ((5, 100), (5, 100), (5, 100)),
                     'conv_width': ((6, 20), (6, 20), (6, 20))})
    if use_RNN:
        grid.update({'GRU_size': (10, 50), 'TDD_size': (20, 60)})

    # Backend is RandomSearch; if using Python 2, can also specify MOESearch
    # (requires separate installation)
    searcher = HyperparameterSearcher(SequenceDNN, fixed_hyperparameters, grid, X_train, y_train,
                                      validation_data=(X_valid, y_valid), backend=RandomSearch)
    searcher.search(num_hyperparameter_trials)
    print('Best hyperparameters: {}'.format(searcher.best_hyperparameters))
    model = searcher.best_model

# Test model

print('Test results: {}'.format(model.test(X_test, y_test)))

# Plot DeepLift and ISM scores for the first 10 test examples, and model architecture

if sys.version[0] == 2:
    model.plot_deeplift(X_test[:10], output_directory='deeplift_plots')
model.plot_in_silico_mutagenesis(X_test[:10], output_directory='ISM_plots')
model.plot_architecture(output_file='architecture_plot.png')
