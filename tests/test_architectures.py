from __future__ import absolute_import, division, print_function
import numpy as np, pytest, random, sys
sys.path.append('../dragonn')
np.random.seed(1)
random.seed(1)
from collections import OrderedDict
from dragonn.models import SequenceDNN
from dragonn.simulations import simulate_single_motif_detection
from dragonn.utils import one_hot_encode, reverse_complement
from sklearn.cross_validation import train_test_split


def run(use_deep_CNN, use_RNN, label, golden_results):
    seq_length = 50
    num_sequences = 100
    num_positives = 50
    num_negatives = num_sequences - num_positives
    GC_fraction = 0.4
    test_fraction = 0.2
    validation_fraction = 0.2
    num_epochs = 1

    sequences, labels = simulate_single_motif_detection(
        'SPI1_disc1', seq_length, num_positives, num_negatives, GC_fraction)
    encoded_sequences = one_hot_encode(sequences)
    X_train, X_test, y_train, y_test = train_test_split(
        encoded_sequences, labels, test_size=test_fraction)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=validation_fraction)
    X_train = np.concatenate((X_train, reverse_complement(X_train)))
    y_train = np.concatenate((y_train, y_train))
    random_order = np.arange(len(X_train))
    np.random.shuffle(random_order)
    X_train = X_train[random_order]
    y_train = y_train[random_order]
    hyperparameters = {'seq_length': seq_length, 'use_RNN': use_RNN,
                       'num_filters': (45,), 'pool_width': 25, 'conv_width': (10,),
                       'L1': 0, 'dropout': 0.2, 'num_epochs': num_epochs}
    if use_deep_CNN:
        hyperparameters.update({'num_filters': (45, 50, 50), 'conv_width': (10, 8, 5)})
    if use_RNN:
        hyperparameters.update({'GRU_size': 35, 'TDD_size': 45})
    model = SequenceDNN(**hyperparameters)
    model.train(X_train, y_train, validation_data=(X_valid, y_valid))
    results = model.test(X_test, y_test).results[0]
    assert np.allclose(tuple(results.values()), tuple(golden_results.values())), \
        '{}: result = {}, golden = {}'.format(label, results, golden_results)


def test_shallow_CNN():
    run(use_deep_CNN=False, use_RNN=False, label='Shallow CNN',
        golden_results=OrderedDict((('Balanced accuracy', 50.0),
                                    ('auROC', 0.40625),
                                    ('auPRC', 0.32863613048887819),
                                    ('auPRG', 0.030076845553036049),
                                    ('Recall at 5% FDR', 0.0),
                                    ('Recall at 10% FDR', 0.0),
                                    ('Recall at 20% FDR', 0.0),
                                    ('Num Positives', 8),
                                    ('Num Negatives', 12))))


def test_deep_CNN():
    run(use_deep_CNN=True, use_RNN=False, label='Deep CNN',
        golden_results=OrderedDict((('Balanced accuracy', 50.0),
                                    ('auROC', 0.54166666666666663),
                                    ('auPRC', 0.69553449037775672),
                                    ('auPRG', -0.036049414600550951),
                                    ('Recall at 5% FDR', 16.666666666666664),
                                    ('Recall at 10% FDR', 16.666666666666664),
                                    ('Recall at 20% FDR', 16.666666666666664),
                                    ('Num Positives', 12),
                                    ('Num Negatives', 8))))


def test_RNN():
    run(use_deep_CNN=False, use_RNN=True, label='RNN',
        golden_results=OrderedDict((('Balanced accuracy', 65.384615384615387),
                                    ('auROC', 0.70329670329670335),
                                    ('auPRC', 0.82143508515114849),
                                    ('auPRG', 0.34909241712272016),
                                    ('Recall at 5% FDR', 30.76923076923077),
                                    ('Recall at 10% FDR', 30.76923076923077),
                                    ('Recall at 20% FDR', 30.76923076923077),
                                    ('Num Positives', 13),
                                    ('Num Negatives', 7))))


if __name__ == '__main__':
    pytest.main()