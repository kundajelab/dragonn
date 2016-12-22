from __future__ import absolute_import, division, print_function
import numpy as np, pytest, sys
sys.path.append('../dragonn')
from collections import OrderedDict
import os
os.environ["THEANO_FLAGS"] = "device=cpu"

def run(use_deep_CNN, use_RNN, label, golden_results):
    import random
    np.random.seed(1)
    random.seed(1)
    from dragonn.models import SequenceDNN
    from simdna.simulations import simulate_single_motif_detection
    from dragonn.utils import one_hot_encode, reverse_complement
    from sklearn.cross_validation import train_test_split
    seq_length = 100
    num_sequences = 200
    num_positives = 100
    num_negatives = num_sequences - num_positives
    GC_fraction = 0.4
    test_fraction = 0.2
    num_epochs = 1

    sequences, labels = simulate_single_motif_detection(
        'SPI1_disc1', seq_length, num_positives, num_negatives, GC_fraction)
    encoded_sequences = one_hot_encode(sequences)
    X_train, X_test, y_train, y_test = train_test_split(
        encoded_sequences, labels, test_size=test_fraction)
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
    model.train(X_train, y_train, validation_data=(X_test, y_test))
    results = model.test(X_test, y_test).results[0]
    assert np.allclose(tuple(results.values()), tuple(golden_results.values())), \
        '{}: result = {}, golden = {}'.format(label, results, golden_results)


def test_shallow_CNN():
    run(use_deep_CNN=False, use_RNN=False, label='Shallow CNN',
        golden_results=OrderedDict([('Loss', 0.87961949128096106),
                                    ('Balanced accuracy', 50.0),
                                    ('auROC', 0.4987212276214833),
                                    ('auPRC', 0.62190125670014862),
                                    ('Recall at 5% FDR', 13.043478260869565),
                                    ('Recall at 10% FDR', 13.043478260869565),
                                    ('Recall at 20% FDR', 13.043478260869565),
                                    ('Num Positives', 23),
                                    ('Num Negatives', 17)]))


def test_deep_CNN():
    run(use_deep_CNN=True, use_RNN=False, label='Deep CNN',
        golden_results=OrderedDict([('Loss', 0.78251894472793504),
                                    ('Balanced accuracy', 50.0),
                                    ('auROC', 0.40409207161125321),
                                    ('auPRC', 0.49822937725846766),
                                    ('Recall at 5% FDR', 0.0),
                                    ('Recall at 10% FDR', 0.0),
                                    ('Recall at 20% FDR', 0.0),
                                    ('Num Positives', 23),
                                    ('Num Negatives', 17)]))


if __name__ == '__main__':
    pytest.main()
