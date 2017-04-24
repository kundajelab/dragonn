from __future__ import absolute_import, division, print_function
import numpy as np, os, pytest, random, sys
os.environ["THEANO_FLAGS"] = "device=cpu"
np.random.seed(1)
random.seed(1)
from collections import OrderedDict
from dragonn.models import SequenceDNN
from dragonn.utils import one_hot_encode, reverse_complement
try:
    from sklearn.model_selection import train_test_split  # sklearn >= 0.18
except ImportError:
    from sklearn.cross_validation import train_test_split  # sklearn < 0.18

# define expected results for python2 and 3
if sys.version_info[0] == 2:
    golden_results_shallow_CNN = OrderedDict(
        [('Loss', 0.70371496533279465),
         ('Balanced accuracy', 55.639097744360896),
         ('auROC', 0.50877192982456143),
         ('auPRC', 0.58026674651508325),
         ('Recall at 5% FDR', 9.5238095238095237),
         ('Recall at 10% FDR', 9.5238095238095237),
         ('Recall at 20% FDR', 9.5238095238095237),
         ('Num Positives', 21),
         ('Num Negatives', 19)])
    golden_results_deep_CNN = OrderedDict(
        [('Loss', 0.68411321005526782),
         ('Balanced accuracy', 45.833333333333329),
         ('auROC', 0.51822916666666663),
         ('auPRC', 0.41738642611750432),
         ('Recall at 5% FDR', 0.0),
         ('Recall at 10% FDR', 0.0),
         ('Recall at 20% FDR', 0.0),
         ('Num Positives', 16),
         ('Num Negatives', 24)])
else:
    golden_results_shallow_CNN = OrderedDict(
        [('Loss', 0.81189359341516687),
         ('Balanced accuracy', 30.82706766917293),
         ('auROC', 0.20802005012531333),
         ('auPRC', 0.36561113802048162),
         ('Recall at 5% FDR', 0.0),
         ('Recall at 10% FDR', 0.0),
         ('Recall at 20% FDR', 0.0),
         ('Num Positives', 21),
         ('Num Negatives', 19)])
    golden_results_deep_CNN = OrderedDict(
        [('Loss', 0.7036767919621677),
         ('Balanced accuracy', 43.75),
         ('auROC', 0.43489583333333337),
         ('auPRC', 0.34594783625646253),
         ('Recall at 5% FDR', 0.0),
         ('Recall at 10% FDR', 0.0),
         ('Recall at 20% FDR', 0.0),
         ('Num Positives', 16),
         ('Num Negatives', 24)])


def run(use_deep_CNN, use_RNN, label, golden_results):
    seq_length = 100
    num_sequences = 200
    test_fraction = 0.2
    num_epochs = 1
    sequences = np.array([''.join(random.choice('ACGT') for base in range(seq_length)) for sequence in range(num_sequences)])
    labels = np.random.choice((True, False), size=num_sequences)[:, None]
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
        golden_results=golden_results_shallow_CNN)


def test_deep_CNN():
    run(use_deep_CNN=True, use_RNN=False, label='Deep CNN',
        golden_results=golden_results_deep_CNN)


if __name__ == '__main__':
    pytest.main()
