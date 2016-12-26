from __future__ import absolute_import, division, print_function
import numpy as np, os, pytest, random
os.environ["THEANO_FLAGS"] = "device=cpu"
np.random.seed(1)
random.seed(1)
from collections import OrderedDict
from dragonn.models import SequenceDNN
from dragonn.utils import one_hot_encode, reverse_complement
from simdna.simulations import simulate_single_motif_detection
try:
    from sklearn.model_selection import train_test_split  # sklearn >= 0.18
except ImportError:
    from sklearn.cross_validation import train_test_split  # sklearn < 0.18


def run(use_deep_CNN, use_RNN, label, golden_first_sequence, golden_results):
    seq_length = 100
    num_sequences = 200
    num_positives = 100
    num_negatives = num_sequences - num_positives
    GC_fraction = 0.4
    test_fraction = 0.2
    num_epochs = 1
    sequences, labels, embeddings = simulate_single_motif_detection(
        'SPI1_disc1', seq_length, num_positives, num_negatives, GC_fraction)
    assert sequences[0] == golden_first_sequence, 'first sequence = {}, golden = {}'.format(
        sequences[0], golden_first_sequence)
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
        golden_first_sequence='ATTACCGTAATCTACTATTAAGTCACAACCAAACAATGGATTACTTTCTG'
                              'CGTTGGAATCAGTGCCGTGCCAATTGAAGGGGAAGTGCAGTATTTTTGGC',
        golden_results= OrderedDict((('Loss', 0.94293796771704286),
                                     ('Balanced accuracy', 50.0),
                                     ('auROC', 0.46035805626598464),
                                     ('auPRC', 0.5856166173549473),
                                     ('Recall at 5% FDR', 4.3478260869565215),
                                     ('Recall at 10% FDR', 4.3478260869565215),
                                     ('Recall at 20% FDR', 4.3478260869565215),
                                     ('Num Positives', 23),
                                     ('Num Negatives', 17))))


def test_deep_CNN():
    run(use_deep_CNN=True, use_RNN=False, label='Deep CNN',
        golden_first_sequence='CTCGAGGCAAGGTAATACGTTCGGGTAAAAACTCGCAAGTAGAACGAATA'
                              'TGAAGGGGAAGTTGTCACGCTAGAGAAAATACTTTCGACCTTGGATATGT',
        golden_results= OrderedDict((('Loss', 0.73795992129432741),
                                     ('Balanced accuracy', 50.0),
                                     ('auROC', 0.41066666666666674),
                                     ('auPRC', 0.58654343268480047),
                                     ('Recall at 5% FDR', 4.0),
                                     ('Recall at 10% FDR', 4.0),
                                     ('Recall at 20% FDR', 4.0),
                                     ('Num Positives', 25), ('Num Negatives', 15))))


if __name__ == '__main__':
    pytest.main()
