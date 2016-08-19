from __future__ import absolute_import, division, print_function
import json, matplotlib, numpy as np, os, subprocess, tempfile
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from abc import abstractmethod, ABCMeta
from dragonn.metrics import ClassificationResult
from keras.models import Sequential
from keras.callbacks import Callback, EarlyStopping
from keras.layers.core import (
    Activation, Dense, Dropout, Flatten,
    Permute, Reshape, TimeDistributedDense
)
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.recurrent import GRU
from keras.regularizers import l1
from sklearn.svm import SVC as scikit_SVC
from sklearn.tree import DecisionTreeClassifier as scikit_DecisionTree
from sklearn.ensemble import RandomForestClassifier


class Model(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, **hyperparameters):
        pass

    @abstractmethod
    def train(self, X, y, validation_data):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def test(self, X, y):
        return ClassificationResult(y, self.predict(X))

    def score(self, X, y, metric):
        return self.test(X, y)[metric]


class SequenceDNN(Model):
    """
    Sequence DNN models.

    Parameters
    ----------
    seq_length : int
        length of input sequence.
    num_tasks : int,
        number of tasks. Default: 1.
    num_filters : list[int] | tuple[int]
        number of convolutional filters in each layer. Default: (15,).
    conv_width : list[int] | tuple[int]
        width of each layer's convolutional filters. Default: (15,).
    pool_width : int
        width of max pooling after the last layer. Default: 35.
    L1 : float
        strength of L1 penalty.
    dropout : float
        dropout probability in every convolutional layer. Default: 0.
    num_tasks : int
        Number of prediction tasks or labels. Default: 1.
    verbose: int
        Verbosity level during training. Valida values: 0, 1, 2.

    Returns
    -------
    Compiled DNN model.
    """

    class PrintMetrics(Callback):

        def __init__(self, validation_data, sequence_DNN):
            self.X_valid, self.y_valid = validation_data
            self.sequence_DNN = sequence_DNN

        def on_epoch_end(self, epoch, logs={}):
            print('Epoch {}: validation loss: {:.3f}\n{}\n'.format(
                epoch,
                logs['val_loss'],
                self.sequence_DNN.test(self.X_valid, self.y_valid)))

    class LossHistory(Callback):

        def __init__(self, X_train, y_train, validation_data, sequence_DNN):
            self.X_train = X_train
            self.y_train = y_train
            self.X_valid, self.y_valid = validation_data
            self.sequence_DNN = sequence_DNN
            self.train_losses = []
            self.valid_losses = []

        def on_epoch_end(self, epoch, logs={}):
            self.train_losses.append(self.sequence_DNN.model.evaluate(
                self.X_train, self.y_train, verbose=False))
            self.valid_losses.append(self.sequence_DNN.model.evaluate(
                self.X_valid, self.y_valid, verbose=False))

    def __init__(self, seq_length, use_RNN=False, num_tasks=1,
                 num_filters=(15, 15, 15), conv_width=(15, 15, 15),
                 pool_width=35, GRU_size=35, TDD_size=15,
                 L1=0, dropout=0.0, num_epochs=100, verbose=1):
        self.saved_params = locals()
        self.seq_length = seq_length
        self.input_shape = (1, 4, self.seq_length)
        self.num_tasks = num_tasks
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.model = Sequential()
        assert len(num_filters) == len(conv_width)
        for i, (nb_filter, nb_col) in enumerate(zip(num_filters, conv_width)):
            conv_height = 4 if i == 0 else 1
            self.model.add(Convolution2D(
                nb_filter=nb_filter, nb_row=conv_height,
                nb_col=nb_col, activation='linear',
                init='he_normal', input_shape=self.input_shape,
                W_regularizer=l1(L1), b_regularizer=l1(L1)))
            self.model.add(Activation('relu'))
            self.model.add(Dropout(dropout))
        self.model.add(MaxPooling2D(pool_size=(1, pool_width)))
        if use_RNN:
            num_max_pool_outputs = self.model.layers[-1].output_shape[-1]
            self.model.add(Reshape((num_filters[-1], num_max_pool_outputs)))
            self.model.add(Permute((2, 1)))
            self.model.add(GRU(GRU_size, return_sequences=True))
            self.model.add(TimeDistributedDense(TDD_size, activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(output_dim=self.num_tasks))
        self.model.add(Activation('sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy')
        self.train_losses = None
        self.valid_losses = None

    def train(self, X, y, validation_data):
        if y.dtype != bool:
            assert len(np.unique(y)) == 2
            y = y.astype(bool)
        multitask = y.shape[1] > 1
        if not multitask:
            num_positives = y.sum()
            num_sequences = len(y)
            num_negatives = num_sequences - num_positives
        self.callbacks = [EarlyStopping(monitor='val_loss', patience=5)]
        if self.verbose >= 1:
            self.callbacks.append(self.PrintMetrics(validation_data, self))
            print('Training model...')
        self.callbacks.append(self.LossHistory(X, y, validation_data, self))
        self.model.fit(
            X, y, batch_size=128, nb_epoch=self.num_epochs,
            validation_data=validation_data,
            class_weight={True: num_sequences / num_positives,
                          False: num_sequences / num_negatives}
            if not multitask else None,
            callbacks=self.callbacks, verbose=self.verbose >= 2)
        self.train_losses = self.callbacks[-1].train_losses
        self.valid_losses = self.callbacks[-1].valid_losses

    def predict(self, X):
        return self.model.predict(X, batch_size=128, verbose=False)

    def get_sequence_filters(self):
        """
        Returns 3D array of 2D sequence filters.
        """
        return self.model.layers[0].get_weights()[0].squeeze(axis=1)

    def deeplift(self, X, batch_size=200):
        """
        Returns (num_task, num_samples, 1, num_bases, sequence_length) deeplift score array.
        """
        assert len(np.shape(X)) == 4 and np.shape(X)[1] == 1
        from deeplift import keras_conversion as kc
        from deeplift.blobs import MxtsMode
        # normalize sequence convolution weights
        kc.mean_normalise_first_conv_layer_weights(self.model, None)
        # run deeplift
        deeplift_model = kc.convert_sequential_model(
            self.model, mxts_mode=MxtsMode.DeepLIFT)
        target_contribs_func = deeplift_model.get_target_contribs_func(
            find_scores_layer_idx=0)
        return np.asarray([
            target_contribs_func(task_idx=i, input_data_list=[X],
                                 batch_size=batch_size, progress_update=None)
            for i in range(self.num_tasks)])

    def in_silico_mutagenesis(self, X):
        """
        Returns (num_task, num_samples, 1, num_bases, sequence_length) ISM score array.
        """
        mutagenesis_scores = np.empty(
            X.shape + (self.num_tasks,), dtype=np.float32)
        wild_type_predictions = self.predict(X)
        wild_type_predictions = wild_type_predictions[
            :, np.newaxis, np.newaxis, np.newaxis]
        for sequence_index, (sequence, wild_type_prediction) in enumerate(
                zip(X, wild_type_predictions)):
            mutated_sequences = np.repeat(
                sequence[np.newaxis], np.prod(sequence.shape), axis=0)
            # remove wild-type
            arange = np.arange(len(mutated_sequences))
            horizontal_cycle = np.tile(
                np.arange(sequence.shape[-1]), sequence.shape[-2])
            mutated_sequences[arange, :, :, horizontal_cycle] = 0
            # add mutant
            vertical_repeat = np.repeat(
                np.arange(sequence.shape[-2]), sequence.shape[-1])
            mutated_sequences[arange, :, vertical_repeat, horizontal_cycle] = 1
            # make mutant predictions
            mutated_predictions = self.predict(mutated_sequences)
            mutated_predictions = mutated_predictions.reshape(
                sequence.shape + (self.num_tasks,))
            mutagenesis_scores[
                sequence_index] = wild_type_prediction - mutated_predictions
        return np.rollaxis(mutagenesis_scores, -1)

    @staticmethod
    def _plot_scores(X, output_directory, peak_width, score_func, score_name):
        from dragonn.plot import plot_bases_on_ax
        scores = score_func(X).squeeze(axis=2)  # (num_task, num_samples, num_bases, sequence_length)
        try:
            os.makedirs(output_directory)
        except OSError:
            pass
        num_tasks = len(scores)
        for task_index, task_scores in enumerate(scores):
            for sequence_index, sequence_scores in enumerate(task_scores):
                # sequence_scores is num_bases x sequence_length
                basewise_max_sequence_scores = sequence_scores.max(axis=0)
                plt.clf()
                figure, (top_axis, bottom_axis) = plt.subplots(2)
                top_axis.plot(range(1, len(basewise_max_sequence_scores) + 1),
                              basewise_max_sequence_scores)
                top_axis.set_title('{} scores (motif highlighted)'.format(score_name))
                peak_position = basewise_max_sequence_scores.argmax()
                top_axis.axvspan(peak_position - peak_width, peak_position + peak_width,
                                 color='grey', alpha=0.1)
                peak_sequence_scores = sequence_scores[:, peak_position - peak_width :
                                                          peak_position + peak_width].T
                # Set non-max letter_heights to zero
                letter_heights = np.zeros_like(peak_sequence_scores)
                letter_heights[np.arange(len(letter_heights)),
                               peak_sequence_scores.argmax(axis=1)] = \
                    basewise_max_sequence_scores[peak_position - peak_width :
                                                 peak_position + peak_width]
                plot_bases_on_ax(letter_heights, bottom_axis)
                bottom_axis.set_xticklabels(tuple(map(
                    str, np.arange(peak_position - peak_width, peak_position + peak_width + 1))))
                bottom_axis.tick_params(axis='x', labelsize='small')
                plt.xlabel('Position')
                plt.ylabel('Score')
                plt.savefig(os.path.join(output_directory, 'sequence_{}{}'.format(
                    sequence_index, '_task_{}'.format(task_index) if num_tasks > 1 else '')))
                plt.close()

    def plot_deeplift(self, X, output_directory, peak_width=10):
        self._plot_scores(X, output_directory, peak_width,
                          score_func=self.deeplift, score_name='DeepLift')

    def plot_in_silico_mutagenesis(self, X, output_directory, peak_width=10):
        self._plot_scores(X, output_directory, peak_width,
                          score_func=self.in_silico_mutagenesis, score_name='ISM')

    def plot_architecture(self, output_file):
        from dragonn.visualize_util import plot as plot_keras_model
        plot_keras_model(self.model, output_file, show_shape=True)

    def save(self, model_fname, weights_fname):
        if 'self' in self.saved_params:
            del self.saved_params['self']
        json.dump(self.saved_params, open(model_fname, 'wb'), indent=4)
        self.model.save_weights(weights_fname, overwrite=True)

    @staticmethod
    def load(model_fname, weights_fname):
        model_params = json.load(open(model_fname, 'rb'))
        sequence_dnn = SequenceDNN(**model_params)
        sequence_dnn.model.load_weights(weights_fname)
        return sequence_dnn

class MotifScoreRNN(Model):

    def __init__(self, input_shape, gru_size=10, tdd_size=4):
        self.model = Sequential()
        self.model.add(GRU(gru_size, return_sequences=True,
                           input_shape=input_shape))
        if tdd_size is not None:
            self.model.add(TimeDistributedDense(tdd_size))
        self.model.add(Flatten())
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))
        print('Compiling model...')
        self.model.compile(optimizer='adam', loss='binary_crossentropy')

    def train(self, X, y, validation_data):
        print('Training model...')
        multitask = y.shape[1] > 1
        if not multitask:
            num_positives = y.sum()
            num_sequences = len(y)
            num_negatives = num_sequences - num_positives
        self.model.fit(
            X, y, batch_size=128, nb_epoch=100,
            validation_data=validation_data,
            class_weight={True: num_sequences / num_positives,
                          False: num_sequences / num_negatives}
            if not multitask else None,
            callbacks=[EarlyStopping(monitor='val_loss', patience=10)],
            verbose=True)

    def predict(self, X):
        return self.model.predict(X, batch_size=128, verbose=False)


class gkmSVM(Model):

    def __init__(self, prefix='./gkmSVM', word_length=11, mismatches=3, C=1,
                 threads=1, cache_memory=100, verbosity=4):
        self.word_length = word_length
        self.mismatches = mismatches
        self.C = C
        self.threads = threads
        self.prefix = '_'.join(map(str, (prefix, word_length, mismatches, C)))
        options_list = zip(
            ['-l', '-d', '-c', '-T', '-m', '-v'],
            map(str, (word_length, mismatches, C, threads, cache_memory, verbosity)))
        self.options = ' '.join([' '.join(option) for option in options_list])

    @property
    def model_file(self):
        model_fname = '{}.model.txt'.format(self.prefix)
        return model_fname if os.path.isfile(model_fname) else None

    @staticmethod
    def encode_sequence_into_fasta_file(sequence_iterator, ofname):
        """writes sequences into fasta file
        """
        with open(ofname, "w") as wf:
            for i, seq in enumerate(sequence_iterator):
                print('>{}'.format(i), file=wf)
                print(seq, file=wf)

    def train(self, X, y, validation_data=None):
        """
        Trains gkm-svm, saves model file.
        """
        y = y.squeeze()
        pos_sequence = X[y]
        neg_sequence = X[~y]
        pos_fname = "%s.pos_seq.fa" % self.prefix
        neg_fname = "%s.neg_seq.fa" % self.prefix
        # create temporary fasta files
        self.encode_sequence_into_fasta_file(pos_sequence, pos_fname)
        self.encode_sequence_into_fasta_file(neg_sequence, neg_fname)
        # run command
        command = ' '.join(
            ('gkmtrain', self.options, pos_fname, neg_fname, self.prefix))
        process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        process.wait()  # wait for it to finish
        # remove fasta files
        os.system("rm %s" % pos_fname)
        os.system("rm %s" % neg_fname)

    def predict(self, X):
        if self.model_file is None:
            raise RuntimeError("GkmSvm hasn't been trained!")
        # write test fasta file
        test_fname = "%s.test.fa" % self.prefix
        self.encode_sequence_into_fasta_file(X, test_fname)
        # test gkmsvm
        temp_ofp = tempfile.NamedTemporaryFile()
        threads_option = '-T %s' % (str(self.threads))
        command = ' '.join(['gkmpredict',
                            test_fname,
                            self.model_file,
                            temp_ofp.name,
                            threads_option])
        process = subprocess.Popen(command, shell=True)
        process.wait()  # wait for it to finish
        os.system("rm %s" % test_fname)  # remove fasta file
        # get classification results
        temp_ofp.seek(0)
        y = np.array([line.split()[-1] for line in temp_ofp], dtype=float)
        temp_ofp.close()
        return np.expand_dims(y, 1)


class SVC(Model):

    def __init__(self):
        self.classifier = scikit_SVC(probability=True, kernel='linear')

    def train(self, X, y, validation_data=None):
        self.classifier.fit(X, y)

    def predict(self, X):
        return self.classifier.predict_proba(X)[:, 1:]


class DecisionTree(Model):

    def __init__(self):
        self.classifier = scikit_DecisionTree()

    def train(self, X, y, validation_data=None):
        self.classifier.fit(X, y)

    def predict(self, X):
        predictions = np.asarray(self.classifier.predict_proba(X))[..., 1]
        if len(predictions.shape) == 2:  # multitask
            predictions = predictions.T
        else:  # single-task
            predictions = np.expand_dims(predictions, 1)
        return predictions


class RandomForest(DecisionTree):

    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=100)
