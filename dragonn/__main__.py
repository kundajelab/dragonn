import argparse
import numpy as np, random
np.random.seed(1)
random.seed(1)
from dragonn.utils import encode_fasta_sequences, get_sequence_strings
from dragonn.models import SequenceDNN
from sklearn.cross_validation import train_test_split

def parse_args():
    parser = argparse.ArgumentParser(
        description='main script for DragoNN modeling of sequence data.')
    # define parsers with common arguments
    fasta_pair_parser = argparse.ArgumentParser(add_help=False)
    fasta_pair_parser.add_argument('--pos-sequences', type=str, required=True,
                                   help='fasta with positive sequences')
    fasta_pair_parser.add_argument('--neg-sequences', type=str, required=True,
                                   help='fasta with negative sequences')
    single_fasta_parser = argparse.ArgumentParser(add_help=False)
    single_fasta_parser.add_argument('--sequences', type=str, required=True,
                                    help='fasta with sequences')
    prefix_parser = argparse.ArgumentParser(add_help=False)
    prefix_parser.add_argument('--prefix', type=str, required=True,
                               help='prefix to output files')
    model_files_parser = argparse.ArgumentParser(add_help=False)
    model_files_parser.add_argument('--model-file', type=str, required=True,
                                    help='model json file')
    model_files_parser.add_argument('--weights-file', type=str, required=True,
                                    help='weights hd5 file')
    # define commands 
    subparsers = parser.add_subparsers(help='dragonn command help', dest='command')
    train_parser = subparsers.add_parser('train',
                                         parents=[fasta_pair_parser, prefix_parser],
                                         help='model training help')
    train_parser.add_argument('--model-file', type=str, required=False,
                              help='model json file')
    train_parser.add_argument('--weights-file', type=str, required=False,
                              help='weights hd5 file')
    test_parser = subparsers.add_parser('test',
                                        parents=[fasta_pair_parser, model_files_parser],
                                        help='model testing help')
    predict_parser = subparsers.add_parser('predict',
                                           parents=[single_fasta_parser, model_files_parser],
                                           help='model prediction help')
    predict_parser.add_argument('--output-file', type=str, required=True,
                               help='output file for model predictions')
    interpret_parser = subparsers.add_parser('interpret',
                                             parents=[single_fasta_parser, model_files_parser, prefix_parser],
                                             help='model training help')
    interpret_parser.add_argument('--pos-threshold', type=int, default=0.5,
                               help='Only examples with predicted positive class probability above this get interpreted. Default: 0.5')
    # return command and command arguments
    args = vars(parser.parse_args())
    command = args.pop("command", None)
    if command == "train": # check for valid model and weights files
        if args["model_file"] is None and args["weights_file"] is not None:
            parser.error("You must provide a weights file corresponding to the provided model file! Exiting!")
        if args["weights_file"] is None and args["model_file"] is not None:
            parser.error("You must provide a model file corresponding to the provided weights file! Exiting!")
    return command, args


def main_train(pos_sequences=None,
               neg_sequences=None,
               prefix=None,
               model_file=None,
               weights_file=None):
    # encode fastas
    print("loading sequence data...")
    X_pos = encode_fasta_sequences(pos_sequences)
    y_pos = np.array([[True]]*len(X_pos))
    X_neg = encode_fasta_sequences(neg_sequences)
    y_neg = np.array([[False]]*len(X_neg))
    X = np.concatenate((X_pos, X_neg))
    y = np.concatenate((y_pos, y_neg))
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
    if model_file is not None and weights_file is not None: # load  model
        print("loading model...")
        model = SequenceDNN.load(model_file, weights_file)
    else: # initialize model
        print("initializing model...")
        model = SequenceDNN(seq_length=X_train.shape[-1])
    # train
    print("starting model training...")
    model.train(X_train, y_train, validation_data=(X_valid, y_valid))
    valid_result = model.test(X_valid, y_valid)
    print("final validation metrics:")
    print(valid_result)
    # save
    print("saving model files..")
    model.save("%s.model.json" % (prefix), "%s.weights.hd5" % (prefix))
    print("Done!")


def main_test(pos_sequences=None,
              neg_sequences=None,
              model_file=None,
              weights_file=None):
    # encode fastas
    print("loading sequence data...")
    X_test_pos = encode_fasta_sequences(pos_sequences)
    y_test_pos = np.array([[True]]*len(X_test_pos))
    X_test_neg = encode_fasta_sequences(neg_sequences)
    y_test_neg = np.array([[False]]*len(X_test_neg))
    X_test = np.concatenate((X_test_pos, X_test_neg))
    y_test = np.concatenate((y_test_pos, y_test_neg))
    # load model
    print("loading model...")
    model = SequenceDNN.load(model_file, weights_file)
    # test
    print("testing model...")
    test_result = model.test(X_test, y_test)
    print(test_result)


def main_predict(sequences=None,
                 model_file=None,
                 weights_file=None,
                 output_file=None):
    # encode fasta
    print("loading sequence data...")
    X = encode_fasta_sequences(sequences)
    # load model
    print("loading model...")
    model = SequenceDNN.load(model_file, weights_file)
    # predict
    print("getting predictions...")
    predictions = model.predict(X)
    # save predictions
    print("saving predictions to output file...")
    np.savetxt(output_file, predictions)
    print("Done!")


def main_interpret(sequences=None,
                   model_file=None,
                   weights_file=None,
                   pos_threshold=None,
                   peak_width=10,
                   prefix=None):
    # encode fasta
    print("loading sequence data...")
    X = encode_fasta_sequences(sequences)
    # load model
    print("loading model...")
    model = SequenceDNN.load(model_file, weights_file)
    # predict
    print("getting predictions...")
    predictions = model.predict(X)
    # deeplift
    print("getting deeplift scores...")
    deeplift_scores = model.deeplift(X)
    # get important sequences and write to file
    print("extracting important sequences and writing to file...")
    for task_index, task_scores in enumerate(deeplift_scores):
        peak_positions = []
        peak_sequences = []
        for sequence_index, sequence_scores in enumerate(task_scores):
            if predictions[sequence_index, task_index] > pos_threshold:
                #print(sequence_scores.shape)
                basewise_sequence_scores = sequence_scores.max(axis=(0,1))
                peak_position = basewise_sequence_scores.argmax()
                peak_positions.append(peak_position)
                peak_sequences.append(X[sequence_index : sequence_index + 1,
                                        :,
                                        :,
                                        peak_position - peak_width :
                                        peak_position + peak_width])
            else:
                peak_positions.append(-1)
                peak_sequences.append(np.zeros((1, 1, 4, 2 * peak_width)))
        peak_sequences = np.concatenate(peak_sequences)
        peak_sequence_strings = get_sequence_strings(peak_sequences)
        # write important sequences to file
        ofname = "%s.task_%i.important_sequences.txt" % (prefix, task_index)
        with open(ofname, "w") as wf:
            for i, peak_position in enumerate(peak_positions):
                wf.write("> sequence_%i\n" % (i))
                wf.write("%i: %s\n" %(peak_position, peak_sequence_strings[i]))
    print("Done!")


def main():
    command_functions = {'train': main_train,
                         'test': main_test,
                         'predict': main_predict,
                         'interpret': main_interpret}
    command, args = parse_args()
    command_functions[command](**args)
