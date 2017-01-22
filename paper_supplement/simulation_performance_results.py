import argparse
from collections import OrderedDict
import cPickle as pickle
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import random
np.random.seed(1)
random.seed(1)
from sklearn.cross_validation import train_test_split
from simdna import simulations
from dragonn.utils import one_hot_encode, reverse_complement,get_motif_scores
from dragonn.models import SequenceDNN, RandomForest

parser = argparse.ArgumentParser()
parser.add_argument("--model-files-dir", required=True, help="Directory with architecture and weights files.")
parser.add_argument("--data-files-dir", required=True, help="Directory with simulation data files.")
parser.add_argument("--results-dir", required=True, help="Directory where results will be saved.")
parser.add_argument("--verbose", action='store_true', default=False, help="Use this to get more detailed logs.")
args = parser.parse_args()

# setup logging
log_formatter = logging.Formatter('%(levelname)s:%(asctime)s:%(name)s] %(message)s')
logger = logging.getLogger('dragonn')
handler = logging.StreamHandler()
if args.verbose:
    handler.setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)
else:
    handler.setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
handler.setFormatter(log_formatter)
logger.addHandler(handler)
logger.propagate = False

# create results directory
if not os.path.isdir(args.results_dir):
    os.makedirs(args.results_dir)

# define hyperparameter grid
train_set_sizes = range(1000, 13000, 1000)
pool_width_list = [5, 15, 25, 35, 45, 55]
conv_width_list_shallow = [(3,), (5,), (10,), (15,), (20,), (30,)]
conv_width_list_deep = [(3,3,3), (5,5,5), (10,10,10), (15,15,15), (20,20,20), (30,30,30)]
num_filters_list_shallow = [(1,),  (5,), (10,), (20,), (40,)]
num_filters_list_deep = [(1,1,1),  (5,5,5), (10,10,10), (20,20,20), (40,40,40)]

# define simulations functions and their arguments
simulation_func_args = OrderedDict()
simulation_func_args["simulate_single_motif_detection"] = {
    "motif_name": "TAL1_known4", "seq_length": 500, "GC_fraction": 0.4,
    "num_pos": 10000, "num_neg": 10000}
simulation_func_args["simulate_motif_counting"] = {
    "motif_name": "TAL1_known4", "seq_length": 1000, "GC_fraction": 0.4,
    "num_pos": 10000, "num_neg": 10000, "pos_counts": (3, 5), "neg_counts": (0, 2)}
simulation_func_args["simulate_motif_density_localization"] = {
    "motif_name": "TAL1_known4", "seq_length": 1000, "GC_fraction": 0.4,
    "num_pos": 10000, "num_neg": 10000, "min_motif_counts": 2, "max_motif_counts": 4, "center_size": 150}
simulation_func_args["simulate_multi_motif_embedding"] = {
    "motif_names": ["CTCF_known1", "ZNF143_known2", "SIX5_known1"],
    "seq_length": 500, "GC_fraction": 0.4, "num_seqs": 20000,
    "min_num_motifs": 0, "max_num_motifs": 3}
simulation_func_args["simulate_heterodimer_grammar"] = {
    "motif1": "SPI1_known4", "motif2": "IRF_known20", "seq_length": 500,
    "min_spacing": 2, "max_spacing": 5, "num_pos": 10000, "num_neg":10000, "GC_fraction":0.4}

simulation_motif_score_kwargs = OrderedDict()
simulation_motif_score_kwargs["simulate_single_motif_detection"] = {
    "motif_names": ["TAL1_known4"], "max_scores": 1, "return_positions": False, "GC_fraction": 0.4}
simulation_motif_score_kwargs["simulate_motif_counting"] = {
    "motif_names": ["TAL1_known4"], "max_scores": 5, "return_positions": False, "GC_fraction": 0.4}
simulation_motif_score_kwargs["simulate_motif_density_localization"] = {
    "motif_names": ["TAL1_known4"], "max_scores": 4, "return_positions": True, "GC_fraction": 0.4}
simulation_motif_score_kwargs["simulate_multi_motif_embedding"] = {
    "motif_names": ["CTCF_known1", "ZNF143_known2", "SIX5_known1"],
    "max_scores": 1, "return_positions": False, "GC_fraction": 0.4}
simulation_motif_score_kwargs["simulate_heterodimer_grammar"] = {
    "motif_names": ["SPI1_known4", "IRF_known20"],
    "max_scores": 1, "return_positions": True, "GC_fraction": 0.4}

def get_train_valid_test_data(simulation_func, prefix=None, test_size=0.2, valid_size=0.2, **kwargs):
    simulation_fname = ''.join('{}{}'.format(key, val) for key, val in sorted(kwargs.items()))
    simulation_fname = "{}{}.npz".format(prefix, simulation_fname)
    if prefix is not None:
        try:
            logger.debug("Checking for simulation data file {}...".format(simulation_fname) )
            data = np.load(simulation_fname)
            logger.debug("{} found. Loaded simulation data successfully!".format(simulation_fname))
            return ( data['X_train'], data['X_valid'], data['X_test'],
                     data['y_train'], data['y_valid'], data['y_test'])
        except:
            logger.debug("{} not found. Simulating data..".format(simulation_fname))
            pass

    sequences, y, embeddings = simulation_func(**kwargs)
    ( train_sequences, test_sequences,
      train_embeddings, test_embeddings,
      y_train, y_test ) = train_test_split(sequences, embeddings, y, test_size=test_size)
    ( train_sequences, valid_sequences,
      train_embeddings, valid_embeddings,
      y_train, y_valid ) = train_test_split(train_sequences, train_embeddings, y_train, test_size=valid_size)
    X_train = one_hot_encode(train_sequences)
    X_valid = one_hot_encode(valid_sequences)
    X_test = one_hot_encode(test_sequences)
    
    if prefix is not None:
        logger.debug("Saving simulated data to simulation_fname...".format(simulation_fname))
        np.savez_compressed(simulation_fname,
                            X_train=X_train, X_valid=X_valid, X_test=X_test,
                            train_embeddings=train_embeddings,
                            valid_embeddings=valid_embeddings,
                            test_embeddings=test_embeddings,
                            y_train=y_train, y_valid=y_valid, y_test=y_test)
    
    return ( X_train, X_valid, X_test,
             y_train, y_valid, y_test )

def dict2string(dictionary):
    return ''.join('{}{}'.format(key, str(val)) for key, val in sorted(dictionary.items()))

def train_test_dnn_vary_data_size(prefix, model_parameters=None,
                                  X_train=None, y_train=None,
                                  X_valid=None, y_valid=None,
                                  X_test=None, y_test=None,
                                  train_set_sizes=None):
    dnn_results = []
    for train_set_size in train_set_sizes:
        ofname_infix = dict2string(model_parameters)
        ofname_infix = "%s.train_set_size_%s" % (ofname_infix, str(train_set_size))
        ofname_prefix = "%s.%s" % (prefix, ofname_infix)
        model_fname = "%s.arch.json" % (ofname_prefix)
        weights_fname = "%s.weights.h5" % (ofname_prefix)
        try:
            logger.debug("Checking for model files {} and {}...".format(model_fname, weights_fname))
            best_dnn = SequenceDNN.load(model_fname, weights_fname)
            logger.debug("Model files found. Loaded model successfully!")
        except:
            logger.debug("Model files not found. Training model...")
            # try 3 attempts, take best auROC, save that model
            X_train_subset = X_train[:train_set_size]
            X_train_subset = np.concatenate((X_train_subset, reverse_complement(X_train_subset)))
            y_train_subset = np.concatenate((y_train[:train_set_size], y_train[:train_set_size]))
            best_auROC = 0
            best_dnn = None
            for random_seed in [1, 2, 3]:
                np.random.seed(random_seed)
                random.seed(random_seed)
                dnn = SequenceDNN(**model_parameters)
                logger.info("training with %i examples.." % (train_set_size))
                dnn.train(X_train_subset, y_train_subset, (X_valid, y_valid))
                result = dnn.test(X_test, y_test)
                auROCs = [result.results[i]["auROC"] for i in range(y_valid.shape[-1])]
                # get average auROC across tasks
                mean_auROC = sum(auROCs) / len(auROCs)
                if mean_auROC > best_auROC:
                    best_auROC = mean_auROC
                    dnn.save(ofname_prefix)
                    best_dnn = dnn
        dnn_results.append(best_dnn.test(X_test, y_test))
    # reset to original random seed
    np.random.seed(1)
    random.seed(1)
    return dnn_results

def train_test_rf_vary_data_size(prefix, motif_scoring_kwargs=None,
                                 X_train=None, y_train=None,
                                 X_valid=None, y_valid=None,
                                 X_test=None, y_test=None,
                                 train_set_sizes=None):
    motif_scores_train = get_motif_scores(X_train, **motif_scoring_kwargs)
    motif_scores_test = get_motif_scores(X_test, **motif_scoring_kwargs)
    rf_results = []
    for train_set_size in train_set_sizes:
        ofname_infix = dict2string(motif_scoring_kwargs)
        ofname_infix = "%s.train_set_size_%s" % (ofname_infix, str(train_set_size))
        ofname = "%s.%s.rf.pkl" % (prefix, ofname_infix)
        try:
            with open(ofname, 'rb') as fp:
                rf = pickle.load(fp)
        except: 
            logger.info("training with %i examples.." % (train_set_size))
            rf = RandomForest()
            rf.train(motif_scores_train[:train_set_size], y_train[:train_set_size].squeeze())
            with open(ofname, 'wb') as fid:
                pickle.dump(rf, fid)
        rf_results.append(rf.test(motif_scores_test, y_test))

    return rf_results

def train_test_dnn_vary_parameter(prefix,
                                  model_parameters,
                                  param_name,
                                  param_values,
                                  X_train=None, y_train=None,
                                  X_valid=None, y_valid=None,
                                  X_test=None, y_test=None):
    X_train = np.concatenate((X_train, reverse_complement(X_train)))
    y_train = np.concatenate((y_train, y_train))
    dnn_results = []
    for param_value in param_values:
        model_parameters[param_name] = param_value
        ofname_infix = dict2string(model_parameters)
        ofname_prefix = "%s.%s" % (prefix, ofname_infix)
        model_fname = "%s.arch.json" % (ofname_prefix)
        weights_fname = "%s.weights.h5" % (ofname_prefix)
        try:
            logger.debug("Checking for model files {} and {}...".format(model_fname, weights_fname))
            dnn = SequenceDNN.load(model_fname, weights_fname)
            logger.debug("Model files found. Loaded model successfully!")
        except:
            logger.debug("Model files not found. Training model...")
            dnn = SequenceDNN(**model_parameters)
            logger.info("training with %s %s .." % (param_name, param_value))
            dnn.train(X_train, y_train, (X_valid, y_valid))
            dnn.save(ofname_prefix)
        dnn_results.append(dnn.test(X_test, y_test))
        
    return dnn_results

# generate all results for all simulations
for simulation_func_name, kwargs in sorted(simulation_func_args.items()):
    simulation_results_fname = "{}/{}.results.pkl".format(args.results_dir, simulation_func_name)
    # check if simulations results are already saved
    try:
        logger.info("Checking for performance results file {}..".format(simulation_results_fname))
        with open(simulation_results_fname, 'rb') as fp:
            results = pickle.load(fp)
        logger.info("{} found. loaded performance results successfully!".format(simulation_results_fname)) 
        continue
    except:
        logger.info("{} not found. Solving simulation {}...".format(simulation_results_fname, simulation_func_name))
        pass
    # get simulation data
    logger.info("Getting simulation data for {} simulation_func_name...".format(simulation_func_name))
    simulation_func = getattr(simulations, simulation_func_name)
    ( X_train, X_valid, X_test,
      y_train, y_valid, y_test ) = get_train_valid_test_data(simulation_func, prefix=args.data_files_dir, **kwargs)
    # set reference model architecture
    if simulation_func_name=="simulate_heterodimer_grammar":
        model_parameters = {'seq_length': X_train.shape[-1], 'use_RNN': False, 'verbose': False,
                            'num_filters': (15, 15, 15), 'conv_width': (15, 15, 15), 'pool_width': 35,
                            'L1': 0, 'dropout': 0.0}
        conv_width_list = conv_width_list_deep
        num_filters_list = num_filters_list_deep
    else:
        model_parameters = {'seq_length': X_train.shape[-1], 'use_RNN': False,
                            'num_filters': (10,), 'conv_width': (15,), 'pool_width': 35, 'L1': 0, 'dropout': 0.0,
                            'verbose': False, 'num_tasks': len(y_valid[0])}
        conv_width_list = conv_width_list_shallow
        num_filters_list = num_filters_list_shallow
    # get performance vs data size
    logger.info("Getting model performance for varying training data size...")
    dnn_results = train_test_dnn_vary_data_size(prefix="{}/{}".format(args.model_files_dir, simulation_func_name),
                                                model_parameters=model_parameters,
                                                X_train=X_train, y_train=y_train,
                                                X_valid=X_valid, y_valid=y_valid,
                                                X_test=X_test, y_test=y_test,
                                                train_set_sizes=train_set_sizes)
    dnn_auROCs = [[res.results[i]["auROC"] for i in range(y_valid.shape[-1])] for res in dnn_results]
    logger.debug(dnn_auROCs)
    # repeat for motif score random forest benchmark
    logger.info("Getting motif random forest performance for varying training data size...")
    rf_results = train_test_rf_vary_data_size(
        prefix="{}/{}".format(args.results_dir, simulation_func_name),
        motif_scoring_kwargs=simulation_motif_score_kwargs[simulation_func_name], 
        X_train=X_train, y_train=y_train,
        X_valid=X_valid, y_valid=y_valid,
        X_test=X_test, y_test=y_test,
        train_set_sizes=train_set_sizes)
    rf_auROCs = [[res.results[i]["auROC"] for i in range(y_valid.shape[-1])] for res in rf_results]
    logger.debug(rf_auROCs)
    # get performance vs each architecture parameter
    logger.info("Getting model performance for varying pooling width...")
    dnn_pool_width_results = train_test_dnn_vary_parameter(prefix="{}/{}".format(args.model_files_dir, simulation_func_name),
                                                           model_parameters=model_parameters.copy(),
                                                           param_name="pool_width",
                                                           param_values=pool_width_list,
                                                           X_train=X_train, y_train=y_train,
                                                           X_valid=X_valid, y_valid=y_valid,
                                                           X_test=X_test, y_test=y_test)
    dnn_pool_width_auROCs = [[res.results[i]["auROC"] for i in range(y_valid.shape[-1])]
                             for res in dnn_pool_width_results]
    logger.debug(dnn_pool_width_auROCs)
    logger.info("Getting model performance for varying conv width...")
    dnn_conv_width_results = train_test_dnn_vary_parameter(prefix="{}/{}".format(args.model_files_dir, simulation_func_name),
                                                           model_parameters=model_parameters.copy(),
                                                           param_name="conv_width",
                                                           param_values=conv_width_list,
                                                           X_train=X_train, y_train=y_train,
                                                           X_valid=X_valid, y_valid=y_valid,
                                                           X_test=X_test, y_test=y_test)
    dnn_conv_width_auROCs = [[res.results[i]["auROC"] for i in range(y_valid.shape[-1])]
                             for res in dnn_conv_width_results]
    logger.debug(dnn_conv_width_auROCs)
    logger.info("Getting model performance for varying number of filters...")
    dnn_num_filters_results = train_test_dnn_vary_parameter(prefix="{}/{}".format(args.model_files_dir, simulation_func_name),
                                                            model_parameters=model_parameters.copy(),
                                                            param_name="num_filters",
                                                            param_values=num_filters_list,
                                                            X_train=X_train, y_train=y_train,
                                                            X_valid=X_valid, y_valid=y_valid,
                                                            X_test=X_test, y_test=y_test)
    dnn_num_filters_auROCs = [[res.results[i]["auROC"] for i in range(y_valid.shape[-1])]
                             for res in dnn_num_filters_results]
    logger.debug(dnn_num_filters_auROCs)
    # save all simulation results
    simulation_results = dict(zip(['train_set_sizes', 'dnn_auROCs', 'rf_auROCs',
                                   'pool_width_list', 'dnn_pool_width_auROCs',
                                   'conv_width_list', 'dnn_conv_width_auROCs',
                                   'num_filters_list', 'dnn_num_filters_auROCs'],
                                  [train_set_sizes, dnn_auROCs, rf_auROCs,
                                   pool_width_list, dnn_pool_width_auROCs,
                                   conv_width_list, dnn_conv_width_auROCs,
                                   num_filters_list, dnn_num_filters_auROCs]))
    with open(simulation_results_fname, 'wb') as fp:
        pickle.dump(simulation_results, fp)


def simpleaxis(ax): # removes top and right axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


# plot simulation performance results
xy_pairs = [{'x': 'train_set_sizes', 'y': 'rf_auROCs'},
            {'x': 'train_set_sizes', 'y': 'dnn_auROCs'},
            {'x': 'pool_width_list', 'y': 'dnn_pool_width_auROCs'},
            {'x': 'conv_width_list', 'y': 'dnn_conv_width_auROCs'},
            {'x': 'num_filters_list', 'y': 'dnn_num_filters_auROCs'}]
y2fname_infix = {'rf_auROCs': 'RandomForest',
                'dnn_auROCs': 'DNN',
                'dnn_pool_width_auROCs': 'pool_width',
                'dnn_conv_width_auROCs': 'conv_width',
                'dnn_num_filters_auROCs': 'num_filters'}
results_colors = {'simulate_single_motif_detection': ['b'],
                  'simulate_motif_counting': ['b'], 
                  'simulate_motif_density_localization': ['b'],
                  'simulate_multi_motif_embedding': ['k', 'r', 'g'],
                  'simulate_heterodimer_grammar': ['m']}
logger.info("Generating performance plots in {}".format(args.results_dir))
for simulation_func_name, kwargs in sorted(simulation_func_args.items()):
    simulation_results_fname ="{}/{}.results.pkl".format(args.results_dir, simulation_func_name)
    # load simulation results
    logger.debug("loading %s.." %(simulation_results_fname))
    with open(simulation_results_fname, 'rb') as fp:
        results = pickle.load(fp)
    # convert results to arrays
    for key, value in results.items():
        results[key] = np.array(value)
    for xy_pair in xy_pairs:
        for i in range(results[xy_pair['y']].shape[1]):
            plt.plot(results[xy_pair['x']], results[xy_pair['y']][:, i], c=results_colors[simulation_func_name][i])
        plt.ylim((0.4, 1))
        simpleaxis(plt.gca())
        plt.savefig("{}/{}.results.{}.pdf".format(args.results_dir, simulation_func_name, y2fname_infix[xy_pair['y']]),
                    format='pdf')
        plt.clf()
