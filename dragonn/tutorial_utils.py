from __future__ import division
import random
random.seed(1)
import inspect
from collections import namedtuple, defaultdict, OrderedDict
import numpy as np
np.random.seed(1)
from concise.utils.plot import seqlogo, seqlogo_fig    

try:
    from sklearn.model_selection import train_test_split  # sklearn >= 0.18
except ImportError:
    from sklearn.cross_validation import train_test_split  # sklearn < 0.18
from simdna import simulations
from simdna.synthetic import StringEmbeddable
from dragonn.utils import get_motif_scores, one_hot_encode
from dragonn.models import SequenceDNN
from dragonn.plot import add_letters_to_axis, plot_motif
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_learning_curve(history):
    train_losses=history.history['loss']
    valid_losses=history.history['val_loss']
    min_loss_indx = min(enumerate(valid_losses), key=lambda x: x[1])[0]
    f = plt.figure(figsize=(10, 4))
    ax = f.add_subplot(1, 1, 1)
    ax.plot(range(len(train_losses)), train_losses, 'b', label='Training',lw=4)
    ax.plot(range(len(train_losses)), valid_losses, 'r', label='Validation', lw=4)
    ax.plot([min_loss_indx, min_loss_indx], [0, 1.0], 'k--', label='Early Stop')
    ax.legend(loc="upper right")
    ax.set_ylabel("Loss")
    ax.set_ylim((min(train_losses+valid_losses),max(train_losses+valid_losses)))
    ax.set_xlabel("Epoch")
    plt.show()

def plot_ism(ism_mat,title,vmin=None,vmax=None):
    # create discrete colormap of ISM scores
    extent = [0, ism_mat.shape[0], 0, 100*ism_mat.shape[1]]
    plt.figure(figsize=(20,3))
    if vmin==None:
        vmin=np.amin(ism_mat)
    if vmax==None:
        vmax=np.amax(ism_mat)
    plt.imshow(ism_mat.T,extent=extent,vmin=vmin, vmax=vmax)
    plt.xlabel("Sequence base")
    plt.ylabel("ISM Score")
    plt.title(title)
    plt.yticks(np.arange(50,100*ism_mat.shape[1],100),("A","C","G","T"))
    plt.set_cmap('RdBu')
    plt.colorbar()
    plt.show()

    
def plot_sequence_filters(model):
    fig = plt.figure(figsize=(15, 8))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    conv_filters=model.layers[0].get_weights()[0]
    #transpose for plotting
    conv_filters=np.transpose(conv_filters,(3,1,2,0)).squeeze(axis=-1)
    num_plots_per_axis = int(len(conv_filters)**0.5) + 1
    for i, conv_filter in enumerate(conv_filters):
        ax = fig.add_subplot(num_plots_per_axis, num_plots_per_axis, i+1)
        add_letters_to_axis(ax, conv_filter)
        ax.axis("off")
        ax.set_title("Filter %s" % (str(i+1)))

def plot_seq_importance(grads, x, xlim=None, ylim=None, layer_idx=-2, figsize=(25, 3),title="",snp_pos=0):
    """Plot  sequence importance score
    
    Args:
      grads: either deeplift or gradientxinput score matrix 
      x: one-hot encoded DNA sequence
      xlim: restrict the plotted xrange
      figsize: matplotlib figure size
    """
    grads=grads.squeeze()
    x=x.squeeze()
    
    seq_len = x.shape[0]
    vals_to_plot=grads*x
    if xlim is None:
        xlim = (0, seq_len)
    if ylim is None:
        ylim= (np.amin(vals_to_plot),np.amax(vals_to_plot))
    seqlogo_fig(vals_to_plot, figsize=figsize)
    plt.xticks(list(range(xlim[0], xlim[1], 5)))
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title(title)
    plt.axvline(x=snp_pos, color='k', linestyle='--')
        

    
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


def inspect_SequenceDNN():
    print(inspect.getdoc(SequenceDNN))
    print("\nAvailable methods:\n")
    for (method_name, _) in inspect.getmembers(SequenceDNN, predicate=inspect.ismethod):
        if method_name != "__init__":
            print(method_name)


def get_SequenceDNN(SequenceDNN_parameters):
    return SequenceDNN(**SequenceDNN_parameters)


def train_SequenceDNN(dnn, simulation_data):
    assert issubclass(type(simulation_data), tuple)
    random.seed(1)
    np.random.seed(1)
    dnn.train(simulation_data.X_train, simulation_data.y_train,
              (simulation_data.X_valid, simulation_data.y_valid))

      
    
def SequenceDNN_learning_curve(dnn):
    if dnn.valid_metrics is not None:
        train_losses, valid_losses = [np.array([epoch_metrics['Loss'] for epoch_metrics in metrics])
                                      for metrics in (dnn.train_metrics, dnn.valid_metrics)]
        min_loss_indx = min(enumerate(valid_losses), key=lambda x: x[1])[0]
        f = plt.figure(figsize=(10, 4))
        ax = f.add_subplot(1, 1, 1)
        ax.plot(range(len(train_losses)), train_losses, 'b', label='Training',lw=4)
        ax.plot(range(len(train_losses)), valid_losses, 'r', label='Validation', lw=4)
        ax.plot([min_loss_indx, min_loss_indx], [0, 1.0], 'k--', label='Early Stop')
        ax.legend(loc="upper right")
        ax.set_ylabel("Loss")
        ax.set_ylim((0.0,1.0))
        ax.set_xlabel("Epoch")
        plt.show()
    else:
        print("learning curve can only be obtained after training!")


def test_SequenceDNN(dnn, simulation_data):
    print("Test performance:")
    print(dnn.test(simulation_data.X_test, simulation_data.y_test))


def plot_motifs(simulation_data):
    for motif_name in simulation_data.motif_names:
        plot_motif(motif_name, figsize=(10, 4), ylab=motif_name)




def plot_SequenceDNN_layer_outputs(dnn, simulation_data):
    # define layer out functions
    import theano
    get_conv_output = theano.function([dnn.model.layers[0].input],
                                      dnn.model.layers[0].get_output(train=False),
                                          allow_input_downcast=True)
    get_conv_relu_output = theano.function([dnn.model.layers[0].input],
                                            dnn.model.layers[1].get_output(train=False),
                                            allow_input_downcast=True)
    get_maxpool_output = theano.function([dnn.model.layers[0].input],
                                         dnn.model.layers[-4].get_output(train=False),
                                         allow_input_downcast=True)
    # get layer outputs for a positive simulation example
    pos_indx = np.where(simulation_data.y_valid==1)[0][0]
    pos_X = simulation_data.X_valid[pos_indx:(pos_indx+1)]
    conv_outputs = get_conv_output(pos_X).squeeze()
    conv_relu_outputs = get_conv_relu_output(pos_X).squeeze()
    maxpool_outputs = get_maxpool_output(pos_X).squeeze()
    # plot layer outputs
    fig = plt.figure(figsize=(15, 12))
    ax1 = fig.add_subplot(3, 1, 3)
    heatmap = ax1.imshow(conv_outputs, aspect='auto', interpolation='None', cmap='seismic')
    fig.colorbar(heatmap)
    ax1.set_ylabel("Convolutional Filters")
    ax1.set_xlabel("Position")
    ax1.get_yaxis().set_ticks([])
    ax1.get_xaxis().set_ticks([])
    ax1.set_title("SequenceDNN outputs from convolutional layer.\t\
    Locations of motif sites are highlighted in grey.")

    ax2 = fig.add_subplot(3, 1, 2)
    heatmap = ax2.imshow(conv_relu_outputs, aspect='auto', interpolation='None', cmap='seismic')
    fig.colorbar(heatmap)
    ax2.set_ylabel("Convolutional Filters")
    ax2.get_yaxis().set_ticks([])
    ax2.get_xaxis().set_ticks([])
    ax2.set_title("Convolutional outputs after ReLU transformation.\t\
    Locations of motif sites are highlighted in grey.")

    ax3 = fig.add_subplot(3, 1, 1)
    heatmap = ax3.imshow(maxpool_outputs, aspect='auto', interpolation='None', cmap='seismic')
    fig.colorbar(heatmap)
    ax3.set_title("DNN outputs after max pooling")
    ax3.set_ylabel("Convolutional Filters")
    ax3.get_yaxis().set_ticks([])
    ax3.get_xaxis().set_ticks([])

    # highlight motif sites
    motif_scores = get_motif_scores(pos_X, simulation_data.motif_names)
    motif_sites = [np.argmax(motif_scores[0, i, :]) for i in [0, 1]]
    for motif_site in motif_sites:
        conv_output_start = motif_site - max(dnn.conv_width-10, 0)
        conv_output_stop = motif_site + max(dnn.conv_width-10, 0)
        ax1.axvspan(conv_output_start, conv_output_stop, color='grey', alpha=0.5)
        ax2.axvspan(conv_output_start, conv_output_stop, color='grey', alpha=0.5)

def interpret_filters(model,simulation_data):    
    print("Plotting simulation motifs...")
    plot_motifs(simulation_data)
    plt.show()
    print("Visualizing convolutional sequence filters in SequenceDNN...")
    plot_sequence_filters(model)
    plt.show()
        
def interpret_SequenceDNN_filters(dnn, simulation_data):
    print("Plotting simulation motifs...")
    plot_motifs(simulation_data)
    plt.show()
    print("Visualizing convolutional sequence filters in SequenceDNN...")
    plot_sequence_filters(dnn)
    plt.show()


def interpret_data_with_SequenceDNN(dnn, simulation_data):
    # get a positive and a negative example from the simulation data
    pos_indx = np.flatnonzero(simulation_data.y_valid==1)[2]
    neg_indx = np.flatnonzero(simulation_data.y_valid==0)[2]
    pos_X = simulation_data.X_valid[pos_indx:pos_indx+1]
    neg_X = simulation_data.X_valid[neg_indx:neg_indx+1]
    # get motif scores, ISM scores, and DeepLIFT scores
    scores_dict = defaultdict(OrderedDict)
    scores_dict['Positive']['Motif Scores'] = get_motif_scores(pos_X, simulation_data.motif_names)
    scores_dict['Positive']['ISM Scores'] =  dnn.in_silico_mutagenesis(pos_X).max(axis=-2)
    scores_dict['Positive']['DeepLIFT Scores'] = dnn.deeplift(pos_X).max(axis=-2)
    scores_dict['Negative']['Motif Scores'] = get_motif_scores(neg_X, simulation_data.motif_names)
    scores_dict['Negative']['ISM Scores'] =  dnn.in_silico_mutagenesis(neg_X).max(axis=-2)
    scores_dict['Negative']['DeepLIFT Scores'] = dnn.deeplift(neg_X).max(axis=-2)

    # get motif site locations
    # motif_sites = {}
    # motif_sites['Positive'] = [np.argmax(scores_dict['Positive']['Motif Scores'][0, i, :])
    #                            for i in range(len(simulation_data.motif_names))]
    # motif_sites['Negative'] = [np.argmax(scores_dict['Negative']['Motif Scores'][0, i, :])
    #                            for i in range(len(simulation_data.motif_names))]
    motif_sites = {key: [embedded_motif.startPos + len(embedded_motif.what.string) // 2
                         for embedded_motif in
                         (next(embedded_motif for embedded_motif in
                               simulation_data.valid_embeddings[index]
                               if isinstance(embedded_motif.what, StringEmbeddable)
                               and motif_name in embedded_motif.what.stringDescription)
                          for motif_name in simulation_data.motif_names)]
                   for key, index in (('Positive', pos_indx), ('Negative', neg_indx))}
    # organize legends
    motif_label_dict = {}
    motif_label_dict['Motif Scores'] = simulation_data.motif_names
    if len(simulation_data.motif_names) == dnn.num_tasks:
        motif_label_dict['ISM Scores'] = simulation_data.motif_names
    else:
        motif_label_dict['ISM Scores'] = ['_'.join(simulation_data.motif_names)]
    motif_label_dict['DeepLIFT Scores'] = motif_label_dict['ISM Scores']
    # plot scores and highlight motif site locations
    seq_length = pos_X.shape[-1]
    plots_per_row = 2
    plots_per_column = 3
    ylim_dict = {'Motif Scores': (-80, 30), 'ISM Scores': (-1.5, 3.0), 'DeepLIFT Scores': (-1.5, 3.0)}
    motif_colors = ['b', 'r', 'c', 'm', 'g', 'k', 'y']
    font_size = 12
    num_x_ticks = 5
    highlight_width = 5
    motif_labels_cache = []

    f = plt.figure(figsize=(10,12))
    f.subplots_adjust(hspace=0.15, wspace=0.15)
    f.set_tight_layout(True)

    for j, key in enumerate(['Positive', 'Negative']):
        for i, (score_type, scores) in enumerate(scores_dict[key].items()):
            ax = f.add_subplot(plots_per_column, plots_per_row, plots_per_row*i+j+1)
            ax.set_ylim(ylim_dict[score_type])
            ax.set_xlim((0, seq_length))
            ax.set_frame_on(False)
            if j == 0: # put y axis and ticks only on left side
                xmin, xmax = ax.get_xaxis().get_view_interval()
                ymin, ymax = ax.get_yaxis().get_view_interval()
                ax.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2))
                ax.get_yaxis().tick_left()
                for tick in ax.yaxis.get_major_ticks():
                    tick.label.set_fontsize(font_size/1.5)
                ax.set_ylabel(score_type)
            if j > 0: # remove y axes
                ax.get_yaxis().set_visible(False)
            if i < (plots_per_column-1): # remove x axes
                ax.get_xaxis().set_visible(False)
            if i == (plots_per_column-1): # set x axis and ticks on bottom
                ax.set_xticks(seq_length/num_x_ticks*(np.arange(num_x_ticks+1)))
                xmin, xmax = ax.get_xaxis().get_view_interval()
                ymin, ymax = ax.get_yaxis().get_view_interval()
                ax.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
                ax.get_xaxis().tick_bottom()
                for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(font_size/1.5)
                ax.set_xlabel("Position")
            if j>0 and i<(plots_per_column-1): # remove all axes
                ax.axis('off')

            add_legend = False
            for _i, motif_label in enumerate(motif_label_dict[score_type]):
                if score_type=='Motif Scores':
                    scores_to_plot = scores[0, _i, :]
                else:
                    scores_to_plot = scores[0, 0, 0, :]
                if motif_label not in motif_labels_cache:
                    motif_labels_cache.append(motif_label)
                    add_legend = True
                motif_color = motif_colors[motif_labels_cache.index(motif_label)]
                ax.plot(scores_to_plot, label=motif_label, c=motif_color)
            if add_legend:
                leg = ax.legend(loc=[0,0.85], frameon=False, fontsize=font_size,
                                ncol=3, handlelength=-0.5)
                for legobj in leg.legendHandles:
                    legobj.set_color('w')
                for _j, text in enumerate(leg.get_texts()):
                    text_color = motif_colors[
                        motif_labels_cache.index(motif_label_dict[score_type][_j])]
                    text.set_color(text_color)
            for motif_site in motif_sites[key]:
                ax.axvspan(motif_site - highlight_width, motif_site + highlight_width,
                           color='grey', alpha=0.1)

def in_silico_mutagenesis(model, X):
    """
    Returns (num_task, num_samples, 1, num_bases, sequence_length) ISM score array.
    """
    mutagenesis_scores = np.empty(
        X.shape + (model.output_shape[1],), dtype=np.float32)
    wild_type_predictions = model.predict(X)
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
        mutated_predictions = model.predict(mutated_sequences)
        mutated_predictions = mutated_predictions.reshape(
            sequence.shape + (model.output_shape[1],))
        mutagenesis_scores[
            sequence_index] = wild_type_prediction - mutated_predictions
    mutagenesis_scores=np.rollaxis(mutagenesis_scores,-1)
    mutagenesis_scores=np.squeeze(mutagenesis_scores)
    #column-normalize the mutagenesis scores
    col_sum = mutagenesis_scores.sum(axis=0)
    normalized_mutagenesis_scores = (mutagenesis_scores)/col_sum
    return normalized_mutagenesis_scores
    

def input_grad(model,X,layer_idx=-2):
    from keras import backend as K 
    fn = K.function([model.input], K.gradients(model.layers[layer_idx].output, [model.input]))
    return fn([X])[0]


def deeplift_zero_ref(X,score_func,batch_size=200,task_idx=0):        
    # use a 40% GC reference
    input_references = [np.array([0.0, 0.0, 0.0, 0.0])[None, None, None, :]]
    # get deeplift scores
    
    deeplift_scores = score_func(
        task_idx=task_idx,
        input_data_list=[X],
        batch_size=batch_size,
        progress_update=None,
        input_references_list=input_references)
    return deeplift_scores



def deeplift_gc_ref(X,score_func,batch_size=200,task_idx=0):        
    # use a 40% GC reference
    input_references = [np.array([0.3, 0.2, 0.2, 0.3])[None, None, None, :]]
    # get deeplift scores
    
    deeplift_scores = score_func(
        task_idx=task_idx,
        input_data_list=[X],
        batch_size=batch_size,
        progress_update=None,
        input_references_list=input_references)
    return deeplift_scores

def deeplift_shuffled_ref(X,score_func,batch_size=200,task_idx=0,num_refs_per_seq=10,one_hot_func=None):
    from deeplift.util import get_shuffle_seq_ref_function
    from deeplift.dinuc_shuffle import dinuc_shuffle        
    score_func=get_shuffle_seq_ref_function(
        score_computation_function=score_func,
        shuffle_func=dinuc_shuffle,
        one_hot_func=one_hot_func)
    print("got score func!") 
    deeplift_scores=score_func(
        task_idx=task_idx,
        input_data_sequences=X,
        num_refs_per_seq=num_refs_per_seq,
        batch_size=batch_size)
    return deeplift_scores

def deeplift(model, X, batch_size=200,target_layer_idx=-2,task_idx=0, num_refs_per_seq=10,reference="shuffled_ref",one_hot_func=None):
    """
    Returns (num_task, num_samples, 1, num_bases, sequence_length) deeplift score array.
    """
    assert reference in ["shuffled_ref","gc_ref","zero_ref"]
    if one_hot_func==None:
        #check that dataset has been one-hot-encoded
        assert len(np.shape(X)) == 4 and np.shape(X)[1] == 1
    from deeplift.conversion import kerasapi_conversion as kc
    deeplift_model = kc.convert_model_from_saved_files(model,verbose=False)

    #get the deeplift score with respect to the logit 
    score_func = deeplift_model.get_target_contribs_func(
        find_scores_layer_idx=0,
        target_layer_idx=target_layer_idx)

    if reference=="shuffled_ref":
        deeplift_scores=deeplift_shuffled_ref(X,score_func,batch_size,task_idx,num_refs_per_seq,one_hot_func=one_hot_func)
    elif reference=="gc_ref":
        deeplift_scores=deeplift_gc_ref(X,score_func,batch_size,task_idx)
    elif reference=="zero_ref":
        deeplift_scores=deeplift_zero_ref(X,score_func,batch_size,task_idx)
    else:
        raise Exception("supported DeepLIFT references are 'shuffled_ref' and 'gc_ref'")
    return np.asarray(deeplift_scores)
