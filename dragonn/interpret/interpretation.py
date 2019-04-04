from dragonn.tutorial_utils import *

#methods for interpretation of deep learning models
def interpret_filters(model,simulation_data):    
    print("Plotting simulation motifs...")
    plot_motifs(simulation_data)
    plt.show()
    print("Visualizing convolutional sequence filters in SequenceDNN...")
    plot_sequence_filters(model)
    plt.show()


def in_silico_mutagenesis(model, X):
    """
    Returns (num_task, num_samples, 1, num_bases, sequence_length) ISM score array.
    """
    #1. get the wildtype predictions
    wild_type_predictions=model.predict(X)
    #2. for each alternate allele at each position, compute model predictions
    
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

