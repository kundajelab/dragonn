import deeplift
import numpy as np

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

def deeplift_shuffled_ref(X,score_func,batch_size=200,task_idx=0,num_refs_per_seq=10):
    deeplift_scores=score_func(
        task_idx=task_idx,
        input_data_sequences=X,
        num_refs_per_seq=num_refs_per_seq,
        batch_size=batch_size)
    return deeplift_scores

def get_deeplift_scoring_function(model,target_layer_idx=-2,task_idx=0, num_refs_per_seq=10,reference="shuffled_ref",one_hot_func=None):
    """
    Arguments: 
        model -- a string containing the path to the hdf5 exported model 
        target_layer_idx -- Layer in the model whose outputs will be interpreted. For classification models we \ 
                            interpret the logit (input to the sigmoid), which is the output of layer -2. 
                            For regression models we intepret the model output, which is the output of layer -1. 
        reference -- one of 'shuffled_ref','gc_ref','zero_ref'
        one_hot_func -- one hot function to use for encoding FASTA string inputs; if the inputs are already one-hot-encoded, use the default of None 
    Returns:
        deepLIFT scoring function 
    """
    assert reference in ["shuffled_ref","gc_ref","zero_ref"]
    from deeplift.conversion import kerasapi_conversion as kc
    deeplift_model = kc.convert_model_from_saved_files(model,verbose=False)

    #get the deeplift score with respect to the logit 
    score_func = deeplift_model.get_target_contribs_func(
        find_scores_layer_idx=0,
        target_layer_idx=target_layer_idx)
    if reference=="shuffled_ref":
        from deeplift.util import get_shuffle_seq_ref_function
        from deeplift.dinuc_shuffle import dinuc_shuffle        
        score_func=get_shuffle_seq_ref_function(
            score_computation_function=score_func,
            shuffle_func=dinuc_shuffle,
            one_hot_func=one_hot_func)
    return score_func
    
def deeplift(score_func, X, batch_size=200,task_idx=0, num_refs_per_seq=10,reference="shuffled_ref",one_hot_func=None):
    """
    Arguments: 
        score_func -- deepLIFT scoring function 
        X -- numpy array with shape (n_samples, 1, n_bases_in_sample,4) or list of FASTA sequences 
        batch_size -- number of samples to interpret at once 
        task_idx -- index indicating which task to perform interpretation on 
        reference -- one of 'shuffled_ref','gc_ref','zero_ref'
        num_refs_per_seq -- integer indicating number of references to use for each input sequence if the reference is set to 'shuffled_ref';if 'zero_ref' or 'gc_ref' is used, this argument is ignored.
        one_hot_func -- one hot function to use for encoding FASTA string inputs; if the inputs are already one-hot-encoded, use the default of None 
    Returns:
        (num_task, num_samples, 1, num_bases, sequence_length) deeplift score array.
    """
    assert reference in ["shuffled_ref","gc_ref","zero_ref"]
    if one_hot_func==None:
        #check that dataset has been one-hot-encoded
        assert len(np.shape(X)) == 4 and np.shape(X)[1] == 1
    if reference=="shuffled_ref":
        deeplift_scores=deeplift_shuffled_ref(X,score_func,batch_size,task_idx,num_refs_per_seq)
    elif reference=="gc_ref":
        deeplift_scores=deeplift_gc_ref(X,score_func,batch_size,task_idx)
    elif reference=="zero_ref":
        deeplift_scores=deeplift_zero_ref(X,score_func,batch_size,task_idx)
    else:
        raise Exception("supported DeepLIFT references are 'shuffled_ref','gc_ref', 'zero_ref'")
    return np.asarray(deeplift_scores)



