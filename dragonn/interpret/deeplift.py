
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

