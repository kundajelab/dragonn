#utilities for running in-silico mutagenesis within dragonn.

def get_logit_function(model):
    from keras import backend as K
    inp=model.input
    outputs=model.layers[-2].output
    functor=K.function([inp], [outputs])
    return functor 

def get_logit(functor,X):
    logit=functor([X])
    return logit

def in_silico_mutagenesis(model, X):
    """                                                                                                                                                                                                     
    Parameters                                                                                                                                                                                              
    ----------                                                                                                                                                                                              
    model: keras model object                                                                                                                                                                               
    X: input matrix: (num_samples, 1, sequence_length,num_bases)                                                                                                                                            
    Returns                                                                                                                                                                                                      ---------                                                                                                                                                                                              
    (num_task, num_samples, sequence_length,num_bases) ISM score array.                                                                                                                                  
    """
    functor=get_logit_function(model)
    #1. get the wildtype predictions (n,1)                                                                                                                                                         
    wild_type_logits=np.asarray(get_logit(functor,X))
    wild_type_logits=np.squeeze(wild_type_logits,axis=2)
    #2. expand the wt array to dimensions: (num_samples,1,sequence_length)                                                                                                                         
    #Initialize mutants array to the same shape                                                                                                                                                              
    output_dim=wild_type_logits.shape+X.shape[2:4]
    wt_expanded=np.empty(output_dim)
    mutants_expanded=np.empty(output_dim)
    empty_onehot=np.zeros(output_dim[3])
    #3. Iterate through all tasks, positions                                                                                                                                                                 
    for task_index in range(output_dim[0]):
        for sample_index in range(output_dim[1]):
            print(str(task_index)+":"+str(sample_index))
            #fill in wild type logit values into an array of dim (task,sequence_length,num_bases)                                                                                                                    
            wt_logit_for_task_sample=wild_type_logits[task_index][sample_index]
            wt_expanded[task_index][sample_index]=np.tile(wt_logit_for_task_sample,(output_dim[2],output_dim[3]))
            #mutagenize each position                                                                                                                                                                            
            for base_pos in range(output_dim[2]):
                #for each position, iterate through the 4 bases                                                                                                                                                  
                for base_letter in range(output_dim[3]):
                    cur_base=empty_onehot
                    cur_base[base_letter]=1
                    Xtmp=np.expand_dims(X[sample_index],axis=0)
                    Xtmp[0][0][base_pos]=cur_base
                    #get the logit of Xtmp                                                                                                                                                                              
                    mutants_expanded[task_index][sample_index][base_pos][base_letter]=np.squeeze(get_logit(functor,Xtmp)[task_index])
    #subtract mutants_expanded from wt_expanded 
    ism_vals=wt_expanded-mutants_expanded
    #For each position subtract the mean ISM score for that position from each of the 4 values
    ism_vals_mean=np.expand_dims(np.mean(ism_vals,axis=3),axis=3)
    ism_vals_normed=ism_vals-ism_vals_mean
    return ism_vals_normed
