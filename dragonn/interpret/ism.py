#utilities for running in-silico mutagenesis within dragonn.
import numpy as np
from keras.models import Model

def get_preact_function(model,target_layer_idx):
        #load the model to predict preacts
        preact_model=Model(inputs=model.input,
                           outputs=model.layers[target_layer_idx].output)
        return preact_model.predict
def in_silico_mutagenesis(model, X, task_index,target_layer_idx=-2,start_pos=None,end_pos=None):
    """
    Parameters                               
    ----------                                
    model: keras model object
    X: input matrix: (num_samples, 1, sequence_length,num_bases)
    Returns
    ---------
    (num_task, num_samples, sequence_length,num_bases) ISM score array.
    """
    preact_function=get_preact_function(model,target_layer_idx)
    #1. get the wildtype predictions (n,1)    
    wild_type_logits=np.expand_dims(preact_function(X)[:,task_index],axis=1)
    
    #2. expand the wt array to dimensions: (n,1,sequence_length,num_bases)
    
    #Initialize mutants array to the same shape                                     
    output_dim=wild_type_logits.shape+X.shape[2:4]
    wt_expanded=np.zeros(output_dim)
    mutants_expanded=np.zeros(output_dim)
    empty_onehot=np.zeros(output_dim[3])
    if start_pos is None:
        start_pos=0
    if end_pos is None:
        end_pos=output_dim[2] 

    #3. Iterate through all tasks, positions
    for sample_index in range(output_dim[0]):
        print("ISM: task:"+str(task_index)+" sample:"+str(sample_index))
        #fill in wild type logit values into an array of dim (task,sequence_length,num_bases)
        wt_logit_for_task_sample=wild_type_logits[sample_index]
        wt_expanded[sample_index]=np.tile(wt_logit_for_task_sample,(output_dim[2],output_dim[3]))
        #mutagenize each position
        temp_batch = []
        tempbatch_baseposandletter = []
        for base_pos in range(start_pos,end_pos):
            #for each position, iterate through the 4 bases
            for base_letter in range(output_dim[3]):
                cur_base=np.array(empty_onehot)
                cur_base[base_letter]=1
                Xtmp=np.array(X[sample_index])
                Xtmp[0][base_pos]=cur_base
                temp_batch.append(Xtmp)
                tempbatch_baseposandletter.append((base_pos, base_letter))
        #get the logits of the batch
        batch_logits = preact_function([temp_batch]) 
        for logit,(base_pos, base_letter) in zip(batch_logits, tempbatch_baseposandletter):
            mutants_expanded[sample_index][0][base_pos][base_letter]=logit
                
    #subtract wt_expanded from mutants_expanded
    ism_vals=mutants_expanded-wt_expanded
    #For each position subtract the mean ISM score for that position from each of the 4 values
    ism_vals_mean=np.expand_dims(np.mean(ism_vals,axis=3),axis=3)
    ism_vals_normed=ism_vals-ism_vals_mean
    return ism_vals_normed
