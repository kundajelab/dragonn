#utilities for running in-silico mutagenesis within dragonn.

def get_logit(model,X):
    from keras import backend as K 
    inp=model.input
    outputs=model.layers[-2].output
    functor=K.function([inp], [outputs])
    logit=functor([X])
    return logit 
    
def in_silico_mutagenesis(model, X):
    """
    Parameters
    ----------
    model: keras model object 
    X: input matrix: (num_samples, 1, num_bases, sequence_length) 
    Returns 
    ---------
    (num_task, num_samples, 1, num_bases, sequence_length) ISM score array.
    """    
    #1. get the wildtype predictions
    wild_type_logits=np.asarray(get_logit(model,X))
    #2. expand the wt array to dimensions: (num_samples,1,num_bases,sequence_length)
    #Initialize mutants array to the same shape
    output_dim=wild_type_logits.shape+X.shape[2,3]
    wt_expanded=np.empty(output_dim)
    mutants_expanded=np.empty(output_dim)

    empty_onehot=np.zeros(output_dim[4]) 
    #3. Iterate through all tasks, positions

    for sample_index in range(output_dim[1]):
        #fill in wild type logit values into an array of dim (num_bases, sequence_length)
        wt_logit_for_task_sample=wild_type_logits[:][sample_index] 
        wt_expanded[:][sample_index]=np.tile(wt_logit_for_task_sample,(output_dim[3],output_dim[4]))

        #mutagenize each position 
        for base_pos in range(output_dim[3]):
            #for each position, iterate through the 4 bases 
            for base_letter in range(output_dim[4]):
                cur_base=empty_onehot
                cur_base[base_letter]=1
                X[sample_index][:][base_pos]=cur_base
                #make prediction
                mutants_expanded[:][sample_index][base_pos][base_letter]=model.predict(np.expand_dims(X[sample_index],axis=0))
    
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
