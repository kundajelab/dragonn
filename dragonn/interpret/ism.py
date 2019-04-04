#utilities for running in-silico mutagenesis within dragonn.

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
