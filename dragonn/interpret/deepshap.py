import numpy as np
import shap
import tensorflow as tf
from deeplift.dinuc_shuffle import dinuc_shuffle

def combine_mult_and_diffref(mult, orig_inp, bg_data):
    to_return = []

    for l in [0]:
        projected_hypothetical_contribs = \
            np.zeros_like(bg_data[l]).astype("float")
        assert len(orig_inp[l].shape)==2

        # At each position in the input sequence, we iterate over the
        # one-hot encoding possibilities (eg: for genomic sequence, 
        # this is ACGT i.e. 1000, 0100, 0010 and 0001) and compute the
        # hypothetical difference-from-reference in each case. We then 
        # multiply the hypothetical differences-from-reference with 
        # the multipliers to get the hypothetical contributions. For 
        # each of the one-hot encoding possibilities, the hypothetical
        # contributions are then summed across the ACGT axis to 
        # estimate the total hypothetical contribution of each 
        # position. This per-position hypothetical contribution is then
        # assigned ("projected") onto whichever base was present in the
        # hypothetical sequence. The reason this is a fast estimate of
        # what the importance scores *would* look like if different 
        # bases were present in the underlying sequence is that the
        # multipliers are computed once using the original sequence, 
        # and are not computed again for each hypothetical sequence.
        for i in range(orig_inp[l].shape[-1]):
            hypothetical_input = np.zeros_like(orig_inp[l]).astype("float")
            hypothetical_input[:, i] = 1.0
            hypothetical_difference_from_reference = \
                (hypothetical_input[None, :, :] - bg_data[l])
            hypothetical_contribs = hypothetical_difference_from_reference * \
                                    mult[l]
            projected_hypothetical_contribs[:, :, i] = \
                np.sum(hypothetical_contribs, axis=-1) 

        to_return.append(np.mean(projected_hypothetical_contribs,axis=0))

    if len(orig_inp)>1:
        to_return.append(np.zeros_like(orig_inp[1]))

    return to_return


def shuffle_several_times(s):
    numshuffles=20
    if len(s)==2:
        return [np.array([dinuc_shuffle(s[0]) for i in range(numshuffles)]),
                np.array([s[1] for i in range(numshuffles)])]
    else:
        shuffled_seq = np.array([dinuc_shuffle(s[0]) for i in range(numshuffles)])
        #shuffled_seq = np.expand_dims(shuffled_seq,axis=1)
        return shuffled_seq


    
def deep_shap(model,seq,target_layer_idx=-2):
    int_model = tf.keras.Model(model.input, model.layers[target_layer_idx].output)
    inp = tf.keras.layers.Input(model.input.shape[2:])
    out = tf.reduce_sum(int_model(tf.expand_dims(inp,axis=1)),axis=1)
    new_model = tf.keras.Model(inp, out)
    
    
    deep_shap_explainer = shap.explainers.deep.TFDeepExplainer(
    (new_model.input, new_model.outputs[0]),
    shuffle_several_times,
    combine_mult_and_diffref=combine_mult_and_diffref)

    shap_scores = deep_shap_explainer.shap_values(
        seq[:,0,:,:], progress_message=100)

    return shap_scores
    