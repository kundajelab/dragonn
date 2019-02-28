import keras.backend as K
import numpy as np 
def get_weighted_binary_crossentropy(w0_weights, w1_weights,ambig_val=np.nan):
    # Compute the task-weighted cross-entropy loss, where every task is weighted by 1 - (fraction of non-ambiguous examples that are positive)
    # In addition, weight everything with label -1 to 0
    import numpy as np 
    w0_weights=np.array(w0_weights);
    w1_weights=np.array(w1_weights);
    def weighted_binary_crossentropy(y_true,y_pred):
        weightsPerTaskRep = y_true*w1_weights[None,:] + (1-y_true)*w0_weights[None,:]
        nonAmbig = K.cast((y_true != ambig_val),'float32')
        nonAmbigTimesWeightsPerTask = nonAmbig * weightsPerTaskRep
        return K.mean(K.binary_crossentropy(y_true, y_pred)*nonAmbigTimesWeightsPerTask, axis=-1);
    return weighted_binary_crossentropy; 

def get_ambig_binary_crossentropy(ambig_val=np.nan):
    def ambig_binary_crossentropy(y_true,y_pred):
        nonAmbig = K.cast((y_true != ambig_val),'float32')
        return K.mean(K.binary_crossentropy(y_true, y_pred)*nonAmbig, axis=-1);
    return ambig_binary_crossentropy; 

def get_ambig_mean_squared_error(ambig_val=np.nan): 
    def ambig_mean_squared_error(y_true, y_pred):
        nonAmbig = K.cast((y_true != ambig_val),'float32')
        return K.mean(K.square(y_pred - y_true)*nonAmbig, axis=-1)
    return ambig_mean_squared_error
