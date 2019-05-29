import keras.backend as K
import numpy as np 

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
