import keras.backend as K
import tensorflow as tf 
import pandas as pd 

#ambiguous values are indicated with =np.nan
def ambig_mean_squared_error(y_true, y_pred):
    nonAmbig=tf.math.logical_not(tf.is_nan(y_true))
    return K.mean(K.square(tf.boolean_mask(y_pred,nonAmbig) - tf.boolean_mask(y_true,nonAmbig)), axis=-1)

def ambig_binary_crossentropy(y_true,y_pred):
    nonAmbig=tf.math.logical_not(tf.is_nan(y_true))
    return K.mean(K.binary_crossentropy(tf.boolean_mask(y_true,nonAmbig), tf.boolean_mask(y_pred,nonAmbig)), axis=-1);
