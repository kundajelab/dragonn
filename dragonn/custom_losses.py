import keras.backend as K
import pandas as pd 

#ambiguous values are indicated with =np.nan

def ambig_binary_crossentropy(y_true,y_pred):
    nonAmbig = K.cast(~pd.isnull(y_true),'float32')
    return K.mean(K.binary_crossentropy(y_true, y_pred)*nonAmbig, axis=-1);



def ambig_mean_squared_error(y_true, y_pred):
    nonAmbig = K.cast(~pd.isnull(y_true),'float32')
    return K.mean(K.square(y_pred - y_true)*nonAmbig, axis=-1)

