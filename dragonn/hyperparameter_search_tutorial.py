#To prepare for model training, we import the necessary functions and submodules from keras
from keras.models import Sequential
from keras.layers.core import Dropout, Reshape, Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adadelta, SGD, RMSprop;
import keras.losses;
from keras.constraints import maxnorm;
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2
from keras.callbacks import EarlyStopping, History
from keras import backend as K
K.set_image_data_format('channels_last')

def hyperparam_get_model(input_shape,
              num_tasks=1,
              num_layers=1,
              num_filters=10,
              kernel_size=15,
              pool_size=35):
    '''
    Performs hyperparameter search in accordance with Figure 6 in the DragoNN manuscript 

    input_shape: tuple with dimensions of input 
    num_tasks: number of prediction tasks, default=1 
    num_layers: number of 2D convolution layers, default=1 
    num_filters: number of filters to use in each 2D convolution layer, default=10 
    kernel_size: filter dimension for each 2D convolution layer, (1,n), default n=15  
    pool_size: 2D MaxPooling pool size, default=35 

    Returns a Sequential model with the specified parameters. 
    '''
    model=Sequential()
    for i in range(num_layers):
        model.add(Conv2D(filters=num_filters,
                         kernel_size=(1,kernel_size),
                         input_shape=input_shape))
        model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(1,pool_size)))
    model.add(Flatten())
    model.add(Dense(num_tasks))
    model.add(Activation("sigmoid"))
    model.compile(optimizer="adam",loss="binary_crossentropy")
    return model

def hyperparam_train_model(data,num_training_examples,model):
    #train the model 
    history=model.fit(x=data.X_train[0:num_training_examples],
                      y=data.y_train[0:num_training_examples],
                      batch_size=128,
                      epochs=150,
                      verbose=0,
                      callbacks=[EarlyStopping(patience=7),
                                 History()],
                      validation_data=(data.X_valid,data.y_valid))
    #get auPRC on test set
    test_predictions=model.predict(data.X_test)
    test_performance=ClassificationResult(data.y_test,test_predictions)
    return test_performance.results['auROC']

